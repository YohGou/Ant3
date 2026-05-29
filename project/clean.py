"""
clean.py — Pipeline d'ingestion et nettoyage GeoPoliTrade
==========================================================
Sources :
  - data/historique_marche.xlsx   : cours OHLCV journaliers
  - data/portefeuille.xlsx        : positions et expositions géographiques
  - API GeoRisk Events            : événements géopolitiques par région

Sortie :
  - data/dataset_ml.parquet       : DataFrame prêt pour le modèle ML
  - data/dataset_ml.csv           : version CSV pour inspection

Usage :
  python src/clean.py
  python src/clean.py --date 2024-11-15   (backfill sur une date précise)
  python src/clean.py --output csv        (forcer sortie CSV)
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL  = "http://localhost:8000"
API_TIMEOUT   = 10        # secondes
API_DAYS      = 7         # fenêtre événements géopolitiques
DATA_DIR      = Path(__file__).parent.parent / "data"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── 1. Chargement des sources Excel ─────────────────────────────────────────

def load_historique(path: Path) -> pd.DataFrame:
    log.info(f"Chargement historique marché : {path}")
    df = pd.read_excel(path)
    # Normaliser les noms de colonnes
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"nom_action": "nom_action", "rendement_j1": "rendement_j1", "volatilite_5j": "volatilite_5j"})
    df["date"] = pd.to_datetime(df["date"])
    required = {"date","ticker","open","high","low","close","volume","rendement_j1","volatilite_5j"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans historique_marche : {missing}")
    log.info(f"  {len(df)} lignes, {df['ticker'].nunique()} actions, "
             f"période {df['date'].min().date()} → {df['date'].max().date()}")
    return df


def load_portefeuille(path: Path) -> pd.DataFrame:
    log.info(f"Chargement portefeuille : {path}")
    df = pd.read_excel(path, sheet_name="Portefeuille", dtype={"ticker": str})
    df.columns = [c.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("é","e").replace("è","e") for c in df.columns]
    df = df.rename(columns={"valeur_€": "valeur_eur", "poids_%": "poids_pct", "regions_exposees": "regions_exposees"})
    # Normaliser le nom de la colonne régions (robuste aux variantes)
    region_col = [c for c in df.columns if "region" in c]
    if region_col:
        df = df.rename(columns={region_col[0]: "regions_exposees"})
    df["regions_list"] = df["regions_exposees"].apply(
        lambda x: [r.strip() for r in str(x).split(",")]
    )
    log.info(f"  {len(df)} positions chargées")
    return df


# ─── 2. Appel API géopolitique ────────────────────────────────────────────────

def fetch_geopolitical_events(regions: list[str], days: int = API_DAYS) -> pd.DataFrame:
    log.info(f"Appel API GeoRisk pour {len(regions)} régions (fenêtre {days}j)...")
    all_events = []

    for region in regions:
        try:
            resp = requests.get(
                f"{API_BASE_URL}/events",
                params={"region": region, "days": days, "limit": 10},
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            for evt in data["events"]:
                all_events.append({
                    "region":       region,
                    "event_type":   evt["type"],
                    "impact_score": evt["impact_score"],
                    "confidence":   evt["confidence"],
                    "event_date":   evt["date"],
                })
            log.info(f"  {region:20s} : {data['total_events']} événements, "
                     f"impact moyen = {data['avg_impact']:+.2f}")
            time.sleep(0.1)

        except requests.exceptions.ConnectionError:
            log.warning(f"  API indisponible pour '{region}' — impact_score = 0 (fallback)")
            all_events.append({
                "region": region, "event_type": "unavailable",
                "impact_score": 0.0, "confidence": 0.0, "event_date": None,
            })
        except Exception as e:
            log.error(f"  Erreur API région '{region}' : {e}")
            all_events.append({
                "region": region, "event_type": "error",
                "impact_score": 0.0, "confidence": 0.0, "event_date": None,
            })

    df = pd.DataFrame(all_events)

    # Score agrégé par région (moyenne pondérée par confiance)
    geo_scores = (
        df.groupby("region")
        .apply(lambda g: np.average(g["impact_score"], weights=g["confidence"].clip(lower=0.01)))
        .reset_index()
        .rename(columns={0: "geo_impact_score"})
    )
    geo_scores["geo_impact_score"] = geo_scores["geo_impact_score"].round(3)

    event_counts = df.groupby("region").size().reset_index(name="nb_evenements")
    geo_scores = geo_scores.merge(event_counts, on="region")
    log.info(f"  Scores géopolitiques agrégés pour {len(geo_scores)} régions")
    return geo_scores


# ─── 3. Construction des features ─────────────────────────────────────────────

def build_features(df_hist: pd.DataFrame, df_ptf: pd.DataFrame,
                   df_geo: pd.DataFrame, target_date: datetime) -> pd.DataFrame:
    log.info(f"Construction des features pour la date cible : {target_date.date()}")

    # Filtrer sur les N derniers jours de trading disponibles
    window_start = target_date - timedelta(days=20)
    df_w = df_hist[(df_hist["date"] >= window_start) & (df_hist["date"] <= target_date)].copy()

    # Features de marché par ticker (fenêtre récente)
    mkt_features = (
        df_w.groupby("ticker")
        .agg(
            close_last=      ("close",        "last"),
            rendement_moyen= ("rendement_j1", "mean"),
            volatilite_moy=  ("volatilite_5j","mean"),
            volume_moyen=    ("volume",        "mean"),
            rendement_j1=    ("rendement_j1", "last"),
            volatilite_5j=   ("volatilite_5j","last"),
        )
        .reset_index()
    )
    mkt_features["rendement_moyen"] = mkt_features["rendement_moyen"].round(4)
    mkt_features["volatilite_moy"]  = mkt_features["volatilite_moy"].round(4)

    # Momentum : rendement sur 5 jours glissants
    df_sorted = df_w.sort_values(["ticker","date"])
    momentum = (
        df_sorted.groupby("ticker")
        .apply(lambda g: g.tail(5)["rendement_j1"].sum())
        .reset_index()
        .rename(columns={0: "momentum_5j"})
    )
    mkt_features = mkt_features.merge(momentum, on="ticker")
    mkt_features["momentum_5j"] = mkt_features["momentum_5j"].round(4)

    # Jointure avec portefeuille
    df_ptf_slim = df_ptf[["ticker","nom_action","secteur","regions_list",
                           "poids_pct","seuil_risque","profil"]].copy()
    df = mkt_features.merge(df_ptf_slim, on="ticker", how="inner")

    # Calcul du risk_score géopolitique composite par action
    def compute_geo_score(regions_list):
        scores = []
        for region in regions_list:
            match = df_geo[df_geo["region"] == region]["geo_impact_score"]
            if not match.empty:
                scores.append(match.values[0])
        return round(np.mean(scores), 3) if scores else 0.0

    df["geo_risk_score"]   = df["regions_list"].apply(compute_geo_score)
    df["nb_regions_expo"]  = df["regions_list"].apply(len)

    # Feature composite : risque pondéré volatilité × géopolitique
    df["risk_composite"] = (
        df["volatilite_moy"] * 0.4 + abs(df["geo_risk_score"]) * 0.6
    ).round(4)

    # Nettoyage : supprimer les lignes sans données de marché
    n_before = len(df)
    df = df.dropna(subset=["close_last","rendement_j1","volatilite_5j"])
    if len(df) < n_before:
        log.warning(f"  {n_before - len(df)} lignes supprimées (valeurs manquantes)")

    # Ajout métadonnées pipeline
    df["date_prediction"] = target_date.strftime("%Y-%m-%d")
    df["pipeline_run_at"] = datetime.now().isoformat()

    # Suppression colonne liste (non sérialisable en parquet facilement)
    df = df.drop(columns=["regions_list"])

    log.info(f"  DataFrame final : {len(df)} lignes × {len(df.columns)} colonnes")
    return df


# ─── 4. Export ────────────────────────────────────────────────────────────────

def export(df: pd.DataFrame, output_format: str = "parquet") -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = DATA_DIR / "dataset_ml.parquet"
    csv_path     = DATA_DIR / "dataset_ml.csv"

    df.to_parquet(parquet_path, index=False)
    log.info(f"Exporté → {parquet_path}")

    if output_format == "csv" or True:   # toujours exporter le CSV pour inspection
        df.to_csv(csv_path, index=False, sep=";")
        log.info(f"Exporté → {csv_path}")


# ─── 5. Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline GeoPoliTrade — clean.py")
    parser.add_argument("--date",   default=None, help="Date cible YYYY-MM-DD (défaut: aujourd'hui)")
    parser.add_argument("--output", default="parquet", choices=["parquet","csv"])
    args = parser.parse_args()

    target_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime(2024, 11, 15)

    log.info("=" * 60)
    log.info("GeoPoliTrade — Pipeline de nettoyage démarré")
    log.info(f"Date cible : {target_date.date()}")
    log.info("=" * 60)

    try:
        df_hist = load_historique(DATA_DIR / "historique_marche.xlsx")
        df_ptf  = load_portefeuille(DATA_DIR / "portefeuille.xlsx")

        all_regions = sorted(set(
            r.strip()
            for regions_str in df_ptf["regions_exposees"]
            for r in str(regions_str).split(",")
        ))
        log.info(f"Régions à requêter : {all_regions}")

        df_geo = fetch_geopolitical_events(all_regions)
        df_ml  = build_features(df_hist, df_ptf, df_geo, target_date)

        export(df_ml, args.output)

        log.info("=" * 60)
        log.info("Pipeline terminé avec succès")
        log.info(f"Colonnes du DataFrame ML : {list(df_ml.columns)}")
        log.info("=" * 60)
        print(df_ml.to_string(max_rows=10, max_cols=8))

    except Exception as e:
        log.error(f"Erreur pipeline : {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
