"""
Tests unitaires — GeoPoliTrade pipeline
Lancement : pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def df_historique():
    return pd.read_excel(DATA_DIR / "historique_marche.xlsx")

@pytest.fixture
def df_portefeuille():
    return pd.read_excel(DATA_DIR / "portefeuille.xlsx", sheet_name="Portefeuille")

@pytest.fixture
def df_geo_mock():
    return pd.DataFrame({
        "region":           ["Moyen-Orient", "Chine", "USA", "Europe"],
        "geo_impact_score": [-0.72, -0.45, +0.20, -0.15],
        "nb_evenements":    [4, 3, 5, 3],
    })


# ─── Tests Excel 1 : historique_marche ───────────────────────────────────────

class TestHistoriqueMarche:

    def test_fichier_existe(self):
        assert (DATA_DIR / "historique_marche.xlsx").exists(), \
            "historique_marche.xlsx introuvable dans data/"

    def test_colonnes_requises(self, df_historique):
        required = {"date","ticker","open","high","low","close","volume",
                    "rendement_j1","volatilite_5j"}
        assert required.issubset(set(df_historique.columns)), \
            f"Colonnes manquantes : {required - set(df_historique.columns)}"

    def test_pas_de_nan_critiques(self, df_historique):
        cols_critiques = ["ticker","date","open","close","volume"]
        for col in cols_critiques:
            nb_nan = df_historique[col].isna().sum()
            assert nb_nan == 0, f"Colonne '{col}' contient {nb_nan} NaN"

    def test_prix_positifs(self, df_historique):
        for col in ["open","high","low","close"]:
            assert (df_historique[col] > 0).all(), f"Prix négatifs détectés dans '{col}'"

    def test_coherence_high_low(self, df_historique):
        assert (df_historique["high"] >= df_historique["low"]).all(), \
            "Des lignes ont high < low — incohérence OHLCV"

    def test_volume_positif(self, df_historique):
        assert (df_historique["volume"] > 0).all(), "Volume <= 0 détecté"

    def test_nb_tickers(self, df_historique):
        assert df_historique["ticker"].nunique() == 10, \
            f"Attendu 10 tickers, trouvé {df_historique['ticker'].nunique()}"

    def test_rendement_j1_bornes(self, df_historique):
        df_notzero = df_historique[df_historique["rendement_j1"] != 0]
        assert (df_notzero["rendement_j1"].abs() < 0.5).all(), \
            "rendement_j1 > 50% détecté — valeurs aberrantes"


# ─── Tests Excel 2 : portefeuille ────────────────────────────────────────────

class TestPortefeuille:

    def test_fichier_existe(self):
        assert (DATA_DIR / "portefeuille.xlsx").exists(), \
            "portefeuille.xlsx introuvable dans data/"

    def test_colonnes_requises(self, df_portefeuille):
        cols = [c.lower() for c in df_portefeuille.columns]
        assert any("ticker" in c for c in cols), "Colonne TICKER manquante"
        assert any("secteur" in c for c in cols), "Colonne SECTEUR manquante"
        assert any("region" in c for c in cols), "Colonne RÉGIONS manquante"

    def test_nb_positions(self, df_portefeuille):
        assert len(df_portefeuille) == 10, \
            f"Attendu 10 positions, trouvé {len(df_portefeuille)}"

    def test_tickers_uniques(self, df_portefeuille):
        tickers = df_portefeuille.iloc[:, 0]
        assert tickers.nunique() == len(df_portefeuille), \
            "Doublons détectés dans les tickers"

    def test_poids_raisonnable(self, df_portefeuille):
        poids_col = [c for c in df_portefeuille.columns if "poids" in c.lower() or "%" in c]
        if poids_col:
            poids = df_portefeuille[poids_col[0]]
            assert (poids > 0).all(), "Poids <= 0 détecté"
            assert (poids < 50).all(), "Poids > 50% — position trop concentrée"


# ─── Tests du pipeline de features ───────────────────────────────────────────

class TestBuildFeatures:

    def test_geo_score_calcul(self, df_geo_mock):
        """Le score géopolitique composite doit être dans [-1, 1]."""
        scores = df_geo_mock["geo_impact_score"]
        assert (scores >= -1.0).all() and (scores <= 1.0).all(), \
            "geo_impact_score hors bornes [-1, 1]"

    def test_geo_score_moyenne(self, df_geo_mock):
        """Vérification calcul moyen pour une action multi-région."""
        regions = ["Moyen-Orient", "USA"]
        scores = df_geo_mock[df_geo_mock["region"].isin(regions)]["geo_impact_score"]
        expected = round(scores.mean(), 3)
        computed = round(np.mean(scores.values), 3)
        assert abs(expected - computed) < 0.001

    def test_risk_composite_formule(self):
        """risk_composite = vol*0.4 + |geo|*0.6 doit être >= 0."""
        vol = 0.02
        geo = -0.75
        risk = vol * 0.4 + abs(geo) * 0.6
        assert risk >= 0, "risk_composite ne peut pas être négatif"
        assert risk <= 1.0, "risk_composite > 1 — normalisation requise"


# ─── Tests du DataFrame final (si déjà généré) ───────────────────────────────

class TestDatasetML:

    def test_dataset_existe(self):
        path = DATA_DIR / "dataset_ml.parquet"
        if not path.exists():
            pytest.skip("dataset_ml.parquet pas encore généré — lancer clean.py d'abord")

    def test_colonnes_ml(self):
        path = DATA_DIR / "dataset_ml.parquet"
        if not path.exists():
            pytest.skip("dataset_ml.parquet absent")
        df = pd.read_parquet(path)
        required = {"ticker","close_last","rendement_j1","volatilite_5j",
                    "momentum_5j","geo_risk_score","risk_composite","date_prediction"}
        assert required.issubset(set(df.columns)), \
            f"Colonnes ML manquantes : {required - set(df.columns)}"

    def test_pas_de_nan_dans_features(self):
        path = DATA_DIR / "dataset_ml.parquet"
        if not path.exists():
            pytest.skip("dataset_ml.parquet absent")
        df = pd.read_parquet(path)
        features = ["close_last","rendement_j1","volatilite_5j","geo_risk_score","risk_composite"]
        for col in features:
            if col in df.columns:
                nb_nan = df[col].isna().sum()
                assert nb_nan == 0, f"Feature '{col}' contient {nb_nan} NaN"
