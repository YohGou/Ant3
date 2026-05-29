"""
API mock géopolitique — GeoRisk Events API v1.0
Simule une API tierce retournant des événements géopolitiques récents par région.

Lancement : uvicorn api.geopolitical_api:app --reload --port 8000
Docs auto  : http://localhost:8000/docs
"""

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import random
import hashlib

app = FastAPI(
    title="GeoRisk Events API",
    description="API fictive de scoring géopolitique pour le projet GeoPoliTrade",
    version="1.0.0",
)

# ─── Données fictives stables (seed par région+date) ─────────────────────────

EVENT_TEMPLATES = {
    "Moyen-Orient": [
        ("conflit",   "Escalade militaire en mer Rouge — détroit de Bab-el-Mandeb",     -0.85),
        ("sanction",  "Nouvelles sanctions US sur exportations pétrolières iraniennes",  -0.70),
        ("conflit",   "Frappes aériennes signalées dans la région du Golfe",             -0.90),
        ("diplomatie","Accord de cessez-le-feu négocié sous égide ONU",                  +0.55),
        ("économique","Hausse de la production OPEC+ de 500k barils/jour",               -0.30),
    ],
    "Russie": [
        ("sanction",  "UE annonce 14ème paquet de sanctions sur énergie russe",          -0.80),
        ("conflit",   "Intensification des opérations militaires en Ukraine orientale",  -0.75),
        ("économique","Rouble en chute de 8% suite aux restrictions SWIFT",              -0.60),
        ("diplomatie","Pourparlers de paix à Istanbul — médiation turque",               +0.40),
        ("sanction",  "Gel d'avoirs supplémentaires de 12 milliards EUR en Europe",      -0.65),
    ],
    "Chine": [
        ("tension",   "Exercices militaires PLA autour de Taïwan — 72h d'activité",      -0.80),
        ("sanction",  "Washington restreint exports semiconducteurs avancés vers Chine",  -0.70),
        ("économique","PIB chinois Q3 : +4.7% vs +5.2% attendu — déception marchés",    -0.45),
        ("diplomatie","Sommet Biden-Xi : accord commercial partiel sur terres rares",    +0.60),
        ("tension",   "Incident naval en mer de Chine méridionale — îles Spratleys",    -0.55),
    ],
    "USA": [
        ("politique", "Résultats électoraux incertains — recomptage dans 3 États clés",  -0.40),
        ("économique","Fed maintient taux directeur à 5.25% — discours hawkish",         -0.35),
        ("politique", "Accord bipartisan sur le plafond de la dette voté au Congrès",    +0.50),
        ("économique","Rapport emploi NFP : +280k jobs, taux chômage 3.8%",              +0.30),
        ("sanction",  "Nouvelles sanctions secondaires contre partenaires de Moscou",    -0.25),
    ],
    "Europe": [
        ("politique", "Elections européennes : montée des partis eurosceptiques",        -0.35),
        ("économique","BCE annonce réduction du bilan de 200Md€ sur 6 mois",            -0.30),
        ("diplomatie","Accord de défense UE-OTAN renforcé — budget commun +40%",        +0.25),
        ("économique","Inflation zone euro : 2.3% — retour vers cible BCE",              +0.40),
        ("politique", "Crise gouvernementale en France — vote de confiance",             -0.45),
    ],
    "Asie": [
        ("tension",   "Tensions frontalières Inde-Pakistan — ligne de contrôle",        -0.60),
        ("économique","Japon en récession technique — 2 trimestres négatifs",           -0.50),
        ("diplomatie","ASEAN+ signe accord de libre-échange avec UE",                   +0.45),
        ("sanction",  "Restrictions sur exportations de puces mémoire coréennes",       -0.40),
        ("économique","Boom de l'IA en Asie du Sud-Est — investissements +35%",         +0.55),
    ],
    "Afrique": [
        ("politique", "Coup d'État au Sahel — troisième en 18 mois dans la région",     -0.75),
        ("économique","Découverte majeure de lithium en RDC — accord avec UE",          +0.50),
        ("conflit",   "Reprise des combats dans l'est du Congo malgré accord paix",     -0.65),
        ("diplomatie","Sommet UA : accord monnaie commune zone CFA 2027",               +0.30),
        ("économique","Sécheresse critique en Afrique australe — crise alimentaire",    -0.55),
    ],
    "Amérique Latine": [
        ("politique", "Crise institutionnelle au Venezuela — élections contestées",     -0.60),
        ("économique","Brésil relève taux SELIC à 11.75% — lutte inflation",           -0.35),
        ("diplomatie","Accord Mercosur-UE ratifié après 25 ans de négociations",       +0.65),
        ("économique","Production cuivre Chili en baisse 15% — grèves minières",       -0.45),
        ("politique", "Mexique : réforme judiciaire controversée adoptée",              -0.40),
    ],
}

class GeopoliticalEvent(BaseModel):
    event_id:     str
    region:       str
    type:         str
    description:  str
    impact_score: float
    date:         str
    source:       str
    confidence:   float

class EventsResponse(BaseModel):
    region:       str
    days:         int
    total_events: int
    avg_impact:   float
    events:       List[GeopoliticalEvent]
    retrieved_at: str

def stable_sample(region: str, date_str: str, days: int, n: int):
    """Retourne des événements déterministes basés sur région+date (reproductible)."""
    seed_val = int(hashlib.md5(f"{region}{date_str}{days}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed_val)
    templates = EVENT_TEMPLATES.get(region, EVENT_TEMPLATES["Europe"])
    chosen = rng.choices(templates, k=min(n, len(templates)))
    events = []
    for i, (etype, desc, base_impact) in enumerate(chosen):
        noise = rng.uniform(-0.05, 0.05)
        impact = round(max(-1.0, min(1.0, base_impact + noise)), 2)
        event_date = (datetime.now() - timedelta(days=rng.randint(0, days))).strftime("%Y-%m-%d")
        events.append(GeopoliticalEvent(
            event_id=     f"{region[:3].upper()}-{date_str.replace('-','')}-{i:03d}",
            region=       region,
            type=         etype,
            description=  desc,
            impact_score= impact,
            date=         event_date,
            source=       rng.choice(["Reuters", "Bloomberg", "AFP", "AP", "FT", "WSJ"]),
            confidence=   round(rng.uniform(0.65, 0.98), 2),
        ))
    return events

@app.get("/events", response_model=EventsResponse, summary="Événements géopolitiques par région")
def get_events(
    region: str = Query(..., description="Région géopolitique", example="Moyen-Orient"),
    days:   int = Query(7,  ge=1, le=90, description="Fenêtre temporelle en jours"),
    limit:  int = Query(5,  ge=1, le=20, description="Nombre max d'événements"),
):
    """
    Retourne les événements géopolitiques récents pour une région donnée.

    - **impact_score** : de -1.0 (très négatif pour les marchés) à +1.0 (très positif)
    - **confidence** : niveau de fiabilité de la source (0 à 1)
    - **type** : conflit | sanction | diplomatie | économique | politique | tension
    """
    valid_regions = list(EVENT_TEMPLATES.keys())
    if region not in valid_regions:
        raise HTTPException(
            status_code=404,
            detail=f"Région '{region}' inconnue. Régions disponibles : {valid_regions}"
        )
    today = datetime.now().strftime("%Y-%m-%d")
    events = stable_sample(region, today, days, limit)
    avg_impact = round(sum(e.impact_score for e in events) / len(events), 3) if events else 0.0
    return EventsResponse(
        region=       region,
        days=         days,
        total_events= len(events),
        avg_impact=   avg_impact,
        events=       events,
        retrieved_at= datetime.now().isoformat(),
    )

@app.get("/regions", summary="Liste des régions disponibles")
def get_regions():
    """Retourne la liste de toutes les régions couvertes par l'API."""
    return {"regions": list(EVENT_TEMPLATES.keys()), "total": len(EVENT_TEMPLATES)}

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "version": "1.0.0", "timestamp": datetime.now().isoformat()}
