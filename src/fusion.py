"""Fusão multimodal (late fusion) dos achados de vídeo e áudio.

Cada modalidade produz alertas independentes; aqui eles são convertidos em um escore
por modalidade (0–1) e combinados por "OU ruidoso" ponderado:

    risco = 1 − Π (1 − peso_m · escore_m)

Assim, uma evidência forte em uma única fonte já eleva o risco, e evidências moderadas
em fontes diferentes se somam. Quando vídeo e áudio estão alterados ao mesmo tempo,
gera-se um alerta de FUSÃO (sinal precoce de risco que nenhuma fonte isolada mostraria).
"""
from __future__ import annotations

import pandas as pd

from .alerts import Alert

MODALITIES = {
    "VIDEO": ("Vídeo (postura/objetos)", 0.8),
    "AUDIO": ("Áudio (voz)", 0.7),
}
SEV_WEIGHT = {"INFO": 0.1, "ATENCAO": 0.45, "CRITICO": 1.0}


def risk_level(r: float) -> str:
    return "🔴 Crítico" if r >= 0.8 else "🟠 Alto" if r >= 0.6 else "🟡 Moderado" if r >= 0.3 else "🟢 Baixo"


def compute_risk(alerts_df: pd.DataFrame, patients: list[str] | None = None) -> pd.DataFrame:
    """Tabela paciente × modalidade com escores e o risco fundido (considera alertas não reconhecidos)."""
    pats = sorted(set(patients or []) | set(alerts_df.get("patient_id", pd.Series(dtype=str))))
    rows = []
    for p in pats:
        a = alerts_df[(alerts_df["patient_id"] == p) & (alerts_df["source"] != "FUSAO")] if len(alerts_df) else alerts_df
        if len(a) and "acknowledged" in a:
            a = a[a["acknowledged"] == 0]
        row = {"patient_id": p}
        prod = 1.0
        for m, (_, w) in MODALITIES.items():
            sev = a[a["source"] == m]["severity"] if len(a) else []
            w_sev = sorted((SEV_WEIGHT[x] for x in sev), reverse=True)
            # escore = alerta mais grave + pequeno acréscimo por recorrência
            s = min(1.0, w_sev[0] + 0.05 * (len(w_sev) - 1)) if w_sev else 0.0
            row[m] = round(s, 2)
            prod *= 1 - w * s
        row["risco"] = round(1 - prod, 2)
        row["nivel"] = risk_level(row["risco"])
        row["modalidades_alteradas"] = sum(row[m] >= 0.4 for m in MODALITIES)
        rows.append(row)
    return pd.DataFrame(rows)


def fusion_alerts(risk_df: pd.DataFrame) -> list[Alert]:
    out = []
    for _, r in risk_df.iterrows():
        if r["modalidades_alteradas"] >= 2 and r["risco"] >= 0.6:
            mods = [MODALITIES[m][0] for m in MODALITIES if r[m] >= 0.4]
            out.append(Alert(r["patient_id"], "FUSAO", "CRITICO" if r["risco"] >= 0.8 else "ATENCAO",
                             "Risco multimodal elevado",
                             f"Risco fundido {r['risco']:.0%} com alterações simultâneas em: {', '.join(mods)}",
                             timestamp=pd.Timestamp.now().floor("h").isoformat(timespec="seconds")))
    return out
