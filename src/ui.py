"""Componentes compartilhados da interface Streamlit."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import config
from .alerts import SEVERITY_LABEL, AlertManager

SEV_COLOR = {"CRITICO": "#d62728", "ATENCAO": "#ff7f0e", "INFO": "#1f77b4"}

CSS = """
<style>
.block-container {padding-top: 1.6rem;}
.sev-CRITICO {background:#fde8e8;border-left:6px solid #d62728;padding:.55rem .8rem;border-radius:6px;margin-bottom:.4rem}
.sev-ATENCAO {background:#fff3e0;border-left:6px solid #ff7f0e;padding:.55rem .8rem;border-radius:6px;margin-bottom:.4rem}
.sev-INFO {background:#e8f1fb;border-left:6px solid #1f77b4;padding:.55rem .8rem;border-radius:6px;margin-bottom:.4rem}
.small {font-size:.85rem;color:#666}
</style>
"""


def setup(title: str, icon: str = "🏥") -> None:
    st.set_page_config(page_title=f"{title} · Monitoramento Multimodal", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar()


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🏥 Monitoramento Multimodal")
        st.caption("Tech Challenge — Fase 4")
        st.markdown(f"{'🟢' if config.LINK_AVISO_EQUIPE else '⚪'} Aviso da equipe no Teams/Slack "
                    f"{'(ativo)' if config.LINK_AVISO_EQUIPE else '(desligado)'}")
        df = alert_manager().to_dataframe()
        if not df.empty:
            open_ = df[df["acknowledged"] == 0]
            st.markdown("**Alertas em aberto**")
            c1, c2 = st.columns(2)
            c1.metric("🔴 Críticos", int((open_["severity"] == "CRITICO").sum()))
            c2.metric("🟠 Atenção", int((open_["severity"] == "ATENCAO").sum()))


@st.cache_resource
def alert_manager() -> AlertManager:
    return AlertManager()


def patients() -> dict:
    from .sample_data import PATIENTS, VIDEO_SAMPLES, generate_all
    if not all((config.SAMPLES_DIR / f"pose_{n}.json").exists() for n in VIDEO_SAMPLES):
        generate_all()
    return PATIENTS


def patient_selector(key: str, label: str = "Paciente") -> str:
    pats = patients()
    return st.selectbox(label, list(pats), format_func=lambda p: f"{p} — {pats[p]}", key=key)


def publish_alerts(alerts: list) -> int:
    """Envia alertas para a central e mostra notificações na tela."""
    n = alert_manager().emit_many(alerts)
    crit = [a for a in alerts if a.severity == "CRITICO"]
    for a in crit[:3]:
        st.toast(f"🔴 {a.patient_id}: {a.title}", icon="🚨")
    if n:
        st.success(f"{n} novo(s) alerta(s) enviado(s) à Central de Alertas / equipe médica.", icon="📣")
    elif alerts:
        st.caption("Alertas já registrados anteriormente (deduplicados).")
    return n


def alert_card(severity: str, title: str, body: str, meta: str = "") -> None:
    st.markdown(f"<div class='sev-{severity}'><b>{SEVERITY_LABEL[severity]} · {title}</b><br>{body}"
                f"{'<br><span class=small>' + meta + '</span>' if meta else ''}</div>", unsafe_allow_html=True)


def style_severity(df: pd.DataFrame, col: str = "severity"):
    def color(v):
        return f"background-color: {SEV_COLOR.get(v, '')}22; color: {SEV_COLOR.get(v, '')}; font-weight:600" \
            if v in SEV_COLOR else ""
    return df.style.map(color, subset=[col])
