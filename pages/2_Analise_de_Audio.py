import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import audio_analysis, config, ui

ui.setup("Análise de Áudio", "🎙️")
st.title("🎙️ Análise de Áudio de Consultas")
st.caption("Biomarcadores vocais extraídos do sinal de voz: pausas, entonação (F0), jitter, shimmer, HNR e "
           "ruído respiratório → escores de fadiga, dificuldade respiratória e disartria")

LABELS = {
    "duracao_s": "Duração (s)", "tempo_fala_s": "Tempo de fala (s)", "proporcao_pausas": "Proporção de pausas",
    "pausas_longas_por_min": "Pausas longas por minuto", "taxa_articulacao_sil_s": "Taxa de articulação (sílabas/s)",
    "energia_media_db": "Energia média (dB)", "f0_medio_hz": "F0 médio (Hz)",
    "f0_variabilidade_st": "Variação de F0 (semitons)", "jitter_pct": "Jitter (%)", "shimmer_pct": "Shimmer (%)",
    "hnr_db": "HNR (dB)", "ruido_respiratorio_db": "Ruído respiratório nas pausas (dB)",
}

c1, c2 = st.columns([1, 1])
with c1:
    pid = ui.patient_selector("audio_pid")
    source = st.radio("Fonte do áudio", ["Exemplo", "Enviar arquivo"], horizontal=True)
    audio_path = None
    if source == "Exemplo":
        name = st.selectbox("Gravação de exemplo", ["consulta_fadiga_dispneia", "consulta_normal"],
                            format_func=lambda n: {"consulta_fadiga_dispneia": "Consulta com fadiga e dispneia",
                                                   "consulta_normal": "Consulta normal"}[n])
        audio_path = config.SAMPLES_DIR / f"{name}.wav"
    else:
        up = st.file_uploader("Áudio da consulta (wav, mp3, m4a, ogg)", type=["wav", "mp3", "m4a", "ogg", "flac"])
        if up:
            audio_path = Path(tempfile.gettempdir()) / up.name
            audio_path.write_bytes(up.getvalue())
with c2:
    if audio_path:
        st.audio(str(audio_path))
    st.info("Os escores comparam as características da voz com faixas de referência da fala adulta "
            "conversacional. Acima de 60% é gerado alerta de atenção; acima de 80%, alerta crítico.", icon="ℹ️")

if audio_path and st.button("▶️ Analisar consulta", type="primary"):
    with st.spinner("Extraindo biomarcadores vocais…"):
        try:
            res = audio_analysis.analyze_consultation(audio_path, pid)
        except Exception as e:
            st.error(f"Falha ao processar o áudio: {e}")
            st.stop()
    st.session_state["audio_result"] = res
    ui.publish_alerts(res["alerts"])

res = st.session_state.get("audio_result")
if res:
    st.divider()
    f, risk = res["features"], res["risk"]

    st.subheader("🩺 Indicadores vocais")
    g1, g2, g3 = st.columns(3)
    for col, key, label in [(g1, "fadiga", "Fadiga"), (g2, "dificuldade_respiratoria", "Dificuldade respiratória"),
                            (g3, "disartria", "Disartria")]:
        v = risk[key]
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=v * 100, number={"suffix": "%"}, title={"text": label},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                   "steps": [{"range": [0, 60], "color": "#d9f2d9"}, {"range": [60, 80], "color": "#ffe0b3"},
                             {"range": [80, 100], "color": "#f9c0c0"}]}))
        fig.update_layout(height=210, margin=dict(l=20, r=20, t=40, b=0))
        col.plotly_chart(fig, width="stretch")
    if risk["justificativas"]:
        st.markdown("**Por quê:** " + "; ".join(risk["justificativas"]))

    w, ft = st.columns([3, 2])
    with w:
        y = res["signal"]
        t = np.arange(len(y)) / res["sr"]
        fig = go.Figure(go.Scatter(x=t, y=y, mode="lines", line=dict(width=0.6, color="#1f77b4")))
        fig.update_layout(height=260, title="Forma de onda", xaxis_title="s", margin=dict(t=40, b=0))
        st.plotly_chart(fig, width="stretch")
    with ft:
        st.dataframe(pd.DataFrame([(LABELS.get(k, k), v) for k, v in f.items()], columns=["característica", "valor"]),
                     hide_index=True, width="stretch", height=300)

    st.subheader("🚨 Alertas gerados")
    for al in res["alerts"]:
        ui.alert_card(al.severity, al.title, al.message)
    if not res["alerts"]:
        st.success("Nenhum sinal vocal de risco na consulta.")
