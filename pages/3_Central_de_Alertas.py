import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, fusion, ui
from src.alerts import SEVERITY_LABEL

ui.setup("Central de Alertas", "🚨")
st.title("🚨 Central de Alertas da Equipe Médica")
st.caption("Todos os módulos publicam aqui. Alertas de atenção e críticos também podem ser avisados à equipe "
           f"no Teams/Slack — {'🟢 ativo' if config.LINK_AVISO_EQUIPE else '⚪ desligado (configure no arquivo .env)'}.")

mgr = ui.alert_manager()
if st.button("🧠 Recalcular fusão multimodal"):
    risk = fusion.compute_risk(mgr.to_dataframe(), list(ui.patients()))
    ui.publish_alerts(fusion.fusion_alerts(risk))

df = mgr.to_dataframe()
if df.empty:
    st.info("Nenhum alerta registrado. Execute a demonstração no Painel Central ou use os módulos de análise.")
    st.stop()

f1, f2, f3, f4 = st.columns(4)
pats = f1.multiselect("Paciente", sorted(df["patient_id"].unique()))
sevs = f2.multiselect("Severidade", list(SEVERITY_LABEL), default=["CRITICO", "ATENCAO"],
                      format_func=SEVERITY_LABEL.get)
srcs = f3.multiselect("Origem", sorted(df["source"].unique()))
only_open = f4.toggle("Somente em aberto", True)

v = df.copy()
if pats:
    v = v[v["patient_id"].isin(pats)]
if sevs:
    v = v[v["severity"].isin(sevs)]
if srcs:
    v = v[v["source"].isin(srcs)]
if only_open:
    v = v[v["acknowledged"] == 0]
v = v.sort_values(["sev_rank", "timestamp"], ascending=[False, False])

a, b = st.columns([3, 2])
with a:
    st.subheader(f"Fila de alertas ({len(v)})")
    show = v[["id", "severity", "patient_id", "source", "title", "message", "timestamp", "acknowledged", "notified"]]
    st.dataframe(ui.style_severity(show), hide_index=True, width="stretch", height=420,
                 column_config={"acknowledged": st.column_config.CheckboxColumn("reconhecido"),
                                "notified": st.column_config.CheckboxColumn("avisado no Teams/Slack")})
    c1, c2, c3 = st.columns([2, 1, 1])
    ids = c1.multiselect("Reconhecer alertas (ID)", v["id"].tolist())
    if c2.button("✔️ Reconhecer", disabled=not ids, width="stretch"):
        for i in ids:
            mgr.acknowledge(i)
        st.rerun()
    c3.download_button("⬇️ Exportar CSV", df.to_csv(index=False).encode("utf-8"), "alertas.csv",
                       width="stretch")
with b:
    st.subheader("Distribuição")
    fig = px.histogram(df, x="source", color="severity", barmode="stack",
                       color_discrete_map=ui.SEV_COLOR, category_orders={"severity": ["CRITICO", "ATENCAO", "INFO"]})
    fig.update_layout(height=260, margin=dict(t=10, b=0), xaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width="stretch")
    fig2 = px.histogram(df, x="patient_id", color="severity", color_discrete_map=ui.SEV_COLOR,
                        category_orders={"severity": ["CRITICO", "ATENCAO", "INFO"]})
    fig2.update_layout(height=220, margin=dict(t=10, b=0), xaxis_title=None, legend_title=None, showlegend=False)
    st.plotly_chart(fig2, width="stretch")

st.subheader("🔁 Fluxo do alerta")
st.markdown("""
1. **Detecção** — o módulo de vídeo ou de áudio identifica um evento fora do padrão.
2. **Classificação** — severidade 🔵 Informativo · 🟠 Atenção · 🔴 Crítico.
3. **Registro e deduplicação** — gravado em banco (`outputs/alerts.db`) para auditoria.
4. **Aviso** — aviso na tela e, se configurado, mensagem no Microsoft Teams / Slack para atenção/crítico.
5. **Fusão** — alertas simultâneos em 2+ modalidades geram alerta de *Risco multimodal elevado*.
6. **Resposta** — a equipe reconhece o alerta nesta tela (fecha o ciclo e reduz o risco fundido).
""")
