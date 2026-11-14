"""Painel Central — Monitoramento Multimodal de Pacientes (Tech Challenge Fase 4).

Executar:  streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import fusion, pipeline, ui

ui.setup("Painel Central")

st.title("🏥 Painel Central de Monitoramento Multimodal")
st.markdown(
    "Monitoramento de pacientes com **vídeo** (sessões de fisioterapia ou cirurgias) e **áudio** "
    "(consultas). Cada módulo identifica eventos fora do padrão, envia alertas para a **Central de Alertas** "
    "e a **fusão multimodal** calcula o risco consolidado de cada paciente.")

with st.expander("🔀 Fluxo multimodal", expanded=False):
    st.markdown("""
| Entrada | Modelo / técnica | Saída |
|---|---|---|
| 🎥 Vídeo (fisioterapia / cirurgia) | **YOLOv8-Pose** (keypoints estilo OpenPose) + **YOLOv8** (objetos) | quedas, desequilíbrio, assimetria, objetos em área crítica, relatório |
| 🎙️ Áudio de consulta | Biomarcadores vocais (pausas, F0, jitter, shimmer, HNR, ruído respiratório) | fadiga, dificuldade respiratória, disartria |
| 🧠 Fusão | *Late fusion* (OU ruidoso ponderado) | **risco por paciente** → alerta à equipe |
""")

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("▶️ Executar demonstração completa", type="primary", width="stretch"):
        with st.status("Processando dados multimodais…", expanded=True) as status:
            summary = pipeline.run_full_demo(ui.alert_manager(), log=st.write)
            status.update(label="Processamento concluído ✅", state="complete")
        st.toast("Demonstração concluída — veja os alertas abaixo", icon="✅")
        st.session_state["demo_summary"] = summary
    if st.button("🧹 Limpar alertas", width="stretch"):
        ui.alert_manager().clear()
        st.session_state.pop("demo_summary", None)
        st.rerun()
with c2:
    if "demo_summary" in st.session_state:
        s = st.session_state["demo_summary"]
        cols = st.columns(len(s))
        for col, (k, v) in zip(cols, s.items()):
            col.metric(k, f"{v} alertas")

alerts = ui.alert_manager().to_dataframe()
pats = ui.patients()
risk = fusion.compute_risk(alerts, list(pats))

st.divider()
k1, k2, k3, k4 = st.columns(4)
open_ = alerts[alerts["acknowledged"] == 0] if not alerts.empty else alerts
k1.metric("Pacientes monitorados", len(pats))
k2.metric("🔴 Alertas críticos em aberto", int((open_["severity"] == "CRITICO").sum()) if len(open_) else 0)
k3.metric("🟠 Alertas de atenção", int((open_["severity"] == "ATENCAO").sum()) if len(open_) else 0)
k4.metric("Maior risco fundido", f"{risk['risco'].max():.0%}" if len(risk) else "—")

left, right = st.columns([3, 2])
with left:
    st.subheader("🧠 Risco multimodal por paciente")
    view = risk.copy()
    view.insert(1, "descrição", view["patient_id"].map(pats))
    st.dataframe(
        view[["patient_id", "descrição", "nivel", "risco", *fusion.MODALITIES]],
        hide_index=True, width="stretch",
        column_config={
            "patient_id": "Paciente", "nivel": "Nível",
            "risco": st.column_config.ProgressColumn("Risco fundido", min_value=0, max_value=1, format="percent"),
            **{m: st.column_config.ProgressColumn(lab, min_value=0, max_value=1, format="%.2f")
               for m, (lab, _) in fusion.MODALITIES.items()},
        })
    if len(risk) and risk[list(fusion.MODALITIES)].values.sum() > 0:
        hm = risk.set_index("patient_id")[list(fusion.MODALITIES)].rename(
            columns={m: lab for m, (lab, _) in fusion.MODALITIES.items()})
        fig = px.imshow(hm, color_continuous_scale="OrRd", zmin=0, zmax=1, text_auto=".2f", aspect="auto",
                        labels=dict(color="Escore"))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Nenhum alerta ainda. Clique em **Executar demonstração completa** ou use os módulos no menu lateral.")

with right:
    st.subheader("🚨 Últimos alertas")
    if alerts.empty:
        st.caption("Sem alertas.")
    for _, a in alerts.sort_values(["acknowledged", "sev_rank", "id"], ascending=[True, False, False]).head(8).iterrows():
        ui.alert_card(a["severity"], f"{a['patient_id']} · {a['title']}", a["message"],
                      f"{a['source']} · {pd.Timestamp(a['timestamp']):%d/%m %H:%M}"
                      + (" · ✔️ reconhecido" if a["acknowledged"] else ""))
    st.page_link("pages/3_Central_de_Alertas.py", label="Abrir Central de Alertas →", icon="🚨")
