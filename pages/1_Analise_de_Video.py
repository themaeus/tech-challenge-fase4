import json
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src import config, ui
from src.sample_data import VIDEO_SAMPLES
from src import video_analysis as va

ui.setup("Análise de Vídeo", "🎥")
st.title("🎥 Análise de Vídeo Clínico")
st.caption("YOLOv8-Pose (17 keypoints no padrão COCO/OpenPose) para análise postural · "
           "YOLOv8 para objetos e áreas críticas · relatório automático de desvios")

c1, c2, c3 = st.columns([2, 1, 2])
with c1:
    pid = ui.patient_selector("video_pid")
with c2:
    mode = st.radio("Tipo de procedimento", ["Fisioterapia", "Cirurgia"], key="video_mode")
with c3:
    source = st.radio("Fonte", ["Sequência de exemplo (sem GPU, instantâneo)", "Enviar vídeo (YOLOv8)",
                                "Enviar keypoints JSON (OpenPose/YOLO)"], key="video_src")

with st.expander("⚙️ Parâmetros do padrão esperado"):
    p1, p2, p3, p4 = st.columns(4)
    cfg = va.VideoConfig(
        mode=mode,
        tilt_limit_deg=p1.slider("Inclinação máx. do tronco (°)", 10, 60, 25),
        asym_limit_deg=p2.slider("Assimetria máx. entre joelhos (°)", 10, 60, 25),
        knee_min_expected=p3.slider("Meta de flexão do joelho (°)", 60, 150, 90),
        angular_speed_limit=p4.slider("Vel. angular máx. (°/s)", 100, 900, 400, step=50),
        sample_fps=p1.select_slider("Quadros analisados/s", [2, 5, 10, 15], 10),
    )
    z = p2.slider("Área crítica — faixa horizontal (fração)", 0.0, 1.0, (0.25, 0.75))
    zy = p3.slider("Área crítica — faixa vertical (fração)", 0.0, 1.0, (0.25, 0.9))
    cfg.critical_zone = (z[0], zy[0], z[1], zy[1])

seq, video_path = None, None
if source.startswith("Sequência"):
    sample = st.selectbox("Sessão de exemplo", list(VIDEO_SAMPLES), format_func=lambda n: VIDEO_SAMPLES[n][0])
    st.caption(f"Sessão de fisioterapia sintética (agachamentos) — sugerida para o paciente {VIDEO_SAMPLES[sample][1]}.")
    if st.button("▶️ Analisar sequência de exemplo", type="primary"):
        seq = json.loads((config.SAMPLES_DIR / f"pose_{sample}.json").read_text())
elif source.startswith("Enviar vídeo"):
    up = st.file_uploader("Vídeo (mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv"])
    max_s = st.slider("Analisar até (segundos)", 10, 300, 60)
    if up and st.button("▶️ Processar vídeo com YOLOv8", type="primary"):
        tmp = Path(tempfile.gettempdir()) / up.name
        tmp.write_bytes(up.getvalue())
        bar = st.progress(0.0, "Executando YOLOv8-Pose + YOLOv8…")
        try:
            seq, video_path = va.run_models_on_video(tmp, cfg, max_seconds=max_s,
                                                     progress=lambda p: bar.progress(p, f"Processando… {p:.0%}"))
            bar.empty()
        except ImportError:
            st.error("Instale o pacote `ultralytics` (pip install -r requirements.txt).")
else:
    up = st.file_uploader("JSON com {fps, width, height, frames:[{t, keypoints[17][3], objects}]}", type=["json"])
    if up and st.button("▶️ Analisar keypoints", type="primary"):
        seq = json.loads(up.getvalue())

if seq is not None:
    with st.spinner("Analisando padrões de movimento…"):
        metrics, events = va.analyze_sequence(seq, cfg)
        if video_path is None:
            video_path = va.render_sequence_video(seq, events, cfg, f"video_{pid}_{abs(hash(seq['description'])) % 10000}")
        report = va.build_report(seq, metrics, events, cfg, pid)
    st.session_state["video_result"] = dict(metrics=metrics, events=events, video=video_path, report=report, pid=pid)
    ui.publish_alerts(va.events_to_alerts(events, pid, seq.get("description", "vídeo")))

res = st.session_state.get("video_result")
if res:
    metrics, events = res["metrics"], res["events"]
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Quadros analisados", len(metrics))
    m2.metric("🔴 Eventos críticos", sum(e.severity == "CRITICO" for e in events))
    m3.metric("🟠 Eventos de atenção", sum(e.severity == "ATENCAO" for e in events))

    v, t = st.columns([3, 2])
    with v:
        st.subheader("Vídeo anotado")
        st.video(res["video"])
    with t:
        st.subheader("Eventos detectados")
        for e in events:
            ui.alert_card(e.severity, e.kind, e.description, f"{e.t_start:.1f}s – {e.t_end:.1f}s")
        if not events:
            st.success("Nenhum desvio do padrão esperado.")

    angle_cols = [c for c in ["joelho_esq", "joelho_dir", "inclinacao_tronco", "cotovelo_esq", "cotovelo_dir"]
                  if c in metrics]
    if angle_cols:
        st.subheader("Ângulos articulares ao longo do tempo")
        fig = go.Figure()
        for c in angle_cols:
            fig.add_scatter(x=metrics["t"], y=metrics[c], name=c.replace("_", " "), mode="lines")
        for e in events:
            fig.add_vrect(x0=e.t_start, x1=max(e.t_end, e.t_start + 0.2), opacity=0.15, line_width=0,
                          fillcolor=ui.SEV_COLOR[e.severity], annotation_text=e.kind, annotation_position="top left")
        fig.update_layout(height=360, xaxis_title="tempo (s)", yaxis_title="graus", margin=dict(t=30, b=0),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    st.subheader("📄 Relatório automático")
    st.markdown(res["report"])
    st.download_button("⬇️ Baixar relatório (.md)", res["report"], file_name=f"relatorio_video_{res['pid']}.md")
