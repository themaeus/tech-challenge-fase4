"""Orquestra o fluxo multimodal completo sobre os dados de exemplo (usado no Painel e nos testes)."""
from __future__ import annotations

import json

from . import audio_analysis, config, fusion, video_analysis
from .alerts import Alert, AlertManager
from .sample_data import AUDIO_PATIENT, VIDEO_SAMPLES, generate_all


def run_full_demo(manager: AlertManager, log=print) -> dict:
    S = config.SAMPLES_DIR
    if not all((S / f"pose_{n}.json").exists() for n in VIDEO_SAMPLES):
        generate_all()
    summary: dict[str, int] = {}
    all_alerts: list[Alert] = []

    log("🎥 Vídeo: análise postural das sessões de fisioterapia (YOLOv8-Pose/OpenPose)…")
    n = 0
    for name, (desc, pid, _) in VIDEO_SAMPLES.items():
        seq = json.loads((S / f"pose_{name}.json").read_text())
        _, ev = video_analysis.analyze_sequence(seq, video_analysis.VideoConfig())
        a = video_analysis.events_to_alerts(ev, pid, f"Fisioterapia — {desc}")
        n += len(a)
        all_alerts += a
    summary["Vídeo"] = n

    log("🎙️ Áudio: biomarcadores vocais das consultas…")
    n = 0
    for name, pid in AUDIO_PATIENT.items():
        r = audio_analysis.analyze_consultation(S / f"{name}.wav", pid)
        n += len(r["alerts"])
        all_alerts += r["alerts"]
    summary["Áudio"] = n

    manager.emit_many(all_alerts)
    log("🧠 Fusão multimodal e alertas consolidados…")
    risk = fusion.compute_risk(manager.to_dataframe())
    fa = fusion.fusion_alerts(risk)
    manager.emit_many(fa)
    summary["Fusão"] = len(fa)
    return summary
