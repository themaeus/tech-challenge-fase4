"""Executa o pipeline nos dados de exemplo e gera as figuras/tabelas usadas no relatório técnico.

Uso:  python scripts/gerar_resultados.py
Saída: docs/img/*.png, docs/relatorio_video_exemplo.md e docs/resultados.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src import audio_analysis, config, fusion, sample_data, video_analysis  # noqa: E402
from src.alerts import AlertManager  # noqa: E402
from src.pipeline import run_full_demo  # noqa: E402

IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)
SEV = {"CRITICO": "#d62728", "ATENCAO": "#ff7f0e", "INFO": "#1f77b4"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

sample_data.generate_all()
S = config.SAMPLES_DIR
results = {}

# ---------------------------------------------------------------- vídeo
cfg = video_analysis.VideoConfig()
results["video"] = {}
for name, (desc, pid, _) in sample_data.VIDEO_SAMPLES.items():
    seq = json.loads((S / f"pose_{name}.json").read_text())
    vm, ev = video_analysis.analyze_sequence(seq, cfg)
    results["video"][name] = {"descricao": desc, "paciente": pid, "eventos": [e.as_dict() for e in ev]}
    if name == "fisio_queda":
        (ROOT / "docs" / "relatorio_video_exemplo.md").write_text(
            video_analysis.build_report(seq, vm, ev, cfg, pid), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(9, 3))
        for c in ["joelho_esq", "joelho_dir", "inclinacao_tronco"]:
            ax.plot(vm.t, vm[c], label=c.replace("_", " "), lw=1)
        for e in ev:
            ax.axvspan(e.t_start, max(e.t_end, e.t_start + 0.15), color=SEV[e.severity], alpha=0.15)
            ax.text(e.t_start, 185, e.kind, rotation=90, fontsize=7, va="top", color=SEV[e.severity])
        ax.set_xlabel("tempo (s)")
        ax.set_ylabel("graus")
        ax.set_title("Sessão de fisioterapia: ângulos (YOLOv8-Pose/OpenPose) e eventos detectados")
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(IMG / "video_angulos.png")
        plt.close(fig)
        vid = video_analysis.render_sequence_video(seq, ev, cfg, "demo_fisioterapia")
        cap = cv2.VideoCapture(vid)
        frames = []
        for t in [4.5, 10.5, 17.0]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        if frames:
            cv2.imwrite(str(IMG / "video_quadros.png"), cv2.hconcat(frames))

# ---------------------------------------------------------------- áudio
aud = {}
for name, pid in sample_data.AUDIO_PATIENT.items():
    r = audio_analysis.analyze_consultation(S / f"{name}.wav", pid)
    aud[name] = {"paciente": pid, "features": r["features"],
                 "risk": {k: v for k, v in r["risk"].items() if k != "justificativas"},
                 "justificativas": r["risk"]["justificativas"],
                 "alertas": [f"{a.severity}: {a.title}" for a in r["alerts"]]}
results["audio"] = aud
fig, ax = plt.subplots(figsize=(6, 2.8))
keys = ["fadiga", "dificuldade_respiratoria", "disartria"]
x = range(len(keys))
ax.bar([i - 0.2 for i in x], [aud["consulta_normal"]["risk"][k] for k in keys], 0.4, label="consulta normal")
ax.bar([i + 0.2 for i in x], [aud["consulta_fadiga_dispneia"]["risk"][k] for k in keys], 0.4,
       label="consulta com fadiga/dispneia", color="#d62728")
ax.axhline(0.6, ls=":", color="k")
ax.set_xticks(list(x), ["Fadiga", "Dif. respiratória", "Disartria"])
ax.set_ylim(0, 1)
ax.set_title("Escores de biomarcadores vocais (limiar de alerta = 0,6)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(IMG / "audio_escores.png")
plt.close(fig)

# ---------------------------------------------------------------- fusão
mgr = AlertManager(config.OUTPUT_DIR / "relatorio_alerts.db", notify_url="")
mgr.clear()
results["demo_resumo"] = run_full_demo(mgr, log=lambda *_: None)
risk = fusion.compute_risk(mgr.to_dataframe(), list(sample_data.PATIENTS))
results["fusao"] = risk.to_dict("records")
alerts = mgr.to_dataframe()
results["alertas_por_origem"] = alerts.groupby(["source", "severity"]).size().unstack(fill_value=0).to_dict()

(ROOT / "docs" / "resultados.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str),
                                                encoding="utf-8")
print(risk.to_string())
print(json.dumps(results["alertas_por_origem"], indent=1))
print("Figuras em", IMG)
