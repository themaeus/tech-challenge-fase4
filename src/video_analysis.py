"""Análise de vídeo clínico: postura (YOLOv8-Pose, keypoints no padrão COCO/OpenPose)
e objetos/áreas críticas (YOLOv8 detecção).

Pipeline:
  vídeo -> amostragem de quadros -> YOLOv8-Pose (17 keypoints) + YOLOv8 (objetos)
        -> métricas biomecânicas por quadro (ângulos, inclinação, velocidade)
        -> regras de evento (queda, desequilíbrio, assimetria, movimento brusco,
           objeto em área crítica, divergência na contagem de instrumentos)
        -> relatório automático + alertas.

A etapa de inferência (`run_models_on_video`) é separada da análise (`analyze_sequence`),
assim a análise também funciona com keypoints vindos do OpenPose ou de um arquivo JSON.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config
from .alerts import Alert

# Índices COCO-17 (mesma topologia do OpenPose BODY-18 sem o "pescoço")
KP = {"nose": 0, "l_sho": 5, "r_sho": 6, "l_elb": 7, "r_elb": 8, "l_wri": 9, "r_wri": 10,
      "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14, "l_ank": 15, "r_ank": 16}
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

SURGICAL_OBJECTS = {"scissors", "knife"}             # instrumentos (classes COCO)
FORBIDDEN_IN_ZONE = {"cell phone", "cup", "bottle"}  # não deveriam estar no campo estéril


@dataclass
class VideoConfig:
    mode: str = "Fisioterapia"           # "Fisioterapia" ou "Cirurgia"
    tilt_limit_deg: float = 25.0         # inclinação de tronco tolerada
    asym_limit_deg: float = 25.0         # diferença entre joelhos/cotovelos
    knee_min_expected: float = 90.0      # flexão alvo do exercício (ângulo mínimo do joelho)
    angular_speed_limit: float = 400.0   # graus/s -> movimento brusco
    min_event_s: float = 0.4             # duração mínima de um evento
    # área crítica (campo estéril / zona do leito) em frações do quadro: x1, y1, x2, y2
    critical_zone: tuple = (0.25, 0.25, 0.75, 0.9)
    sample_fps: float = 10.0
    conf_kp: float = 0.3


@dataclass
class VideoEvent:
    t_start: float
    t_end: float
    kind: str
    severity: str
    description: str

    def as_dict(self):
        return {"início (s)": round(self.t_start, 1), "fim (s)": round(self.t_end, 1), "evento": self.kind,
                "severidade": self.severity, "descrição": self.description}


# ============================================================================ inferência
def run_models_on_video(path: str | Path, cfg: VideoConfig = VideoConfig(), max_seconds: float = 120,
                        annotate: bool = True, progress=None) -> tuple[dict, str | None]:
    """Executa YOLOv8-Pose + YOLOv8 sobre o vídeo. Retorna (sequência, caminho_do_vídeo_anotado)."""
    from ultralytics import YOLO  # import tardio: só exige a lib quando há vídeo real

    pose_model = YOLO(config.YOLO_POSE_MODEL)
    det_model = YOLO(config.YOLO_DET_MODEL)
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(src_fps / cfg.sample_fps))
    out_fps = src_fps / step
    out_path = config.OUTPUT_DIR / f"{Path(path).stem}_anotado.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H)) if annotate else None

    frames, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or idx / src_fps > max_seconds:
            break
        if idx % step == 0:
            t = idx / src_fps
            pr = pose_model(frame, verbose=False)[0]
            kps = None
            if pr.keypoints is not None and len(pr.boxes):
                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in pr.boxes.xyxy.cpu().numpy()]
                kps = pr.keypoints.data.cpu().numpy()[int(np.argmax(areas))].tolist()  # pessoa principal
            dr = det_model(frame, verbose=False, conf=0.35)[0]
            objs = [{"label": dr.names[int(b.cls)], "conf": float(b.conf), "box": b.xyxy[0].tolist()}
                    for b in dr.boxes if dr.names[int(b.cls)] != "person"]
            frames.append({"t": round(t, 3), "keypoints": kps, "objects": objs})
            if writer:
                canvas = pr.plot()
                draw_objects_and_zone(canvas, objs, cfg)
                writer.write(canvas)
            if progress and total:
                progress(min(idx / min(total, max_seconds * src_fps), 1.0))
        idx += 1
    cap.release()
    if writer:
        writer.release()
    seq = {"fps": out_fps, "width": W, "height": H, "frames": frames, "description": Path(path).name}
    return seq, (to_browser_mp4(out_path) if annotate else None)


def _ascii(s: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def draw_objects_and_zone(img, objs, cfg: VideoConfig, events_text: str = ""):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = cfg.critical_zone
    cv2.rectangle(img, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 200, 255), 2)
    cv2.putText(img, "AREA CRITICA", (int(x1 * W) + 4, int(y1 * H) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    for o in objs:
        a, b, c, d = map(int, o["box"])
        col = (0, 0, 255) if o["label"] in SURGICAL_OBJECTS | FORBIDDEN_IN_ZONE else (255, 160, 0)
        cv2.rectangle(img, (a, b), (c, d), col, 2)
        cv2.putText(img, f"{o['label']} {o['conf']:.2f}", (a, max(b - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    if events_text:
        cv2.rectangle(img, (0, H - 34), (W, H), (0, 0, 180), -1)
        cv2.putText(img, _ascii(events_text), (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def render_sequence_video(seq: dict, events: list[VideoEvent], cfg: VideoConfig, name: str) -> str:
    """Desenha o esqueleto (para sequências de keypoints sem vídeo original, ex.: demo/OpenPose JSON)."""
    W, H, fps = seq["width"], seq["height"], seq["fps"]
    out_path = config.OUTPUT_DIR / f"{name}.mp4"
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for fr in seq["frames"]:
        img = np.full((H, W, 3), 245, np.uint8)
        active = [e for e in events if e.t_start <= fr["t"] <= e.t_end + 0.5]
        if fr["keypoints"]:
            k = np.array(fr["keypoints"])
            color = (0, 0, 220) if any(e.severity == "CRITICO" for e in active) else \
                (0, 140, 255) if active else (60, 160, 60)
            for a, b in SKELETON:
                cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), color, 4)
            for x, y, c in k:
                cv2.circle(img, (int(x), int(y)), 5, (40, 40, 40), -1)
        txt = " | ".join(sorted({e.kind for e in active})).upper()
        draw_objects_and_zone(img, fr.get("objects", []), cfg, txt)
        cv2.putText(img, f"t={fr['t']:.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        wr.write(img)
    wr.release()
    return to_browser_mp4(out_path)


def to_browser_mp4(path: Path) -> str:
    """Recodifica para H.264 (reproduzível no navegador) usando o ffmpeg disponível."""
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return str(path)
    out = path.with_name(path.stem + "_h264.mp4")
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), "-vcodec", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)])
    return str(out) if r.returncode == 0 and out.exists() else str(path)


# ============================================================================ análise
def _angle(a, b, c) -> float:
    """Ângulo ABC em graus (180 = articulação estendida)."""
    ba, bc = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def frame_metrics(kps, conf_min: float = 0.3) -> dict:
    k = np.asarray(kps, dtype=float)
    ok = lambda *ix: all(k[i, 2] >= conf_min for i in ix)  # noqa: E731
    m = {}
    if ok(11, 13, 15):
        m["joelho_esq"] = _angle(k[11, :2], k[13, :2], k[15, :2])
    if ok(12, 14, 16):
        m["joelho_dir"] = _angle(k[12, :2], k[14, :2], k[16, :2])
    if ok(5, 7, 9):
        m["cotovelo_esq"] = _angle(k[5, :2], k[7, :2], k[9, :2])
    if ok(6, 8, 10):
        m["cotovelo_dir"] = _angle(k[6, :2], k[8, :2], k[10, :2])
    if ok(5, 6, 11, 12):
        sh, hip = k[[5, 6], :2].mean(0), k[[11, 12], :2].mean(0)
        v = sh - hip  # eixo do tronco; vertical "para cima" = (0, -1) em coordenadas de imagem
        m["inclinacao_tronco"] = float(np.degrees(np.arccos(np.clip(-v[1] / (np.linalg.norm(v) + 1e-9), -1, 1))))
        m["quadril_y"] = float(hip[1])
    vis = k[k[:, 2] >= conf_min, :2]
    if len(vis) >= 4:
        w, h = np.ptp(vis[:, 0]), np.ptp(vis[:, 1])
        m["razao_aspecto"] = float(w / (h + 1e-9))  # >1 => corpo na horizontal
    return m


def _runs(mask: pd.Series, t: pd.Series, min_s: float):
    """Intervalos contínuos onde mask é True com duração mínima."""
    out, start = [], None
    for i, (v, tt) in enumerate(zip(mask, t)):
        if v and start is None:
            start = tt
        if (not v or i == len(mask) - 1) and start is not None:
            end = tt
            if end - start >= min_s:
                out.append((start, end))
            start = None
    return out


def analyze_sequence(seq: dict, cfg: VideoConfig = VideoConfig()) -> tuple[pd.DataFrame, list[VideoEvent]]:
    rows = []
    for fr in seq["frames"]:
        r = {"t": fr["t"], "pessoa_detectada": fr["keypoints"] is not None}
        if fr["keypoints"] is not None:
            r.update(frame_metrics(fr["keypoints"], cfg.conf_kp))
        r["objetos"] = ", ".join(o["label"] for o in fr.get("objects", []))
        rows.append(r)
    df = pd.DataFrame(rows)
    events: list[VideoEvent] = []
    if df.empty:
        return df, events
    t = df["t"]
    dt = t.diff().replace(0, np.nan).fillna(1 / max(seq.get("fps", 10), 1))

    # pessoa fora do quadro
    for a, b in _runs(~df["pessoa_detectada"], t, 2.0):
        events.append(VideoEvent(a, b, "Paciente fora do campo de visão", "INFO",
                                 "Nenhuma pessoa detectada — verificar enquadramento/presença do paciente"))

    if "inclinacao_tronco" in df:
        tilt = df["inclinacao_tronco"].fillna(0)
        ar = df.get("razao_aspecto", pd.Series(0, index=df.index)).fillna(0)
        hip_v = df["quadril_y"].diff().div(dt).fillna(0)  # px/s (positivo = descendo)
        fallen = (tilt > 60) & (ar > 1.0)
        fall_runs = _runs(fallen, t, cfg.min_event_s)
        for a, b in fall_runs:
            pre = hip_v[(t >= a - 1.0) & (t <= a + 0.5)].max()
            events.append(VideoEvent(a, b, "Queda", "CRITICO",
                                     f"Corpo na horizontal (inclinação > 60°) após descida rápida do quadril "
                                     f"({pre:.0f} px/s). Acionar equipe imediatamente."))
        in_fall = pd.Series(False, index=df.index)
        for a, b in fall_runs:
            in_fall |= (t >= a - 1.0) & (t <= b)
        for a, b in _runs((tilt > cfg.tilt_limit_deg) & ~fallen & ~in_fall, t, cfg.min_event_s):
            mx = tilt[(t >= a) & (t <= b)].max()
            events.append(VideoEvent(a, b, "Perda de equilíbrio", "ATENCAO",
                                     f"Inclinação do tronco de até {mx:.0f}° (limite {cfg.tilt_limit_deg:.0f}°)"))

    if {"joelho_esq", "joelho_dir"} <= set(df.columns):
        diff = (df["joelho_esq"] - df["joelho_dir"]).abs()
        tilt_ok = df.get("inclinacao_tronco", pd.Series(0, index=df.index)).fillna(0) < 60
        for a, b in _runs((diff > cfg.asym_limit_deg) & tilt_ok, t, cfg.min_event_s):
            events.append(VideoEvent(a, b, "Assimetria articular", "ATENCAO",
                                     f"Diferença entre joelhos de até {diff[(t >= a) & (t <= b)].max():.0f}° "
                                     f"(limite {cfg.asym_limit_deg:.0f}°) — possível compensação/dor"))
        for col in ["joelho_esq", "joelho_dir"]:
            speed = df[col].diff().abs().div(dt)
            for a, b in _runs((speed > cfg.angular_speed_limit) & tilt_ok, t, 0.0):
                events.append(VideoEvent(a, b, "Movimento brusco", "ATENCAO",
                                         f"{col.replace('_', ' ')}: velocidade angular {speed[(t >= a) & (t <= b)].max():.0f}°/s"))
        if cfg.mode == "Fisioterapia":
            kmin = df.loc[tilt_ok, ["joelho_esq", "joelho_dir"]].min().min()
            if pd.notna(kmin) and kmin > cfg.knee_min_expected + 15:
                events.append(VideoEvent(float(t.iloc[0]), float(t.iloc[-1]), "Amplitude reduzida", "INFO",
                                         f"Flexão máxima do joelho atingiu {kmin:.0f}° (meta ≤ {cfg.knee_min_expected:.0f}°)"))

    # objetos / área crítica
    W, H = seq["width"], seq["height"]
    x1, y1, x2, y2 = cfg.critical_zone
    def in_zone(box):
        cx, cy = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H
        return x1 <= cx <= x2 and y1 <= cy <= y2
    forb = pd.Series([any(o["label"] in FORBIDDEN_IN_ZONE and in_zone(o["box"]) for o in fr.get("objects", []))
                      for fr in seq["frames"]])
    for a, b in _runs(forb, t, cfg.min_event_s):
        events.append(VideoEvent(a, b, "Objeto indevido na área crítica", "ATENCAO",
                                 "Objeto não estéril (ex.: celular/copo) detectado dentro da área crítica"))
    if cfg.mode == "Cirurgia":
        counts = pd.Series([sum(o["label"] in SURGICAL_OBJECTS for o in fr.get("objects", []))
                            for fr in seq["frames"]])
        n = max(len(counts) // 10, 1)
        c0, c1 = counts.head(n).median(), counts.tail(n).median()
        if c0 != c1:
            events.append(VideoEvent(float(t.iloc[-n]), float(t.iloc[-1]), "Contagem de instrumentos divergente",
                                     "CRITICO", f"Instrumentos visíveis no início: {c0:.0f}; no fim: {c1:.0f} — "
                                     "conferir contagem (risco de item retido)"))
        out_zone = pd.Series([any(o["label"] in SURGICAL_OBJECTS and not in_zone(o["box"])
                                  for o in fr.get("objects", [])) for fr in seq["frames"]])
        for a, b in _runs(out_zone, t, 1.0):
            events.append(VideoEvent(a, b, "Instrumento fora do campo", "INFO",
                                     "Instrumento cirúrgico detectado fora da área crítica/estéril"))

    events.sort(key=lambda e: e.t_start)
    return df, events


def build_report(seq: dict, metrics: pd.DataFrame, events: list[VideoEvent], cfg: VideoConfig,
                 patient_id: str) -> str:
    dur = metrics["t"].max() if not metrics.empty else 0
    det = metrics["pessoa_detectada"].mean() * 100 if not metrics.empty else 0
    lines = [f"# Relatório automático de vídeo — {cfg.mode}",
             f"- **Paciente:** {patient_id}", f"- **Arquivo:** {seq.get('description', '-')}",
             f"- **Duração analisada:** {dur:.1f} s ({len(metrics)} quadros a {seq.get('fps', 0):.0f} fps)",
             f"- **Pessoa detectada em:** {det:.0f}% dos quadros",
             "- **Modelos:** YOLOv8-Pose (17 keypoints COCO/OpenPose) + YOLOv8 (detecção de objetos)", ""]
    if "inclinacao_tronco" in metrics:
        lines += ["## Indicadores biomecânicos",
                  "| Métrica | Mín | Média | Máx |", "|---|---|---|---|"]
        for c in ["joelho_esq", "joelho_dir", "cotovelo_esq", "cotovelo_dir", "inclinacao_tronco"]:
            if c in metrics:
                s = metrics[c].dropna()
                lines.append(f"| {c.replace('_', ' ')} (°) | {s.min():.0f} | {s.mean():.0f} | {s.max():.0f} |")
        lines.append("")
    lines.append("## Desvios / falhas detectados")
    if not events:
        lines.append("Nenhum desvio relevante identificado. ✅")
    for e in events:
        icon = {"CRITICO": "🔴", "ATENCAO": "🟠", "INFO": "🔵"}[e.severity]
        lines.append(f"- {icon} **{e.kind}** ({e.t_start:.1f}s–{e.t_end:.1f}s): {e.description}")
    crit = [e for e in events if e.severity == "CRITICO"]
    lines += ["", "## Conclusão",
              ("⚠️ Procedimento com **falhas críticas** — equipe notificada." if crit else
               "Procedimento com desvios de atenção — revisar com o fisioterapeuta/cirurgião." if events else
               "Procedimento dentro do padrão esperado.")]
    return "\n".join(lines)


def events_to_alerts(events: list[VideoEvent], patient_id: str, source_name: str) -> list[Alert]:
    from datetime import datetime
    now = datetime.now().replace(microsecond=0).isoformat()
    return [Alert(patient_id, "VIDEO", e.severity, e.kind,
                  f"{source_name} @ {e.t_start:.1f}s: {e.description}", timestamp=now,
                  details={"t_start": e.t_start, "t_end": e.t_end})
            for e in events if e.severity != "INFO"]
