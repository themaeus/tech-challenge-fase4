"""Gera dados de demonstração sintéticos (áudios de consulta e sequência de pose de fisioterapia),
com eventos fora do padrão inseridos em pontos conhecidos para permitir conferir os resultados.
"""
from __future__ import annotations

import json

import numpy as np

from . import config

PATIENTS = {
    "P-001": "Maria, 68 anos — fisioterapia após cirurgia no joelho",
    "P-002": "João, 62 anos — recuperação de pneumonia (cansaço e falta de ar)",
    "P-003": "Ana, 35 anos — fisioterapia para dor nas costas",
}


# ---------------------------------------------------------------- áudio sintético
SR = 16000


def _voice(duration: float, f0: float, jitter: float, shimmer: float, pause_ratio: float,
           breath_noise: float, seed: int, intonation: float = 0.08) -> np.ndarray:
    """Sinal vocal sintético (fonte glotal harmônica + formantes simples + pausas)."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    out = np.zeros(n)
    pos = 0
    while pos < n:
        seg = int(rng.uniform(0.25, 0.6) * SR)  # "sílabas/palavras"
        seg = min(seg, n - pos)
        tt = np.arange(seg) / SR
        f = f0 * (1 + intonation * np.sin(2 * np.pi * rng.uniform(1, 3) * tt)) * (1 + rng.normal(0, jitter))
        phase = 2 * np.pi * np.cumsum(f) / SR
        sig = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 8))
        env = np.sin(np.pi * np.arange(seg) / seg) * (1 + rng.normal(0, shimmer))
        out[pos:pos + seg] = sig * env
        pos += seg
        gap = int(SR * (rng.uniform(0.05, 0.15) + (rng.uniform(0.4, 1.0) if rng.random() < pause_ratio else 0)))
        if breath_noise and pos < n:  # ruído respiratório nas pausas
            g = min(gap, n - pos)
            out[pos:pos + g] = rng.normal(0, breath_noise, g)
        pos += gap
    out += rng.normal(0, 0.003, n)
    return (out / np.abs(out).max() * 0.8).astype(np.float32)


AUDIO_SAMPLES = {
    "consulta_normal": dict(
        params=dict(duration=12, f0=180, jitter=0.004, shimmer=0.03, pause_ratio=0.1, breath_noise=0.0, seed=1,
                    intonation=0.2)),
    "consulta_fadiga_dispneia": dict(
        params=dict(duration=12, f0=120, jitter=0.03, shimmer=0.18, pause_ratio=0.55, breath_noise=0.05, seed=2,
                    intonation=0.04)),
}


def write_audio_samples() -> list[str]:
    from scipy.io import wavfile
    paths = []
    for name, spec in AUDIO_SAMPLES.items():
        wav = config.SAMPLES_DIR / f"{name}.wav"
        wavfile.write(wav, SR, (_voice(**spec["params"]) * 32767).astype(np.int16))
        paths.append(str(wav))
    return paths


# ---------------------------------------------------------------- sequência de pose sintética
def make_pose_sequence(fps: int = 15, seconds: int = 20, seed: int = 5,
                       imbalance: bool = True, fall: bool = True) -> dict:
    """Keypoints COCO-17 (formato YOLOv8-Pose/OpenPose) de uma sessão de fisioterapia com
    agachamentos repetidos. Opcionalmente: perda de equilíbrio aos 9 s e queda aos 15 s."""
    do_imbalance, do_fall = imbalance, fall
    rng = np.random.default_rng(seed)
    W, H = 640, 480
    frames = []
    for i in range(fps * seconds):
        t = i / fps
        squat = (1 - np.cos(2 * np.pi * t / 3)) / 2  # 0=em pé, 1=agachado (ciclo de 3 s)
        knee_bend = 70 * squat
        cx, hip_y = 320, 240 + 60 * squat
        lean = 0.0
        asym = 0.0
        if do_imbalance and 9 <= t < 12:  # perda de equilíbrio: tronco inclina e um joelho cede
            lean, asym = 85 * np.sin(np.pi * (t - 9) / 3), 35
        fall = 0.0
        if do_fall and t >= 15:
            fall = min((t - 15) / 0.6, 1.0)  # queda em ~0,6 s
        def P(x, y):
            return [x + rng.normal(0, 1.5), y + rng.normal(0, 1.5), 0.9]
        kp = []
        head_y = hip_y - 170 + 20 * squat
        sh_y = hip_y - 120 + 10 * squat
        kp += [P(cx + lean, head_y)] * 1 + [P(cx - 8 + lean, head_y - 5), P(cx + 8 + lean, head_y - 5),
                                            P(cx - 15 + lean, head_y), P(cx + 15 + lean, head_y)]
        kp += [P(cx - 40 + lean, sh_y), P(cx + 40 + lean, sh_y)]
        kp += [P(cx - 55 + lean, sh_y + 60), P(cx + 55 + lean, sh_y + 60)]
        kp += [P(cx - 50 + lean, sh_y + 110), P(cx + 50 + lean, sh_y + 110)]
        kp += [P(cx - 25, hip_y), P(cx + 25, hip_y)]
        kL = np.radians(knee_bend + asym)
        kR = np.radians(knee_bend)
        kp += [P(cx - 25 + 60 * np.sin(kL), hip_y + 60 * np.cos(kL) + 30),
               P(cx + 25 + 60 * np.sin(kR), hip_y + 60 * np.cos(kR) + 30)]
        kp += [P(cx - 25, 440), P(cx + 25, 440)]
        kp = np.array(kp)
        if fall > 0:  # corpo rotaciona ~85° em torno dos tornozelos
            ang = np.radians(-85 * fall)  # cai para a esquerda
            pivot = np.array([cx, 440])
            xy = kp[:, :2] - pivot
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            kp[:, :2] = xy @ rot.T + pivot
        frames.append({"t": round(t, 3), "keypoints": kp.round(1).tolist(),
                       "objects": [{"label": "chair", "conf": 0.8, "box": [480, 300, 600, 460]}]})
    desc = "Sessão de fisioterapia sintética (agachamentos)" + (
        " com perda de equilíbrio e queda" if do_imbalance and do_fall else
        " com perda de equilíbrio" if do_imbalance else " com queda" if do_fall else " dentro do padrão")
    return {"fps": fps, "width": W, "height": H, "frames": frames, "description": desc + "."}


# cenários de vídeo de exemplo: arquivo -> (descrição, paciente, parâmetros)
VIDEO_SAMPLES = {
    "fisio_queda": ("Perda de equilíbrio (9 s) e queda (15 s)", "P-001", dict(imbalance=True, fall=True)),
    "fisio_desequilibrio": ("Perda de equilíbrio (9 s), sem queda", "P-002", dict(imbalance=True, fall=False)),
    "fisio_normal": ("Sessão dentro do padrão esperado", "P-003", dict(imbalance=False, fall=False)),
}
# consultas de áudio de exemplo: arquivo -> paciente
AUDIO_PATIENT = {"consulta_fadiga_dispneia": "P-002", "consulta_normal": "P-003"}


def generate_all() -> None:
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, (_, _, params) in VIDEO_SAMPLES.items():
        (config.SAMPLES_DIR / f"pose_{name}.json").write_text(json.dumps(make_pose_sequence(**params)))
    (config.SAMPLES_DIR / "patients.json").write_text(json.dumps(PATIENTS, ensure_ascii=False, indent=2))
    write_audio_samples()


if __name__ == "__main__":
    generate_all()
    print("Dados de exemplo gerados em", config.SAMPLES_DIR)
