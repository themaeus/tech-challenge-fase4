"""Análise de áudio de consultas: biomarcadores vocais extraídos do sinal de voz.

Indicadores acústicos extraídos (sem dependências pesadas — apenas numpy/scipy):
  - energia (RMS), proporção de pausas e número de pausas longas;
  - taxa de articulação (núcleos silábicos por segundo de fala);
  - F0 (autocorrelação): média e variabilidade (monotonia);
  - jitter, shimmer e HNR (qualidade/estabilidade vocal);
  - ruído respiratório nas pausas.
Eles alimentam três escores interpretáveis (0–1): fadiga, dificuldade respiratória e disartria.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks

from . import config
from .alerts import Alert

SR = 16000


def to_wav16k(path: str | Path) -> str:
    """Converte qualquer áudio (mp3, m4a, ogg, wav estéreo...) para WAV PCM 16 kHz mono."""
    path = Path(path)
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = None
    if ff:
        out = Path(tempfile.gettempdir()) / f"{path.stem}_16k.wav"
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR),
                            "-sample_fmt", "s16", str(out)])
        if r.returncode == 0:
            return str(out)
    if path.suffix.lower() != ".wav":
        raise RuntimeError("ffmpeg não encontrado: envie um arquivo .wav ou instale imageio-ffmpeg")
    return str(path)


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    sr, y = wavfile.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if np.abs(y).max() > 1.5:
        y /= 32768.0
    if sr != SR:  # reamostragem simples
        from scipy.signal import resample_poly
        y = resample_poly(y, SR, sr).astype(np.float32)
        sr = SR
    return y, sr


def _frames(y, sr, win=0.03, hop=0.01):
    w, h = int(win * sr), int(hop * sr)
    n = 1 + max(0, (len(y) - w) // h)
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    return y[idx] * np.hanning(w), h


def acoustic_features(y: np.ndarray, sr: int = SR) -> dict:
    fr, hop = _frames(y, sr)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    thr = max(db.max() - 30, np.percentile(db, 20) + 6)
    voiced = db > thr
    hop_s = hop / sr
    duration = len(y) / sr

    # pausas (sequências não vozeadas)
    pauses, run = [], 0
    for v in voiced:
        if not v:
            run += 1
        elif run:
            pauses.append(run * hop_s)
            run = 0
    pauses = [p for p in pauses if p >= 0.15]
    speech_time = voiced.sum() * hop_s

    # núcleos silábicos: picos do envelope de energia suavizado
    env = np.convolve(rms, np.ones(7) / 7, mode="same")
    peaks, _ = find_peaks(env, height=10 ** (thr / 20), distance=int(0.12 / hop_s), prominence=env.max() * 0.05)
    artic_rate = len(peaks) / max(speech_time, 1e-3)

    # F0 por autocorrelação + HNR
    f0, hnr, amps = [], [], []
    lo, hi = int(sr / 400), int(sr / 70)
    for f, v, a in zip(fr, voiced, rms):
        if not v:
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lo + np.argmax(ac[lo:hi])
        r = ac[lag]
        if r > 0.45:
            f0.append(sr / lag)
            hnr.append(10 * np.log10(r / max(1 - r, 1e-3)))
            amps.append(a)
    f0, amps = np.array(f0), np.array(amps)
    periods = 1 / f0 if len(f0) > 2 else np.array([1.0, 1.0])
    jitter = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods) * 100) if len(f0) > 2 else 0.0
    shimmer = float(np.mean(np.abs(np.diff(amps))) / np.mean(amps) * 100) if len(amps) > 2 else 0.0

    # ruído respiratório: energia nas pausas relativa à fala
    unv = db[~voiced]
    breath = float(np.percentile(unv, 75) - np.percentile(db[voiced], 50)) if voiced.any() and len(unv) else -60.0

    return {
        "duracao_s": round(duration, 2),
        "tempo_fala_s": round(float(speech_time), 2),
        "proporcao_pausas": round(float(1 - speech_time / max(duration, 1e-3)), 3),
        "pausas_longas_por_min": round(sum(p > 0.5 for p in pauses) / max(duration / 60, 1e-3), 1),
        "taxa_articulacao_sil_s": round(float(artic_rate), 2),
        "energia_media_db": round(float(db[voiced].mean()) if voiced.any() else -90.0, 1),
        "f0_medio_hz": round(float(np.median(f0)) if len(f0) else 0.0, 1),
        "f0_variabilidade_st": round(float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 2 else 0.0, 2),
        "jitter_pct": round(min(jitter, 30), 2),
        "shimmer_pct": round(min(shimmer, 100), 2),
        "hnr_db": round(float(np.mean(hnr)) if len(hnr) else 0.0, 1),
        "ruido_respiratorio_db": round(breath, 1),
    }


def _ramp(x, lo, hi):
    """0 quando x<=lo, 1 quando x>=hi (linear entre eles)."""
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def vocal_risk(f: dict) -> dict:
    """Escores 0–1 e justificativas. Faixas de referência de fala adulta conversacional."""
    fatigue = np.mean([_ramp(f["proporcao_pausas"], 0.25, 0.55), _ramp(-f["f0_variabilidade_st"], -2.5, -0.8),
                       _ramp(f["pausas_longas_por_min"], 6, 25), _ramp(-f["taxa_articulacao_sil_s"], -4.0, -2.0)])
    resp = np.mean([_ramp(f["pausas_longas_por_min"], 8, 30), _ramp(f["ruido_respiratorio_db"], -40, -15),
                    _ramp(f["proporcao_pausas"], 0.3, 0.6)])
    dys = np.mean([_ramp(f["jitter_pct"], 1.0, 3.0), _ramp(f["shimmer_pct"], 6, 20), _ramp(-f["hnr_db"], -15, -5)])
    why = []
    if f["proporcao_pausas"] > 0.35:
        why.append(f"muitas pausas ({f['proporcao_pausas']:.0%} do tempo)")
    if f["f0_variabilidade_st"] < 1.5:
        why.append(f"voz monótona (variação de F0 {f['f0_variabilidade_st']:.1f} semitons)")
    if f["jitter_pct"] > 1.5:
        why.append(f"jitter elevado ({f['jitter_pct']:.1f}%)")
    if f["shimmer_pct"] > 10:
        why.append(f"shimmer elevado ({f['shimmer_pct']:.1f}%)")
    if f["ruido_respiratorio_db"] > -30:
        why.append("ruído respiratório audível nas pausas")
    return {"fadiga": round(float(fatigue), 2), "dificuldade_respiratoria": round(float(resp), 2),
            "disartria": round(float(dys), 2), "justificativas": why}


def analyze_consultation(audio_path: str | Path, patient_id: str) -> dict:
    """Pipeline de uma consulta: áudio -> características acústicas -> escores vocais -> alertas."""
    wav = to_wav16k(audio_path)
    y, sr = load_audio(wav)
    feats = acoustic_features(y, sr)
    risk = vocal_risk(feats)

    alerts = []
    labels = {"fadiga": "Fadiga vocal", "dificuldade_respiratoria": "Dificuldade respiratória na fala",
              "disartria": "Possível disartria"}
    for k, lab in labels.items():
        if risk[k] >= 0.6:
            alerts.append(Alert(patient_id, "AUDIO", "CRITICO" if risk[k] >= 0.8 else "ATENCAO", lab,
                                f"Escore {risk[k]:.2f}: " + "; ".join(risk["justificativas"])))
    return {"features": feats, "risk": risk, "alerts": alerts, "signal": y[::40], "sr": sr / 40}
