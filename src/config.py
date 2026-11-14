"""Configurações centrais do projeto (lidas do arquivo .env)."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv é opcional
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
ALERTS_DB = OUTPUT_DIR / "alerts.db"

if load_dotenv:
    load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# link para avisar a equipe no Microsoft Teams / Slack (opcional)
LINK_AVISO_EQUIPE = _env("LINK_AVISO_EQUIPE")

YOLO_DET_MODEL = _env("YOLO_DET_MODEL", "yolov8n.pt")
YOLO_POSE_MODEL = _env("YOLO_POSE_MODEL", "yolov8n-pose.pt")

