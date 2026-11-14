"""Central de alertas: persiste, deduplica e notifica a equipe médica.

Todos os módulos (vídeo e áudio)
geram objetos `Alert` e os enviam para `AlertManager.emit()`, que:
  1. grava no banco SQLite (histórico / auditoria);
  2. evita duplicatas (mesmo paciente + origem + título em janela curta);
  3. avisa a equipe no Microsoft Teams / Slack quando o link de aviso está configurado.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config

SEVERITIES = {"INFO": 1, "ATENCAO": 2, "CRITICO": 3}
SEVERITY_LABEL = {"INFO": "🔵 Informativo", "ATENCAO": "🟠 Atenção", "CRITICO": "🔴 Crítico"}


@dataclass
class Alert:
    patient_id: str
    source: str          # VIDEO | AUDIO | FUSAO
    severity: str        # INFO | ATENCAO | CRITICO
    title: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"Severidade inválida: {self.severity}")


class AlertManager:
    def __init__(self, db_path: Path | str = config.ALERTS_DB, notify_url: str | None = None,
                 dedup_minutes: int = 10):
        self.db_path = str(db_path)
        self.notify_url = config.LINK_AVISO_EQUIPE if notify_url is None else notify_url
        self.dedup_minutes = dedup_minutes
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, created_at TEXT, patient_id TEXT, source TEXT,
                    severity TEXT, title TEXT, message TEXT, details TEXT,
                    acknowledged INTEGER DEFAULT 0, notified INTEGER DEFAULT 0)"""
            )

    def _conn(self):
        return sqlite3.connect(self.db_path)

    # ------------------------------------------------------------------ escrita
    def emit(self, alert: Alert) -> bool:
        """Registra o alerta. Retorna False se for duplicado."""
        now = datetime.now()
        window = (now - timedelta(minutes=self.dedup_minutes)).isoformat(timespec="seconds")
        with self._conn() as c:
            dup = c.execute(
                "SELECT 1 FROM alerts WHERE patient_id=? AND source=? AND title=? AND message=? "
                "AND created_at>=? LIMIT 1",
                (alert.patient_id, alert.source, alert.title, alert.message, window),
            ).fetchone()
            if dup:
                return False
            notified = self._notify(alert)
            c.execute(
                "INSERT INTO alerts (timestamp, created_at, patient_id, source, severity, title, "
                "message, details, notified) VALUES (?,?,?,?,?,?,?,?,?)",
                (alert.timestamp, now.isoformat(timespec="seconds"), alert.patient_id, alert.source,
                 alert.severity, alert.title, alert.message,
                 json.dumps(alert.details, ensure_ascii=False, default=str), int(notified)),
            )
        return True

    def emit_many(self, alerts: list[Alert]) -> int:
        return sum(self.emit(a) for a in alerts)

    def acknowledge(self, alert_id: int) -> None:
        with self._conn() as c:
            c.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (int(alert_id),))

    def clear(self) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM alerts")

    # ------------------------------------------------------------------ leitura
    def to_dataframe(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql_query("SELECT * FROM alerts ORDER BY id DESC", c)
        # coluna sempre presente (mesmo sem alertas) para ordenações na interface
        df["sev_rank"] = df["severity"].map(SEVERITIES).astype("Int64") if not df.empty else pd.Series(dtype="Int64")
        return df

    # ------------------------------------------------------------------ notificação
    def _notify(self, alert: Alert) -> bool:
        """Envia alertas ATENCAO/CRITICO para o Teams/Slack pelo link de aviso configurado."""
        if not self.notify_url or SEVERITIES[alert.severity] < 2:
            return False
        text = (f"{SEVERITY_LABEL[alert.severity]} | Paciente {alert.patient_id} | {alert.source}\n"
                f"**{alert.title}** — {alert.message}")
        try:
            import requests
            r = requests.post(self.notify_url, json={"text": text, "content": text}, timeout=5)
            return r.ok
        except Exception:
            return False


def alerts_to_dicts(alerts: list[Alert]) -> list[dict]:
    return [asdict(a) for a in alerts]
