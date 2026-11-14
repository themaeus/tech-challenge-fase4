"""Testes automatizados dos módulos de análise (executar: pytest -q)."""
import json

import pytest

from src import audio_analysis, config, fusion, pipeline, sample_data, video_analysis
from src.alerts import Alert, AlertManager


@pytest.fixture(scope="session", autouse=True)
def data():
    sample_data.generate_all()


def _events(name, **cfg):
    seq = json.loads((config.SAMPLES_DIR / f"pose_{name}.json").read_text())
    return video_analysis.analyze_sequence(seq, video_analysis.VideoConfig(**cfg))


def test_video_detects_fall_and_imbalance():
    metrics, events = _events("fisio_queda")
    kinds = {e.kind for e in events}
    assert "Queda" in kinds and "Perda de equilíbrio" in kinds
    fall = next(e for e in events if e.kind == "Queda")
    assert 14.5 <= fall.t_start <= 16.5 and fall.severity == "CRITICO"


def test_video_imbalance_without_fall():
    _, events = _events("fisio_desequilibrio")
    kinds = {e.kind for e in events}
    assert "Perda de equilíbrio" in kinds and "Queda" not in kinds


def test_video_normal_session_has_no_events():
    _, events = _events("fisio_normal")
    assert not [e for e in events if e.severity != "INFO"]


def test_video_report():
    seq = sample_data.make_pose_sequence()
    metrics, events = video_analysis.analyze_sequence(seq)
    report = video_analysis.build_report(seq, metrics, events, video_analysis.VideoConfig(), "P-001")
    assert "Queda" in report and "Conclusão" in report


def test_video_surgery_instrument_count():
    seq = sample_data.make_pose_sequence(seconds=6, imbalance=False, fall=False)
    for i, fr in enumerate(seq["frames"]):
        n = 2 if i < len(seq["frames"]) // 2 else 1  # um instrumento "some" do campo
        fr["objects"] = [{"label": "scissors", "conf": .9, "box": [300, 200, 320, 230]}] * n
    _, events = video_analysis.analyze_sequence(seq, video_analysis.VideoConfig(mode="Cirurgia"))
    assert any(e.kind == "Contagem de instrumentos divergente" for e in events)


def test_audio_vocal_markers():
    S = config.SAMPLES_DIR
    normal = audio_analysis.analyze_consultation(S / "consulta_normal.wav", "P-003")
    tired = audio_analysis.analyze_consultation(S / "consulta_fadiga_dispneia.wav", "P-002")
    assert tired["risk"]["fadiga"] > normal["risk"]["fadiga"]
    assert tired["risk"]["dificuldade_respiratoria"] > normal["risk"]["dificuldade_respiratoria"]
    assert not normal["alerts"] and tired["alerts"]


def test_alert_manager_dedup_and_fusion(tmp_path):
    m = AlertManager(tmp_path / "a.db", notify_url="")
    assert m.emit(Alert("P-9", "VIDEO", "CRITICO", "Queda", "teste"))
    assert not m.emit(Alert("P-9", "VIDEO", "CRITICO", "Queda", "teste"))  # duplicado
    m.emit(Alert("P-9", "AUDIO", "ATENCAO", "Fadiga vocal", "x"))
    risk = fusion.compute_risk(m.to_dataframe())
    assert risk.iloc[0]["risco"] > 0.8
    assert fusion.fusion_alerts(risk)
    m.acknowledge(int(m.to_dataframe().iloc[0]["id"]))
    assert fusion.compute_risk(m.to_dataframe()).iloc[0]["risco"] < risk.iloc[0]["risco"]


def test_empty_alerts_dataframe_has_sev_rank(tmp_path):
    df = AlertManager(tmp_path / "vazio.db", notify_url="").to_dataframe()
    assert "sev_rank" in df.columns
    df.sort_values(["acknowledged", "sev_rank", "id"])  # usado no Painel Central


def test_full_demo(tmp_path):
    m = AlertManager(tmp_path / "demo.db", notify_url="")
    pipeline.run_full_demo(m, log=lambda *_: None)
    risk = fusion.compute_risk(m.to_dataframe(), list(sample_data.PATIENTS)).set_index("patient_id")
    assert risk.loc["P-001", "risco"] >= 0.8          # queda no vídeo
    assert risk.loc["P-002", "modalidades_alteradas"] == 2  # vídeo + áudio -> fusão
    assert risk.loc["P-003", "risco"] == 0             # paciente sem alterações
