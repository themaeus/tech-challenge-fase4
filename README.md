# 🏥 Monitoramento Multimodal de Pacientes — Tech Challenge Fase 4

Sistema que analisa **vídeos** (sessões de fisioterapia ou cirurgias) e **áudios** (consultas) para identificar
sinais precoces de risco e **alertar a equipe médica**. Interface gráfica em **Streamlit**.

| Entrega | Onde está |
|---|---|
| Análise de vídeo (OpenPose / YOLOv8) + relatório automático de desvios | `src/video_analysis.py` · página **🎥 Análise de Vídeo** |
| Análise de áudio: alterações vocais (fadiga, dificuldade respiratória, disartria) | `src/audio_analysis.py` · página **🎙️ Análise de Áudio** |
| Alertas para a equipe médica | `src/alerts.py` · página **🚨 Central de Alertas** (+ aviso opcional no Teams/Slack) |
| Fusão multimodal (vídeo + áudio) | `src/fusion.py` · **Painel Central** |
| Relatório técnico | [`docs/RELATORIO_TECNICO.md`](docs/RELATORIO_TECNICO.md) |

## ▶️ Como executar

Requer **Python 3.10+**.

**Windows (mais fácil):** dê dois cliques em **`iniciar.bat`**. Na primeira vez ele cria o ambiente
virtual e instala as dependências; nas próximas, abre a aplicação direto no navegador.

**Manual (qualquer sistema):**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python -m src.sample_data        # gera os dados de exemplo (também é feito automaticamente)
streamlit run app.py             # abre http://localhost:8501
```

No **Painel Central**, clique em **“▶️ Executar demonstração completa”**: os módulos de vídeo e áudio rodam nos
dados de exemplo, os alertas aparecem na Central e o risco fundido de cada paciente é calculado.

> Os pesos `yolov8n.pt` e `yolov8n-pose.pt` são baixados automaticamente pelo `ultralytics`
> na primeira análise de vídeo real (CPU é suficiente para vídeos curtos).

### Testes (opcional)

```bash
pip install -r requirements-dev.txt
pytest -q                          # 9 testes dos módulos de análise
python scripts/gerar_resultados.py # regenera figuras e números do relatório (docs/img, docs/resultados.json)
```

### Notificação externa (opcional)

Copie `.env.example` para `.env` e preencha `LINK_AVISO_EQUIPE` com o link de entrada de um canal do
Microsoft Teams ou do Slack para que os alertas de atenção e críticos também cheguem lá.

## 📁 Estrutura

```
app.py                       Painel Central (risco por paciente, últimos alertas, demonstração completa)
iniciar.bat                  inicialização automática no Windows
pages/                       páginas da interface (vídeo, áudio, central de alertas)
src/
  video_analysis.py          YOLOv8-Pose + YOLOv8, métricas biomecânicas, eventos, relatório
  audio_analysis.py          biomarcadores vocais (pausas, F0, jitter, shimmer, HNR, ruído respiratório)
  fusion.py                  fusão multimodal (late fusion) e alerta de risco combinado
  alerts.py                  central de alertas (SQLite, deduplicação, aviso no Teams/Slack)
  sample_data.py             dados sintéticos de vídeo (keypoints) e áudio
  pipeline.py                orquestração da demonstração completa
data/samples/                dados de exemplo gerados
docs/                        relatório técnico e figuras
tests/                       testes automatizados
```

## 📊 Usando dados reais

- **Vídeos**: qualquer mp4/avi/mov de sessão de fisioterapia ou procedimento (ex.: vídeos livres do Pexels).
- **Keypoints do OpenPose**: converta para o JSON `{fps, width, height, frames:[{t, keypoints[17][3], objects}]}`
  (ordem COCO-17) e envie na página de vídeo.
- **Áudio**: wav/mp3/m4a/ogg/flac (ex.: gravações próprias ou trechos de fala do AudioSet).

> ⚠️ Projeto acadêmico. Limites e faixas de referência são didáticos — não é um dispositivo médico.
