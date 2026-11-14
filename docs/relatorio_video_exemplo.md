# Relatório automático de vídeo — Fisioterapia
- **Paciente:** P-001
- **Arquivo:** Sessão de fisioterapia sintética (agachamentos) com perda de equilíbrio e queda.
- **Duração analisada:** 19.9 s (300 quadros a 15 fps)
- **Pessoa detectada em:** 100% dos quadros
- **Modelos:** YOLOv8-Pose (17 keypoints COCO/OpenPose) + YOLOv8 (detecção de objetos)

## Indicadores biomecânicos
| Métrica | Mín | Média | Máx |
|---|---|---|---|
| joelho esq (°) | 77 | 133 | 180 |
| joelho dir (°) | 96 | 137 | 180 |
| cotovelo esq (°) | 151 | 160 | 172 |
| cotovelo dir (°) | 152 | 160 | 170 |
| inclinacao tronco (°) | 0 | 24 | 86 |

## Desvios / falhas detectados
- 🟠 **Assimetria articular** (9.0s–10.0s): Diferença entre joelhos de até 43° (limite 25°) — possível compensação/dor
- 🟠 **Movimento brusco** (9.0s–9.1s): joelho esq: velocidade angular 622°/s
- 🟠 **Perda de equilíbrio** (9.7s–11.4s): Inclinação do tronco de até 38° (limite 25°)
- 🟠 **Assimetria articular** (11.1s–12.0s): Diferença entre joelhos de até 39° (limite 25°) — possível compensação/dor
- 🟠 **Movimento brusco** (12.0s–12.1s): joelho esq: velocidade angular 576°/s
- 🔴 **Queda** (15.5s–19.9s): Corpo na horizontal (inclinação > 60°) após descida rápida do quadril (447 px/s). Acionar equipe imediatamente.

## Conclusão
⚠️ Procedimento com **falhas críticas** — equipe notificada.