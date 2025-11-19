# Snake AI – Algoritmo Genético + Rede Neural

Projeto didático em Python que treina agentes para jogar o clássico jogo da cobrinha (Snake),
usando algoritmo genético para otimizar os pesos de uma rede neural simples.

## Destaques do projeto

- Rede neural feedforward enxuta (11 × 16 × 16 × 3) controlando as ações da cobra.
- Algoritmo genético com **fitness sharing**, **Hall of Fame** e **cataclysm** para evitar ótimos locais.
- Penalidades específicas (ex.: colisão com o próprio corpo) e função de fitness com bônus por consistência.
- Treinamento paralelo com `multiprocessing`, checkpoints automáticos e gráficos gerados durante o treino.
- Scripts utilitários para reproduzir experimentos, replays e comparação de resultados.

## Estrutura

    snake_ai/
      snake_ai/
        env/
        agents/
        training/
        visualization/
        config/
        scripts/
        tests/
      results/
      requirements.txt
      README.md

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ou
.venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Treinar experimento

Dentro da pasta `snake_ai` (raiz do projeto):

```bash
python -m snake_ai.scripts.treinar_experimento --config snake_ai/config/experimento_default.yaml
```

No Windows/PowerShell, tudo em uma linha:

```powershell
python -m snake_ai.scripts.treinar_experimento --config "snake_ai/config/experimento_default.yaml"
```

Isso vai treinar, salvar logs em `results/<experimento>/logs/`,
o melhor modelo em `results/<experimento>/models/` e gráficos em `results/<experimento>/plots/`.

Principais opções do arquivo `experimento_default.yaml`:

- `env`: parâmetros do grid e recompensas (inclui `self_body_penalty` para punir colisões no corpo).
- `ga`: hiperparâmetros do algoritmo genético (população 250, 5000 gerações, mutações fortes, etc.).
- `optimization`: paralelização, early stopping, checkpointing, plot automático e mutação adaptativa.

Com `visual.show_live: true`, a cada N gerações será aberta uma janela `pygame`
mostrando o melhor indivíduo daquela geração jogando Snake.
Use **SPACE** para pausar/continuar e **ESC** para sair da janela.

## Replay do melhor agente

```bash
python -m snake_ai.scripts.replay_melhor_agente --experiment snake_ga_default --episodes 5 --speed 1.5
```

## Comparar experimentos

```bash
python -m snake_ai.scripts.comparar_experimentos --experiments snake_ga_default outro_experimento --metric fitness_max
```

## Resultados recentes

| Experimento                           | Gerações | Fitness máx. | Maçãs máx. | Observações                                  |
|--------------------------------------|----------|--------------|------------|----------------------------------------------|
| `snake_ga_treino_pesado_20251119_000048` | 381      | **31.63**     | **21.09**   | População 250, fitness sharing + cataclysm    |
| `snake_ga_treino_pesado_20251118_224936` | 2541     | 39.87         | 26.10       | Versão anterior (sem fitness sharing completo) |

Os valores vêm de `results/<experimento>/logs/training_log.csv`, e os gráficos correspondentes estão em `results/<experimento>/plots/fitness.png` e `apples.png`.

Para reproduzir um desses treinos basta copiar o diretório em `results/` e rodar o comando de replay com o nome do experimento.
