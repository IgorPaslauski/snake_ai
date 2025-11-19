# Snake AI – Algoritmo Genético + Rede Neural

Projeto didático em Python que treina agentes para jogar o clássico jogo da cobrinha (Snake),
usando algoritmo genético para otimizar os pesos de uma rede neural simples.

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
