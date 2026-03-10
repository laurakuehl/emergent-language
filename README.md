# emergent-language
An implementation of Emergence of Grounded Compositional Language in Multi-Agent Populations by Igor Mordatch and Pieter Abbeel

## Train
Run training in an environment with PyTorch installed:

```bash
python3 train.py --n-epochs 300 --save-model-weights model_weights.pt --seed 0
```

Useful options:
- `--no-utterances`: disable communication channel
- `--penalize-words`: add word-usage regularization
- `--metrics-jsonl train_metrics.jsonl`: save per-epoch metrics for sample-efficiency analysis
- `--holdout-combos "0-1,2-0" --holdout-mode exclude`: train while excluding selected color-shape goal combos

## Evaluate Games + Communication Ablations
Run deterministic rollouts (no learning) and compare channel conditions:

```bash
python3 evaluate.py \
  --model-weights model_weights.pt \
  --episodes 100 \
  --seed 0 \
  --channel-modes normal muted shuffled random \
  --save-episodes eval_episodes.jsonl \
  --save-messages eval_messages.jsonl \
  --summary-json eval_summary.json
```

Metrics include:
- success rate
- average final distance to goal
- per-agent episode cost

## Analyze Emergent Language
Analyze logged message records:

```bash
python3 language_analysis.py \
  --messages-jsonl eval_messages.jsonl \
  --summary-json language_summary.json
```

Output includes:
- token entropy
- mutual information with target color/shape/agent
- topographic similarity
- meaning-token purity
- seen vs holdout splits (if holdout combos are enabled)

## Replay Episodes (Qualitative Inspection)
Print a step-by-step replay:

```bash
python3 replay.py \
  --episodes-jsonl eval_episodes.jsonl \
  --episode-id 0 \
  --channel-mode normal
```

Create a visual animated replay (2D coordinates with colored agents and landmarks):

```bash
python3 visual_replay.py \
  --episodes-jsonl eval_episodes.jsonl \
  --episode-id 0 \
  --channel-mode normal \
  --show-trails \
  --show-tokens \
  --save replay.gif
```

## Run Full Ablations
Automate train + eval + language analysis across multiple seeds and variants:

```bash
python3 run_ablations.py \
  --output-dir ablation_runs \
  --seeds 0 1 2 \
  --n-epochs 200 \
  --eval-episodes 100
```

To include compositional holdout generalization:

```bash
python3 run_ablations.py \
  --output-dir ablation_runs_holdout \
  --seeds 0 1 2 \
  --holdout-combos "0-1,2-0"
```

## Main Files
- `train.py`: training harness
- `evaluate.py`: deterministic evaluation and communication ablations
- `language_analysis.py`: emergent-language metrics
- `replay.py`: human-readable episode replay
- `visual_replay.py`: animated 2D replay for episodes
- `run_ablations.py`: end-to-end experiment runner
- `modules/game.py`: world dynamics and costs
- `modules/agent.py`: multi-agent policy rollouts
- `configs.py`: configuration defaults and parsing
- `visualize.py`: PyTorch graph visualization utility
