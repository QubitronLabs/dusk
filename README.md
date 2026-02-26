# 🌅 Dusk

**A Diffusion Language Model by Qubitron Labs**

## Overview

Dusk is a diffusion-based language model that generates text through iterative denoising rather than autoregressive token-by-token generation. All tokens crystallize simultaneously from noise — like meaning emerging from chaos.

Key innovations:
1. **Unified diffusion architecture** with a shared probabilistic formulation and modality-agnostic design
2. **Mixed long chain-of-thought (CoT) fine-tuning** strategy with unified CoT format across modalities
3. **UniGRPO** — a unified policy-gradient-based RL algorithm tailored for diffusion foundation models

## Model Variants

| Model | Description |
|-------|-------------|
| **Dusk-8B** | Base model — text generation, image generation, image captioning |
| **Dusk-8B-CoT** | Mixed CoT fine-tuned — complex reasoning across modalities |
| **Dusk-8B-Max** | Post-UniGRPO RL — best quality (coming soon) |

## Quick Start

### Installation

```bash
git clone https://github.com/QubitronLabs/dusk.git
cd dusk
pip install -r requirements.txt
```

### Text Generation

```bash
python generate.py --model_path QubitronLabs/dusk-8b --prompt "The meaning of life is"
```

### Web Interface

```bash
# Start backend (mock mode for testing)
cd server && python main.py

# Start frontend
cd web && bun dev
```

### Real Model (requires quantized weights)

```bash
cd server
python main.py --real --model QubitronLabs/dusk-8b-int4
```

## Training

### Stage 1: Visual Pretraining
```bash
accelerate launch --config_file path/to/accelerate_config \
  training/train_dusk.py config=configs/dusk_pretraining_stage1_dusk_instruct.yaml
```

### Stage 2: Image-Text Training
```bash
accelerate launch --config_file path/to/accelerate_config \
  training/train_dusk_stage2.py config=configs/dusk_pretraining_stage2_dusk_instruct.yaml
```

### Stage 3: Instruction Following
```bash
accelerate launch --config_file path/to/accelerate_config \
  training/train_dusk_stage3.py config=configs/dusk_pretraining_stage3_dusk_instruct.yaml
```

## Architecture

Dusk uses a masked diffusion transformer architecture:
- **Encoder**: Standard transformer with rotary position embeddings
- **Diffusion**: Forward process masks tokens; reverse process predicts masked tokens
- **Sampling**: Confidence-based iterative unmasking over T steps

## Project Structure

```
dusk/
├── models/               # Core model code
│   ├── modeling_dusk.py          # Main model (DuskModelLM)
│   ├── modeling_dusk_base.py     # Base transformer (DuskBaseLM)
│   └── configuration_dusk.py    # Model config
├── server/               # FastAPI backend
├── web/                  # Next.js frontend
├── training/             # Training scripts
├── configs/              # Training configs
├── evaluation/           # Eval scripts
├── notebooks/            # Colab notebooks
└── scripts/              # Utility scripts
```

## License

Apache 2.0

## Author

**Dhiraj Lochib** — [Qubitron Labs](https://github.com/QubitronLabs)




