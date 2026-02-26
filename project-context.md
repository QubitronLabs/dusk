# DUSK — Project Context

> **Last Updated:** 2026-02-26
> **Status:** Phase 2 — Dusk Base Model Integration
> **AI Coding Agent:** Read this file FIRST before touching any code.

---

## 1. Project Vision

**DUSK (Discrete Unified Sparse Knowledge)** is a diffusion-based language model startup project. After validating that training from scratch requires prohibitive compute (Chinchilla scaling: 670M model needs ~50B byte-tokens), we pivoted to using **Dusk** (Multimodal Masked Diffusion with Alignment) as our pretrained base model.

### Why Diffusion LMs?
- **Parallel decoding**: Generate all tokens simultaneously, not one-by-one
- **Global planning**: The model sees the full output at every step — no "reversal curse"
- **Bidirectional reasoning**: Every token attends to every other token
- **Multimodality**: Text + images in one unified architecture
- **Speed**: Commercial diffusion LMs (Mercury 2, Gemini Diffusion) achieve 1000-2000 tok/s

### Why Dusk?
- **Only open-source multimodal diffusion model** — text reasoning + image understanding + image generation in ONE model
- NeurIPS 2025 paper, MIT license, active development
- Built on Dusk-8B-Instruct (strong text base) + Show-o image tokenizer (MAGVITv2)
- Already has UniGRPO RL alignment infrastructure
- Competitive benchmarks: MMLU 68.4, GSM8K 73.4, MATH 36.0
- Clear training recipe: pretrain → mixed CoT SFT → UniGRPO RL

---

## 2. Competitor Landscape (as of Feb 2026)

| Model | Params | Open? | Modality | Speed | Notes |
|-------|--------|-------|----------|-------|-------|
| **Dusk** (QubitronLabs) | 8B | ✅ MIT | Text+Image | ~100 tok/s | Our base. NeurIPS 2025 |
| **Dream 7B** (Dream-org) | 7B | ✅ | Text only | ~100 tok/s | CART schedule, from Qwen 2.5 |
| **Dusk 2.0/2.1** (Ant Group) | 16B-100B MoE | ✅ | Text only | 535-935 tok/s | Block-level WSD, Token Editing |
| **Mercury 2** (Inception Labs) | ? | ❌ API | Text only | 1000 tok/s | $$$, closed source |
| **Gemini Diffusion** (Google) | ? | ❌ | Text only | 1000-2000 tok/s | Experimental, closed |
| **CDLM** (Together AI) | Various | ✅ | Text only | 14.5x speedup | Consistency training for dLLMs |

**Our Edge**: Dusk is the ONLY open multimodal diffusion model. Everyone else is text-only or closed-source.

---

## 3. Dusk Architecture Deep Dive

### 3.1 Model Structure
```
Dusk-8B = Dusk-8B-Instruct (text backbone) + MAGVITv2 (image tokenizer)

Text tokens:  BPE tokenizer (vocab_size = 126,464, from LLaMA 3)
Image tokens: MAGVITv2 VQ codebook (codebook_size = 8,192)
Total vocab:  134,656 (text + image + special tokens)
Mask token:   ID 126,336 (used for diffusion masking)

Special tokens (reserved IDs):
  <|soi|> = 126084  (start of image)
  <|eoi|> = 126085  (end of image)
  <|t2i|> = 126088  (text-to-image task)
  <|mmu|> = 126089  (multimodal understanding task)
  <|r2i|> = 126094  (reasoning-to-image task)
  [iPAD]  = 126093  (image padding)
```

### 3.2 Class Hierarchy
```
DuskBaseLM (models/modeling_dusk.py, 1500 lines)
  └─ DuskModelLM (models/modeling_dusk.py, 657 lines)
       ├─ t2i_generate()          — Text→Image generation (MaskGIT-style)
       ├─ mmu_generate()          — Multimodal understanding (text generation with image input)
       ├─ mmu_generate_fast()     — Faster MMU with early stopping
       ├─ forward_process()       — Training forward: t2i + lm + mmu losses
       ├─ forward_process_with_r2i() — Training forward with reasoning-to-image
       └─ forward_t2i()           — Text-to-image only forward
```

### 3.3 Diffusion Mechanism
- **Forward process**: Randomly mask tokens with probability `p` sampled from a schedule
- **Reverse process**: Iteratively unmask tokens over `steps` denoising steps
- **Noise schedule**: Cosine (for images), Linear (for text)
- **Remasking strategies**: `low_confidence` (remask least confident) or `random`
- **Gumbel noise**: Added to logits for diverse sampling (float64 for stability)
- **Semi-autoregressive**: Text uses block-by-block generation (block_length < gen_length)
- **Classifier-free guidance**: For image generation (guidance_scale > 0)

### 3.4 Image Pipeline
```
Image (512x512) → MAGVITv2 Encoder → 1024 discrete tokens (32x32 grid)
1024 tokens → MAGVITv2 Decoder → Image (512x512)

Codebook size: 8,192
Resolution: 512x512 (or 256x256 for stage 1)
```

### 3.5 Prompting Format
```
# Text-to-Image:
[iPAD]...[iPAD] <|t2i|> <bos> text_1 ... text_n <eos> <|soi|> img_1 ... img_1024 <|eoi|>

# Multimodal Understanding:
<|mmu|> <|soi|> img_1 ... img_1024 <|eoi|> <bos> user_text ... <eos>

# Language Modeling:
<bos> text_1 ... text_n <eos>

# Chat template (LLaMA 3 style):
<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n
```

---

## 4. Codebase Map

### Source Tree
```
DLLM/
├── project-context.md          ← THIS FILE
├── tasks.md                    ← Detailed task tracking
├── README.md                   ← Dusk original README
├── requirements.txt            ← Python dependencies
├── LICENSE                     ← MIT License
│
├── models/                     ← Core model code
│   ├── __init__.py             ← Exports: MAGVITv2, DuskModelLM, DuskConfig, sampling
│   ├── modeling_dusk.py       ← Dusk model (657 lines) — generation + training forward
│   ├── modeling_dusk.py       ← Dusk base model (1500 lines) — transformer blocks
│   ├── configuration_dusk.py  ← Model config dataclasses (464 lines)
│   ├── modeling_magvitv2.py    ← MAGVITv2 VQ-VAE (440 lines) — image tokenizer
│   ├── common_modules.py       ← Conv blocks, attention, up/downsampling (358 lines)
│   ├── modeling_utils.py       ← ConfigMixin, ModelMixin base classes
│   ├── sampling.py             ← Mask schedules, Gumbel noise, top-k/top-p filtering
│   ├── lr_schedulers.py        ← Learning rate schedulers
│   ├── misc.py                 ← Miscellaneous utilities
│   ├── logging.py              ← Logging utilities
│   └── training_utils.py       ← Training utility functions
│
├── training/                   ← Training scripts
│   ├── train_dusk.py          ← Stage 1.1: ImageNet pretraining (984 lines)
│   ├── train_dusk_stage2.py   ← Stage 1.2: Image-Text pretraining
│   ├── train_dusk_stage3.py   ← Stage 1.3: Text instruction following
│   ├── train_dusk_cot_sft.py  ← Stage 2.1: Mix-CoT text SFT
│   ├── train_dusk_stage4.py   ← Stage 2.2: Mix-CoT multimodal SFT
│   ├── data.py                 ← Dataset classes (Text2ImageDataset, WebDataset)
│   ├── prompting_utils.py      ← UniversalPrompting — formats inputs for all tasks
│   ├── utils.py                ← Config parsing, image transforms, masking utils
│   ├── optimizer.py            ← Optimizer setup
│   ├── imagenet_dataset.py     ← ImageNet dataset loader
│   └── questions.json          ← Caption prompts for MMU training
│
├── configs/                    ← Training configs (OmegaConf YAML)
│   ├── dusk_demo.yaml                                ← Demo/inference config
│   ├── dusk_pretraining_stage1_dusk_instruct.yaml   ← Stage 1.1 config
│   ├── dusk_pretraining_stage2_dusk_instruct.yaml   ← Stage 1.2 config
│   ├── dusk_pretraining_stage3_dusk_instruct.yaml   ← Stage 1.3 config
│   ├── dusk_pretraining_stage3_dusk_instruct_512_cot.yaml ← Stage 2.1 config
│   └── dusk_pretraining_stage4_dusk_instruct.yaml   ← Stage 2.2 config
│
├── generate.py                 ← Text generation script (standalone)
├── inference_mmu.py            ← Multimodal understanding inference
├── inference_t2i.py            ← Text-to-image inference
├── app.py                      ← Gradio demo app (894 lines)
│
├── accelerate_configs/         ← HuggingFace Accelerate configs
│   ├── 1_gpu.yaml
│   ├── 1_node_8_gpus_deepspeed_zero2.yaml
│   ├── 1_node_8_gpus_deepspeed_zero3.yaml
│   └── 8_node_8_gpus_deepspeed_zero2.yaml
│
├── evaluation/                 ← Evaluation scripts
│   ├── eval.md
│   ├── commands.sh
│   ├── lm/                     ← LM evaluation
│   │   ├── eval_dusk.py
│   │   ├── eval.sh
│   │   └── generate.py
│   └── VLMEvalKit/             ← Vision-Language evaluation
│
├── validation_prompts/         ← Test prompts
│   ├── text2image_prompts.txt
│   ├── imagenet_prompts.txt
│   ├── quantative.txt
│   └── test.txt
│
├── mmu_validation/             ← MMU test images + prompts
│   ├── prompts.jsonl
│   ├── prompts_with_vqa.json
│   └── *.jpg/png
│
├── lm_chat_validation/         ← LM chat validation data
│   ├── description.txt
│   └── questions.jsonl
│
├── parquet/                    ← Parquet dataset loader
│   ├── __init__.py
│   └── my_dataset.py
│
└── assets/                     ← Images for README
```

---

## 5. Dusk Training Pipeline

### Original 3-Stage Training (from paper)
```
Stage 1: Pretraining (3 sub-stages)
  1.1: ImageNet → Learn basic image generation (Dusk-8B-Instruct init)
  1.2: Image-Text pairs → Learn image-text alignment
  1.3: Text instructions → Learn instruction following
  Combined loss: L = λ_t2i * L_t2i + λ_lm * L_lm + λ_mmu * L_mmu

Stage 2: Mixed CoT SFT (2 sub-stages)
  2.1: Text-only CoT → Complex reasoning (GSM8K, MATH, etc.)
  2.2: Multimodal CoT → Multimodal reasoning + image quality

Stage 3: UniGRPO RL
  → Policy gradient with diversified rewards
  → Implemented in separate repo: github.com/QubitronLabs/dLLM-RL
```

### Training Config Key Parameters
```yaml
model:
  dusk:
    pretrained_model_path: "QubitronLabs/dusk-8b"  # or MixCoT
    new_vocab_size: 134656       # text(126464) + image(8192)
    codebook_size: 8192          # MAGVITv2 codebook
    num_vq_tokens: 1024          # 32x32 image tokens (512x512 res)
    mask_token_id: 126336        # [MASK] token for diffusion

training:
  batch_size_t2i: 5              # text-to-image samples per GPU
  batch_size_lm: 1               # language modeling samples per GPU
  batch_size_mmu: 2              # multimodal understanding samples per GPU
  gradient_accumulation_steps: 4
  mixed_precision: "bf16"
  learning_rate: 5e-5
  max_train_steps: 500000
  t2i_coeff: 1.0                 # loss weight for t2i
  lm_coeff: 0.1                  # loss weight for LM
  mmu_coeff: 1.0                 # loss weight for MMU
```

---

## 6. Available Checkpoints

| Checkpoint | HuggingFace ID | Description |
|------------|----------------|-------------|
| Dusk-8B-Base | `QubitronLabs/dusk-8b` | After pretraining + instruction tuning |
| Dusk-8B-MixCoT | `QubitronLabs/dusk-8B-MixCoT` | After mixed CoT fine-tuning |
| Dusk-8B-Max | Coming soon | After UniGRPO RL |
| Dusk-Parallel-M | `tyfeld/Dusk-Parallel-M` | Thinking-aware image editing |
| Dusk-Parallel-A | `tyfeld/Dusk-Parallel-A` | Thinking-aware image editing |

---

## 7. Hardware & Resources

### Available Compute
- **Colab Pro+**: RTX PRO 6000 Blackwell Server Edition (102GB VRAM)
- **Budget**: ~491 compute units remaining (out of 600)
- **Local**: Apple Silicon Mac (for development, MPS inference supported)

### VRAM Estimates (8B model, bf16)
```
Model weights:   ~16 GB
Optimizer (Adam): ~32 GB (fp32 states)
Activations:     ~20-40 GB (depends on batch size + sequence length)
Total:           ~70-90 GB → fits in 102GB with grad checkpointing
```

---

## 8. Key Technical Details

### Mask Diffusion Loss (Dusk/Dusk)
Given input `x`, mask ratio `t ~ U(0,1)`:
1. Create mask `m` where each token is masked with probability `t`
2. Replace masked tokens with `[MASK]` (token ID 126336)
3. Forward pass → predict original tokens at masked positions
4. Loss = Cross-entropy on masked positions only, scaled by `1/t`

$$L = -\frac{1}{t} \sum_{i \in \text{masked}} \log p(x_i | x_{\text{masked}})$$

### Semi-Autoregressive Generation
For long text generation, Dusk uses block-by-block decoding:
```
gen_length = 512, block_length = 128 → 4 blocks
Each block: steps/4 denoising steps
Within each block: full diffusion (mask → unmask)
Between blocks: autoregressive (left-to-right)
```

### Gumbel Noise for Sampling
```python
# Temperature-controlled stochastic sampling
logits = logits.to(float64)
noise = torch.rand_like(logits, dtype=float64)
gumbel_noise = (-torch.log(noise)) ** temperature
return logits.exp() / gumbel_noise
```

---

## 9. Integration Notifications

### Telegram Bot
- **Token**: `8207572563:AAHhbNGBmWh5ncF5FZgNGQz6GHVAnoj1LwU`
- **Chat ID**: `1594485798`
- Used for training progress notifications on Colab

---

## 10. Project History

### Phase 1: From-Scratch Training (Completed, Lessons Learned)
1. Built custom DUSK architecture (MLX-based, byte-level tokenizer)
2. Trained 670M model on Colab: 15K steps, 15.7h, loss 5.5→1.71
3. Generation output was garbage — Chinchilla scaling mismatch
4. Key lesson: 670M needs ~50B byte-tokens, we only had ~2B
5. Byte-level tokenizer is 5-10x less efficient than BPE
6. Decision: Use pretrained base model instead of training from scratch

### Phase 2: Dusk Integration (Current)
1. Researched all diffusion LMs (Dusk, Dream, Dusk, Mercury 2, Gemini Diffusion)
2. Chose Dusk as base — only open multimodal diffusion model
3. Downloaded codebase, studying architecture
4. Next: Fine-tune on domain-specific data, build DUSK product

---

## 11. References

### Papers
1. **Dusk**: Yang et al., "Multimodal Large Diffusion Language Models", NeurIPS 2025. arXiv:2505.15809
2. **Dusk**: Nie et al., "Large Language Diffusion Models", arXiv:2502.09992
3. **Dream**: "Diffusion Reasoning Models", Dream-org, 2025
4. **MDLM**: Sahoo et al., "Simple and Effective Masked Diffusion Language Models", NeurIPS 2024
5. **Show-o**: Xie et al., unified multimodal model with image tokenizer
6. **dLLM-RL**: QubitronLabs, RL framework for diffusion LLMs (TraceRL)

### Repos
- Dusk: `github.com/QubitronLabs/dusk` (MIT)
- dLLM-RL: `github.com/QubitronLabs/dLLM-RL` (RL infra)
- Dusk 2.X: `github.com/inclusionAI/Dusk2.X`
- MAGVITv2: `showlab/magvitv2` (HuggingFace)
