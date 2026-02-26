# Dusk - Task Tracker

> **Created:** 2026-02-26
> **Last Updated:** 2026-02-26
> **Goal:** Transform Dusk into Dusk - a production-ready multimodal diffusion LM

---

## Phase 2: Dusk Base Integration

### 2.1 Environment Setup [DONE]
- [x] Download Dusk source code from GitHub
- [x] Copy to DLLM workspace
- [x] Clean old from-scratch code
- [x] Update project-context.md

### 2.2 Model Download & Local Inference [IN PROGRESS]
> **Priority:** HIGH

- [ ] Download Dusk-8B-MixCoT weights (~16GB bf16, auto-downloads via from_pretrained)
- [ ] Install backend dependencies in .venv
- [x] Build Dusk web interface (Next.js + FastAPI)
- [ ] Start API server and test text generation
- [ ] Verify diffusion text effect in browser

### 2.3 Codebase Deep Understanding
- [ ] Trace text generation flow end-to-end
- [ ] Trace image generation flow
- [ ] Trace multimodal understanding flow
- [ ] Trace training flow

---

## Phase 3: Fine-Tuning Plan

### 3.1 Choose Fine-Tuning Vertical
- [ ] Code generation / Math reasoning / Hindi / Creative writing / Domain-specific

### 3.2 Prepare Fine-Tuning Data
- [ ] Source + format + validate training data

### 3.3 Set Up Colab Training Notebook
- [ ] Create notebooks/dusk_finetune.ipynb

### 3.4 Run Fine-Tuning
- [ ] Validate pipeline, full training, save best checkpoint

---

## Phase 4: Inference Optimization
- [ ] Benchmark tok/s
- [ ] Reduce denoising steps, larger block_length
- [ ] KV caching, torch.compile, quantization

---

## Phase 5: Product Development

### 5.1 Web Interface [DONE]
- [x] FastAPI backend with SSE streaming
- [x] Next.js frontend with diffusion text effect
- [x] Dark theme (zinc/violet) + settings panel
- [x] Switched to Bun

### 5.2 Deployment & Evaluation
- [ ] Deploy API, add auth + rate limiting
- [ ] Run MMLU, GSM8K, MATH, HumanEval benchmarks
- [ ] Branding, blog post, HuggingFace model card

---

## Quick Start

Backend: cd server && python main.py
Frontend: cd web && bun dev
