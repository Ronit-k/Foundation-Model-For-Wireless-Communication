# 🤖 MAT-Pseudo-Heavy Autoresearch Swarm Setup

Welcome to the autonomous research laboratory. 

Based on Andrej Karpathy's original `autoresearch` repository, this framework has been completely retooled to allow frontier language models (like Claude or GPT-4o) to **autonomously experiment with, stabilize, and tune your heavy Mask-Aware Vision Transformer (MAT)**.

## 📂 Architecture Overview
The user's code inherently separated execution across `pretrain.py` and `mat_vit_lwm.py`. The core tenet of `autoresearch` is that the agent only touches **one file**, and operates in iterative loops based on `git` changes.

This folder consolidates your heavy architecture into a tracking-ready environment:
- **`autoresearch/`**: The entire cloned auto-lab git repository. Everything happens inside this nested folder.
  - **`prepare.py`**: I abstracted your `load_channels_ri` dataset fetching into this read-only utility. The AI Agent is **forbidden** from modifying this. This acts as the data-ground-truth. It requires access to your `.npy` channel cache. 
  - **`train.py`**: The *only* file the AI touches. It is a massive 200+ line consolidated script that I built by copy-pasting the exact mathematical structures of `WindowMHA`, `ATB`, `MATStage`, and `MATPseudoLWM` natively alongside the Training/Warmup Loop. Hardcoded to exactly **10 Epochs** (~4-5 mins run time), the agent modifies this script, compiles loss, prints the final `val_mse`, and commits or discards changes depending on whether it beat the standing baseline MSE!
  - **`program.md`**: The system prompt injected to your AI agent. I modified this to guide it out of text generation space and fully into the MIM Vision Transformer logic. It explicitly limits the agent to MSE scoring and forbids the usage of `LayerNorm` to prevent spatial leakage.

## 🚀 How to Run the Swarm

### 1. Requirements
Ensure you are in the `lwm_cuda` conda environment. You need a CLI LLM interface that handles files and executes commands (like `Claude Code`, `Cursor`, `Aider`, or using `gemini` here!).

The `prepare.py` script attempts to load `../../channels_cache.npy`. As long as you have already executed a DeepMIMO dataset export previously to that cache, everything is functional.

### 2. Kickstarting
Open your terminal.
```bash
# 1. Enter the autonomous workspace
cd mat_autoresearch_heavy/autoresearch

# 2. Command your chosen LLM Agent to begin its endless search
<activate agent CLI e.g: aider or claude-code>

# 3. Use this exact prompt to the agent:
"Hello! Please read program.md carefully. Let's kick off a new experiment and tune the MAT network. Set up the git branches and initialize results.tsv. Get straight to work."
```

### 3. What to Expect
The agent will execute its first run of `train.py`, taking roughly 5 minutes. It will establish the `baseline` validation MSE using the exact initializations and architecture we designed today.
Then, it will forever loop in the background:
1. Come up with a structural thesis (e.g. "Maybe scaling window sizes in stage 2 drops MSE?")
2. Hardcode the change into `train.py`.
3. Train for 10 epochs.
4. Record keeping. If it fails, `git reset`. If it succeeds, `git commit` and push the envelope.

Leave it running overnight, and you'll awake to a tuned super-model.
