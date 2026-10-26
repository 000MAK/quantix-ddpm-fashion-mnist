# Denoising Diffusion Model | Fashion-MNIST

A compact denoising diffusion probabilistic model (DDPM) trained from scratch on Fashion-MNIST. It learns to predict added noise and generates a grid of sample images.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --epochs 1 --limit 1000
```

The dataset downloads into `data/`; generated samples, training loss history, and weights are saved under `outputs/`. Both paths are excluded from Git.

## Model and limits

This is a small educational DDPM baseline. The reported noise-prediction loss is a training diagnostic, not a sample-quality benchmark. The default model is intentionally compact; more training and a stronger architecture are needed for meaningful image-quality comparisons.

Dataset source: [Fashion-MNIST, Zalando Research](https://github.com/zalandoresearch/fashion-mnist).
