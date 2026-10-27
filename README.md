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

## Results

A one-epoch smoke run ended with training noise-prediction MSE **≈ 1.0**, the value of predicting zero noise: the model had not learned yet. Treat it as a pipeline check, not a result; meaningful samples need many more epochs.

## Repository contents

| File | Purpose |
|---|---|
| `.gitignore` | Keeps data, weights and outputs out of Git |
| `LICENSE` | MIT licence for the code |
| `main.py` | Training and evaluation script |
| `requirements.txt` | Python dependencies |

## License

Code is released under the MIT License (see `LICENSE`). Datasets keep their own licences and are downloaded at run time, not redistributed here.

## Related

Part of a set of compact, reproducible ML/DL baselines by [000MAK](https://github.com/000MAK). Flagship research project: [QMASE martial-arts analytics engine](https://github.com/000MAK/quantix-martial-arts-engine).
