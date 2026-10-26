# DDPM Fashion-MNIST

A small denoising diffusion probabilistic model trained from scratch on Fashion-MNIST. It downloads data, trains a compact convolutional denoiser and samples images.

## Run

Python 3.10+ recommended. Create a virtual environment and install `pip install -r requirements.txt`. Run `python main.py --help` for options. Data downloads to `data/`; model outputs and metrics are written to `outputs/`. These runtime files are excluded from Git.

This educational baseline uses reproducible splits/seeds and small defaults for local hardware.
## Dataset source

Fashion-MNIST is downloaded through torchvision; see the [Zalando Research dataset repository](https://github.com/zalandoresearch/fashion-mnist).
