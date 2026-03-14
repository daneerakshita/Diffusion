# Variational Autoencoder

A Variational Autoencoder built from scratch in PyTorch, trained on the MNIST handwritten digits dataset.

Built by following [Aladdin Persson's](https://www.youtube.com/@AladdinPersson) VAE tutorial, with the goal of understanding the core theory behind generative models including the ELBO loss, KL divergence, and the reparameterization trick.

---

## What is a VAE?

A Variational Autoencoder is a generative model that learns to compress data into a structured latent space and reconstruct it. Unlike a regular autoencoder, a VAE encodes inputs as a **distribution** (mean μ and variance σ²) rather than a single point — allowing it to generate new, unseen samples by sampling from that distribution.

```
Input Image (784)
      ↓
   Encoder
      ↓
  μ  and  σ        ← latent distribution
      ↓
  z = μ + σ * ε    ← reparameterization trick
      ↓
   Decoder
      ↓
Reconstructed Image (784)
```

---

## Architecture

| Component | Details |
|-----------|---------|
| Input dim | 784 (28×28 flattened) |
| Hidden dim | 200 |
| Latent dim (z) | 20 |
| Encoder | Linear → ReLU → Linear (μ), Linear (σ) |
| Decoder | Linear → ReLU → Linear → Sigmoid |
| Loss | BCE Reconstruction Loss + KL Divergence |
| Optimizer | Adam (lr = 3e-4) |

---

## Loss Function

The VAE is trained to minimise the **ELBO loss** which is a combination of two terms:

- **Reconstruction loss** — how accurately the decoder rebuilds the original image
- **KL divergence** — keeps the latent space organised and close to a standard normal distribution N(0, I)

```
Loss = BCE(x_reconstructed, x) + KL Divergence
```

---

## Project Structure

```
vae-mnist/
├── model.py        # VAE architecture (Encoder, Decoder, VAE)
├── train.py        # Training loop + inference
├── requirements.txt
└── README.md
```

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/daneerakshita/vae-mnist.git
cd vae-mnist
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run training**
```bash
python3 train.py
```

The MNIST dataset will be downloaded automatically on first run.

---

## Dependencies

- Python 3.9+
- PyTorch
- torchvision
- tqdm
- matplotlib

---

## Key Concepts Learned

- Encoder-decoder architecture
- Latent space representations
- Reparameterization trick
- ELBO loss (reconstruction + KL divergence)
- PyTorch model building with `nn.Module`

---

## Credits

Tutorial by [Aladdin Persson](https://www.youtube.com/@AladdinPersson) — highly recommended for anyone learning deep learning with PyTorch.
