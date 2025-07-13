---
layout: home
title: "IT ALL STARTS WITH NOISE"
---

# IT ALL STARTS WITH NOISE !

Avada Kedavra!


![Architecture](/assets/a.png)

*Don't worry, this isn't the Dark Arts class. This high-level architecture diagram of a Stable Diffusion model may look terrifying, but it's going to make sense in just a few minutes.*

In this blog, we will understand a diffusion model from scratch. Here we will discuss the various elements that make a stable diffusion model.

---

## Stable Diffusion

Stable Diffusion belongs to a class of deep learning models called **diffusion models**. These are generative models, which means they’re designed to generate new data similar to what they saw during training. In the case of Stable Diffusion, this data is images.

### Forward Diffusion

The forward process is all about adding noise to an image, step-by-step, until the image becomes like random noise.

Imagine we trained the model only on images of cats and dogs. Initially, these images form two distinct clusters. As we apply forward diffusion, we add noise over many steps, and eventually, all images (cat or dog) end up looking like pure noise.

### Reverse Diffusion

What if we could revert the noise process? Starting from noise, what if we could recover the original image?

That’s exactly what diffusion models do: they learn how to reverse the diffusion process. Starting with random noise, they denoise step-by-step until a realistic image emerges.

### Training the Noise Predictor and Reverse Process

So how do we train a model to do this reverse process?

We teach a neural network to **predict the noise** that was added during the forward process. In Stable Diffusion, this noise predictor is a **U-Net** model.

Steps:

1. Take a clean image.  
2. Generate some random noise.  
3. Add that noise to the image (forward diffusion).  
4. Train the U-Net to predict the noise that was added.

During generation:

1. Start with pure noise.  
2. Use the trained U-Net to predict the noise.  
3. Subtract the predicted noise.  
4. Repeat the process many times.

Initially, this process is **unconditional**. With **conditioning** (e.g., text prompts), we can guide the output.

---

## Latent Diffusion Model: Making It Fast

One of the biggest challenges with earlier diffusion models was speed. They operated directly on high-resolution image pixels — and that’s expensive.

Stable Diffusion solves this by using a **latent diffusion model** — it doesn't operate in pixel space but in a smaller **latent space** (about 48× smaller), making everything faster and more efficient.

### Enter: Variational Autoencoder (VAE)

Compression to latent space is done using a **Variational Autoencoder (VAE)**.

A VAE has:

1. **Encoder** – Compresses image into latent space  
2. **Decoder** – Reconstructs the image

In Stable Diffusion, all forward and reverse diffusion happens in this **latent space**, not pixel space.

> The latent space in Stable Diffusion is typically `4 × 64 × 64`, much smaller than full image size `3 × 512 × 512`.

So now we add noise to a latent tensor, recover the clean latent, and decode it to get the final image.

---

## Forward Diffusion Process

We gradually add Gaussian noise to the data over multiple steps. Eventually, the image becomes almost pure noise.

![Forward Diffusion](/assets/forward_diffusion.png)

Each forward step is:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t \mathbf{I})
$$

Where:

- \( q \): forward process  
- \( x_t \): noisy data at step \( t \)  
- \( $$\beta_t  $$\): noise variance  
- \( $$ \mathcal{N} $$ \): Gaussian distribution  

We simplify:

$$
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

So,
$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$
or,

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

---

## Reverse Diffusion

This is the reverse process — removing noise step by step.

![Reverse Diffusion](/assets/reverse_diffusion.png)

Each forward step is:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t \mathbf{I})
$$

We simplify:

$$
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

So,

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

or,

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$


> Math reference: [Lilian Weng’s post](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process)

---

## Architecture

Now that we've covered the math, here's what the architecture looks like:

- **U-Net** for noise prediction  
- **VAE** for latent space encoding/decoding  
- Optional **text encoder** for conditional generation  

---

