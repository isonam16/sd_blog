**Avada Kedavra!**

![Architecture Diagram](content/arch.png)
<!-- ![Architecture Diagram](content/arch.png "High-level architecture of Stable Diffusion") -->

*Don't worry, this isn't the Dark Arts class. This high-level architecture diagram of a Stable Diffusion model may look terrifying, but it's going to make sense in just a few minutes.*

---

In this blog, we will understand a diffusion model from scratch. Here we will discuss the various elements that make a stable diffusion model.

## Diffusion Models

Stable Diffusion belongs to a class of deep learning models called **diffusion models**. These are generative models, which means they’re designed to generate new data similar to what they saw during training. In the case of Stable Diffusion, this data is nothing but images.

### Forward Diffusion

The forward process is all about adding noise to an image, step-by-step, until the image becomes almost indistinguishable from random noise.

> Imagine we trained the model only on images of cats and dogs. Initially, these images form two distinct clusters. As we apply forward diffusion, we add noise over many steps, and eventually, all images (cat or dog) end up looking like pure noise and we can't tell if it was originally a cat or a dog.

### Reverse Diffusion

Here's the fun part: **reverse diffusion**. What if we could play the noise process backwards? Starting from noise, what if we could recover the image — a cat or a dog?

That’s exactly what diffusion models do: they learn how to reverse the diffusion process. Starting with random noise, they denoise step-by-step until a realistic image emerges.

Technically, diffusion is modeled with two parts:
- A drift term (which gently guides the image toward the data distribution)
- A random motion term (the noise).

The reverse process leans toward one of the learned modes (cat or dog) — but not something in between.

### Training the Noise Predictor

So how do we train a model to do this magical reverse process?

The key is to teach a neural network to **predict the noise** that was added during the forward process. In Stable Diffusion, this noise predictor is a **U-Net** model. Here's how the training works:

1. Take a clean image (say, a cat).
2. Generate some random noise.
3. Add that noise to the image for a certain number of steps (simulate forward diffusion).
4. Train the U-Net to predict the noise that was added.

After training, the model gets good at predicting the noise at any point in the diffusion process.

### Sampling: Reverse in Action

During image generation, we:

1. Start with pure noise.
2. Use the trained U-Net to predict the noise.
3. Subtract the predicted noise.
4. Repeat this process many times.

Slowly, the noise transforms into a meaningful image — either a cat or a dog, depending on the learned distribution. For now, this process is **unconditional** — we can't control what the output is (a cat or dog). But when we introduce **conditioning** (like using text prompts), we’ll gain control.

That’s the core idea behind diffusion models — turn order into noise (forward), then learn to reverse it (backward), and guide that reverse process to generate completely new data.

---

## What is Stable Diffusion?

Stable Diffusion is a type of machine learning model used for generating images from text descriptions. It belongs to a class of models called diffusion models, which are generative models designed to gradually transform random noise into meaningful data (e.g., images) through a series of denoising steps.

Diffusion is divided into two parts: Forward Diffusion and Reverse Diffusion. Let's dive into each of these processes.

---

## Forward Diffusion Process

In the forward diffusion step, we gradually add noise to the image or data over multiple steps until it becomes close to pure Gaussian noise. This process is repeated over a fixed number of timesteps, with a small amount of Gaussian noise added at each step. As the number of steps increases, the image becomes more noisy and approaches pure Gaussian noise.

This sequence can be visualized as follows:

![Forward Diffusion](content/forward diffusion.png)
*Figure: Forward diffusion process – gradual noising of an image over multiple timesteps.*

Each step of the forward diffusion process is defined as:

q(x_t | x_{t-1}) = N(x_t; √(1 - β_t) * x_{t-1}, β_t * I)


Where:
- `q` is the forward process distribution.
- `x_t` is the noisy data at timestep `t`.
- `x_{t-1}` is the data from the previous timestep.
- `N` denotes a normal (Gaussian) distribution.
- The mean is `√(1 - β_t) * x_{t-1}`
- The variance is `β_t * I`, where `I` is the identity matrix.

The equation shows we are adding more noise to our data as we progress to the next step by changing the mean and variance, making it converge to pure Gaussian Noise.

`β_t` is a **schedule** that decides how much noise is added in each step. OpenAI proposed their own — the **cosine schedule**.

Now, we represent `x_t` using `x_0` instead of `x_{t-1}` recursively:

q_t(q_{t-1}(q_{t-2}(... q_1(x_0))))


Only the mean depends on the previous step, and `β_t * I` is already known for each `t`.

Define:

α_t = 1 - β_t
α̅_t = ∏_{s=1}^t α_s


Then,

q(x_1 | x_0) = N(x_1; √α_1 * x_0, (1 - α_1) * I)


As the process continues, the mean evolves:

μ_t = √(α_t) * x_{t-1} = √(α_t * α_{t-1}) * x_{t-2} = ... = √(α̅_t) * x_0


So finally,

q(x_t | x_0) = N(x_t; √(α̅_t) * x_0, (1 - α̅_t) * I)


Or using reparameterization:

x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε, where ε ~ N(0, I)


---

## Reverse Diffusion

As the name suggests, reverse diffusion removes the noise and creates a new image. But instead of directly removing the noise, the model **predicts the noise** that needs to be removed and subtracts it from the noisy image to get a cleaner version. This step is repeated many times until a high-quality image emerges.

![Reverse Diffusion](content/reverse diffusion.png)
*Figure: Reverse diffusion process – gradually removing noise over multiple timesteps.*

---

Reference for reverse math: [Lilian Weng - Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process)
