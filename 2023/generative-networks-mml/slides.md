class: middle
background-image: url(../img/brain.png)

# Introduction to<br>.red[generative networks]

.footnote[ Alexandre Boucaud  -  [@alxbcd][twitter]]

[twitter]: https://twitter.com/alxbcd
---
## Full convolutional networks FCN

They provide a way to recover the original shape of the input tensors through .green[transposed convolutions]

.center[
  <img src="../img/encode_decode.png", width="600px", vspace="0px", hspace="0px"/>
]

---
## Transpose convolutions

.center[
  <img src="../img/unrolling-transpose-conv2d.png", width="700px", vspace="0px", hspace="0px"/>
]

.footnote[[credit](https://www.mdpi.com/1424-8220/19/19/4251)]

---
## Auto encoders

Efficient way of doing non-linear dimensionality reduction through a compression (encoding) in a low dimensional space called the .red[latent space] followed by a reconstruction (decoding).

.center[
  <img src="../img/ae_simple.png", width="600px", vspace="0px", hspace="0px"/>
]

---
## Auto encoders

.center[
  <br>
  <br>
  <img src="../img/ae_visual.png", width="700px", vspace="0px", hspace="0px"/>
]

.big.center[👉 cf. demo in [Colab notebook][vaenotebook]]

[vaenotebook]: https://colab.research.google.com/drive/1kQOba1mZk8hv6djd2LeaPY70Kdv0QpvA?usp=sharing

---
## Feature extraction in images

Here segmentation of overlapping galaxies

.center[
  <img src="../img/U_net.png", width="600px", vspace="0px", hspace="0px"/>
]

.footnote[Boucaud+19]

---
## Probabilistic segmentation

Same but with captured "uncertainty"

.center[
  <img src="../img/proba_unet.png", width="550px", vspace="0px", hspace="0px"/>
]

.footnote[Bretonnière+21]

---
## Accelerating N-Body sims

Capturing the displacement of particules

.center[
  <img src="../img/fastpm.png", width="700px", vspace="10px", hspace="0px"/>
]

.footnote[He+18]

---
## 3D simulations

Finding dark matter halos in density fields

.center[
  <img src="../img/dmdensity.png", width="590px", vspace="0px", hspace="0px"/>
]

.footnote[Berger&Stein+18]

---
class: center, middle

# .red[Variational] Auto Encoders

---
## VAE

Principle is simple: replace the deterministic latent space with a multivariate distribution.

.center[
<img src="../img/vae_best.png" width="90%" />
]

This ensures that .red[close points] in latent space lead to .red[the same reconstruction].

.footnote[[Kingma & Welling 2014](https://arxiv.org/abs/1312.6114)]

---
## Imposing structure

Adding to the loss term a Kullback-Leibler (KL) divergence term regularizes the structure of the latent space.

.center[<iframe width="540" height="335" src="../img/vid/dkl_2.mp4" title="Justine KL explanation" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>]
---
## Reparametrisation trick

Making sure once can .green[backpropagate] through the network even when one of the nodes is .blue[non-deterministic].

.center[
<img src="../img/reparm_trick.png" width="100%" />
]

.footnote[credit: Marc Lelage]

---
## FlowVAE for realistic galaxies

Added in the Euclid simulation pipeline for generating realistic galaxies .red[from input properties] (ellipticity, magnitude, B/T, ...).

.center[
<img src="../img/flowvae.png" width="85%" />
]

---
## Painting baryons

.center[
<img src="../img/baryons.png" width="100%" />
]

.footnote[Horowitz+21]

---
class: center, middle

# Generative adversarial networks

---
## GANs

The idea behing GANs is to train two networks jointly
* a discriminator $\mathbf{D}$ to classify samples as "real" or "fake"
* a generator $\mathbf{G}$ to map a fixed distribution to samples that fool $\mathbf{D}$

.center[
<img src="../img/gan.png" width="80%" />
]

.footnote[
[Goodfellow et al. 2014](https://arxiv.org/abs/1406.2661)
]

---
## GAN training

The discriminator $\mathbf{D}$ .red[is a classifier] and $\mathbf{D}(x)$ is interpreted as the probability for $x$ to be a real sample.

The generator $\mathbf{G}$ takes as input a Gaussian random variable $z$ and produces a fake sample $\mathbf{G}(z)$.

The discriminator and the generator .red[are learned alternatively], i.e. when parameters of $\mathbf{D}$ are learned $\mathbf{G}$ is fixed and vice versa.

--

When $\mathbf{G}$ is fixed, the learning of $\mathbf{D}$ is the standard learning process of a binary classifier (sigmoid layer + BCE loss).

--

The learning of $\mathbf{G}$ is more subtle. The performance of $\mathbf{G}$ is evaluated thanks to the discriminator $\mathbf{D}$, i.e. the generator .red[maximizes] the loss of the discriminator.

---
## N-Body emulation

.center[
<img src="../img/nbodygan1.png" width="100%" />
]

.footnote[Perraudin+19]

---
count: false
## N-Body emulation

.center[
<img src="../img/nbodygan2.png" width="85%" />
]

.footnote[Perraudin+19]

---
## Generating is easy but...

estimating the density from data samples is hard

.center[
<img src="../img/vaevsdensity.png" width="90%" />
]

---
class: center, middle
name: density

# .red[Denoising diffusion models]

---
## Denoising diffusion models

with stochastic differential equations (SDE)

Learn the distribution through a shochastic noise diffusion process

.center[
<img src="../img/perturb_vp.gif" width="85%" />
]

.footnote[credit: Yang Song – read his detailed [blog post](https://yang-song.net/blog/2021/score/)]

---
count: false
## Denoising diffusion models

with stochastic differential equations (SDE)

New samples are generated by reversing the SDE flow.

.center[
<img src="../img/denoise_vp.gif" width="85%" />
]

The process avoids the .red[mode collapse] inherent to GANs.

.footnote[credit: Yang Song – read his detailed [blog post](https://yang-song.net/blog/2021/score/)]

---
## Realistic galaxy simulations

.center[
<img src="../img/ddpm.png" width="75%" />
]
.center[
<img src="../img/ddpm_img.png" width="85%" />
]

.footnote[Smith+21 + [GitHub](https://github.com/Smith42/astroddpm)]

---
class: center, middle
name: backup

# .red[Backup] slides

---
## Dropout

A % of random neurons are .green[switched off] during training  
it mimics different architectures being trained at each step 

.center[<img src="../img/dropout.png" width="500 px" />]
.footnote[[Srivastava et al. (2014)](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)]

---
## Dropout

```python
...
dropout_rate = 0.1

model = tfk.Sequential()
model.add(tfkl.Conv2D(2, (3, 3), activation='relu', input_shape=(9, 9, 1)))
*model.add(tfkl.Dropout(dropout_rate))
model.add(tfkl.Conv2D(4, (3, 3), activation='relu'))
*model.add(tfkl.Dropout(dropout_rate))
...
```

- efficient regularization technique 
- .green[prevents overfitting]

**Note:** dropout is .red[not used during evaluation], which accounts for a small gap between **`loss`** and **`val_loss`** during training.


.footnote[[Srivastava et al. (2014)](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)]

---
## Batch normalization

```python
...
model = tfk.Sequential()
model.add(tfkl.Conv2D(..., activation=None))
model.add(tfkl.Activation('relu'))
*model.add(tfkl.BatchNormalization())
```

- technique that .green[adds robustness against bad initialization]
- forces activations layers to take on a unit gaussian distribution at the beginning of the training
- ongoing debate on whether this must be used before or after activation  
  => current practices (2022) seem to favor .red[after activation] 

.footnote[[Ioffe & Szegedy (2015)](http://arxiv.org/abs/1502.03167)]

---
## Skip connections

.left-column[
- The skip connection, also know as a .red[residual block], .green[bypasses the convolution block] and concatenates the input with the output.  
- It allows the gradients to be better propagated back to the first layers and solves the .blue[*vanishing gradients*] problem.
]

.right-column[

A skip connection looks like this
  .center[<img src="../img/resnet-block.png" width="350px" />]

They are at the heart of [ResNet](https://arxiv.org/abs/1512.03385) and [UNet](https://arxiv.org/abs/1505.04597).
]

---
## Skip connections

.left-column[
```python
d = {"activation": "relu", 
     "padding": "same"}

def resblock(inputs):
  x = tfkl.Conv2D(64, 3, **d)(inputs)
  x = tfkl.Conv2D(64, 3, **d)(x)
  return tfkl.add([inputs, x])

inputs = tfk.Input(shape=(32, 32, 3))
x = tfkl.Conv2D(32, 3, **d)(inputs)
x = tfkl.Conv2D(64, 3, **d)(x)
x = tfkl.MaxPooling2D()(x)
*x = resblock(x)
*x = resblock(x)
x = tfkl.Conv2D(64, 3, **d)(x)
x = tfkl.GlobalAveragePooling2D()(x)
x = tfkl.Dense(256, "relu")(x)
x = tfkl.Dropout(0.5)(x)
outputs = tfkl.Dense(10)(x)

model = tfk.Model(inputs, outputs)
```
]

.right-column[

One needs to use .green[the functional API] of Keras/TensorFlow in order to write residual blocks since they are no longer sequential.  

👈 on the left is a short model with two residual blocks in it.  

⚠️ The convolution layers in the residual block .red[must preserve the tensor shape] in order to be concatenated.
]
