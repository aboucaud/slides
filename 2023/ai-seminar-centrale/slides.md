class: center, middleclass: middle
<!-- background-image: url(../img/brain.png) -->

# Introduction to .red[machine learning]
### in astrophysics
#### ST4 - Black Swans Detection in Particle Physics and Cosmology – April 12th 2023

.bottomlogo[<img src="../img/apc_logo_transp.png" width='125px'>]
.footnote[ Alexandre Boucaud  -  [@alxbcd][twitter]]

[twitter]: https://twitter.com/alxbcd
---

## Alexandre Boucaud <img src="https://aboucaud.github.io/img/profile.png" class="circle-image" alt="AB" style="float: right">

Ingénieur de recherche at APC, CNRS

<!-- [@alxbcd][twitter] on twitter -->


.left-column[
  .medium.red[Background]

.small[**2010-2013** – PhD on weak gravitational lensing]  
.small[**2013-2017** – postdoc on the **Euclid** mission]  
.small[**2017-2019** – postdoc on the development of a<br>[ML challenge platform](https://ramp.studio/) for researchers]  
.small[**since 2019** – permanent software engineer position]
]

.right-column[
  .medium.red[Interests]
  - .small[**data processing** for cosmological surveys]
  - .small[**ML applications** in astrophysics]
  - .small[**open source** scientific ecosystem]
]

.bottomlogo[
  <img src="../img/apc_logo_transp.png" height='100px'> 
  <img src="../img/vera_rubin_logo_horizontal.png" height='100px'>
  <img src="../img/euclid_logo.png" height='100px'>
]

.footnote[[aboucaud@apc.in2p3.fr][mail]]
<!-- <img src="http://www.apc.univ-paris7.fr/APC_CS/sites/default/files/logo-apc.png" height="120px" alt="Astroparticule et Cosmologie" style="float: right"> -->

[mail]: mailto:aboucaud@apc.in2p3.fr
[twitter]: https://twitter.com/alxbcd

---
exclude: true
# PDF animation replacement

.middle.center[<br><br><br><br>Animation .red[skipped] in PDF version, watch on [online slides][slides] 👈]

[slides]: https://aboucaud.github.io/slides/2022/euclid-school-ml-cycle2

---
class: middle
# Today's goal


1. Introduce the basic concepts of machine learning

2. Focus on image processing 

3. Learn about current ML applications in astrophysics  

#### 

#### 
   
---
exclude: True
## What does "deep" means ?

.center[
<img src="../img/imagenet.png" , width="700px" / >
]

.footnote[more on these common net architectures [here][archi]]

[archi]: https://www.jeremyjordan.me/convnet-architectures/

---
## Entering the data driven era

.center[
  <img src="../img/data_driven1.png" width="700px" vspace="0px"/>
]

---
count: false
## Entering the data driven era

.center[
  <img src="../img/data_driven2.png" width="700px" vspace="0px"/>
]

---
count: false
## Entering the data driven era

.center[
  <img src="../img/data_driven3.png" width="700px" vspace="0px"/>
]

---
## Why data driven ?

.center[
  .medium[Physical model .green[too complex] or dataset .green[too large], leading to convergence issues] 

  .big.red[speed]
]

---
## Observed sky area vs. time

.center[<iframe width="590" height="495" src="../img/vid/hstvseuclid.mp4" title="Euclid vs HST" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>]

.footnote[courtesy Jarle Brinchmann]

---
## Large imaging surveys in < 5y

.center[<img src="../img/big_surveys.png" , width="680px", vspace="0px">]

.footnote[credit: LSST DC2 simulation]

---
## Why data driven ?

.center[
  .medium[Physical model .green[too complex] or dataset .green[too large], leading to convergence issues] 

  .big.red[speed]

  .medium[No suitable .green[physical model] available]

  .big.red[accuracy]
]

---
## Increased complexity of data

.center[<img src="../img/complex_surveys.png" , width="680px", vspace="0px">]

---
## ... and simulations

.center[<img src="../img/complex_simulations.png" , width="680px", vspace="0px">]

---
## Example with Euclid

.center[<img src="../img/euclid_sim0.png" , width="750px", vspace="0px">]

.footnote[credit: Marc Huertas-Company]

---
count: false
## Example with Euclid

.center[<img src="../img/euclid_sim1.png" , width="750px", vspace="0px">]

.footnote[credit: Marc Huertas-Company]

---
count: false
## Example with Euclid

.center[<img src="../img/euclid_sim2.png" , width="750px", vspace="0px">]

.footnote[credit: Marc Huertas-Company]

---
## Why data driven ?

.center[
  .medium[Physical model .green[too complex] or dataset .green[too large], leading to convergence issues] 

  .big.red[speed]

  .medium[No suitable .green[physical model] available]

  .big.red[accuracy]

  .medium[There might be .green[hidden information] in the data, beyond the summary statistics we traditionally compute]
  
  .big.red[discovery]
]

---
## How can we do this ?

.center[
<img src="../img/arxiv-neural-2022.png" , width="600px" / >
]

.footnote[Word frequency on astro-ph – Huertas-Company & Lanusse 2022]

---
## Why is ML trending ?

- .medium[specialized .blue[hardware]] .right[e.g. GPU, TPU, Intel Xeon Phi]

--
- .medium[.blue[data] availability] .right[switch to data driven algorithms]

--
- .medium[ML .blue[algorithm] research] .right[e.g. self-supervised, active learning, ...]

--
- .medium[.blue[open source] tools] .right[huge ecosystem available in a few clicks]


---
## Computational power availability
GPU architectures are .blue[excellent] for the kind of computations required by the training of NN

.center[<img src="../img/tensor_core2.png" , width="600px", vspace="0px">]

| year |     hardware      | computation (TFLOPS) | price (K$) |
| ---- | :---------------: | :------------------: | :--------: |
| 2000 |  IBM ASCI White   |          12          | 100 000 K  |
| 2005 |  IBM Blue Gene/L  |         135          |  40 000 K  |
| 2021 | Nvidia Tesla A100 |        ~ 300         |   < 2 K    |

.footnote[[Wikipedia: Nvidia GPUs](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units)]

---
exclude: true
## CCIN2P3

.left-column[  
- GPU cluster dedicated to R&D
- .green[must request access] first
- [https://doc.cc.in2p3.fr]()

**GPU capacity**
- A100 and V100 available on queue  
=> [how to submit a job](https://doc.cc.in2p3.fr/fr/Computing/slurm/examples.html#job-gpu)
- K80 deployed on CC  
Jupyter notebook platform  
[https://notebook.cc.in2p3.fr]()


]

.right-column.center[
  <img src="../img/ccin2p3.png" width="350px" vspace="0px"/>  
]

.footnote[.big[powered by ]<img src="../img/in2p3-logo.png" height='80px'>]
<!-- .bottomlogo[<img src="../img/in2p3-logo.png" height='100px'>] -->

---
## Jean Zay

.left-column[
- French AI supercomputer  
- dedicated for public research but hosts external GPUs
- .red[must request hours] on it first  
=> see [this page](http://www.idris.fr/eng/info/gestion/demandes-heures-eng.html)
- a bit cumbersome to connect to it outside of a French lab (highly secure)
- [link to ressources](http://www.idris.fr/su/debutant.html)
]

.right-column.center[
  <img src="../img/jeanzay.png" width="450px" vspace="0px"/>  
]

.footnote[[Jean Zay presentation](http://www.idris.fr/media/ia/guide_nouvel_utilisateur_ia.pdf) (slides in French)]

---
class: middle, center

# Time for a poll

---
class: middle

.left-column[
  # Go to .red[slido.com]

### enter the poll code 

## \#12 49 864
]

.right-column[
  <img src="../img/qrcode_CentraleSupelec.png" width="400px" vspace="0px"/>  
]

---
## Deep learning in the last decade

.left-column[
- translation
- image captioning
- speech synthesis
- style transfer
]

.right-column[
- cryptocurrency mining
- self-driving cars
- games 
- etc.
]

.reset-column[]
.center[
  <img src="../img/dl_ex1.png" width="700px" vspace="30px"/>
]

---
## Deep learning examples

.center[
  <img src="../img/dl_ex2.png" width="800px"/>
]

---
## Speech analysis

.center[
<img src="../img/WaveNet.gif" style="width: 500px;" vspace="80px" />
]

.footnote[[WaveNet][wavenet] - TTS with sound generation - DeepMind (2017)]

[wavenet]: https://deepmind.com/blog/high-fidelity-speech-synthesis-wavenet/

---
## Image colorization

.center[
<img src="../img/imgbw.png" style="width: 166px"/>
]
.center[
<img src="../img/imgcolor.png" style="width: 500px"/>
]

.footnote[[Real-time image colorization][deepcolor] (2017)]

[deepcolor]: https://richzhang.github.io/ideepcolor/

---
## AI for strategy games

.center[
<img src="../img/starcraft-alpha_star.png" style="width: 700px"/>
]

.footnote[[AlphaStar][alphastar] - Starcraft II AI - DeepMind (2019)]

[alphastar]: https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/

---
## Data science ecosystem

.center[
 <img src="../img/rapids.png" style="width: 200px"/>
]

.center[
<img src="../img/rapids-desc.png" style="width: 650px"/>
]

.footnote[[rapids.ai][rapids] - Nvidia (2019)]

[rapids]: https://rapids.ai

--
count: false

powering e.g. chatbots, Google translate, GitHub Copilot, etc.

---
## State of the art image generation

.center[
<img src="../img/dalle2.png" style="width:90%"/>
]
<!-- .singleimg[![](../img/dalle2.png)] -->

.footnote[[OpenAI - DALL•E 2][dalle2] (2022)]

[dalle2]: https://openai.com/dall-e-2/

---
## State of the art image generation

.left-column.center[
<img src="../img/midjourney2.png" style="width:50%"/>
]
.right-column.center[
    <br>
<img src="../img/midjourney1.png" style="width:50%"/>
]
<!-- .singleimg[![](../img/dalle2.png)] -->

.footnote[[Midjourney v5][midj] (2023)]

[midj]: https://www.midjourney.com/home

---
## Advanced natural-language tasks

.center[
<img src="../img/gpt3.png" style="width: 750px"/>
]

.footnote[[OpenAI - GPT-3][openai] (2020)]

[openai]: https://openai.com/api/


---
## Advanced natural-language tasks

.center[
<img src="../img/chatgpt-centrale.png" style="width:75%"/>
]

.footnote[[OpenAI - ChatGPT][chatgpt] (2022-2023)]

[chatgpt]: https://chat.openai.com/chat

---
class: middle
# Ok, that's great ! But...


---
## How can machine learning solve our data driven problems ?

--

Most of the time it will use a technique called .red[supervised learning]

--

...which is in essence training a .green[very flexible] .small[(non linear)] function to .green[approximate] the relationship between the (raw) data and the target task
  
- classification for .blue[discrete] output
- regression for .blue[continuous] output

---
## Supervised learning

.center[
<img src="../img/classif1.png" style="width: 700px"/>
]

.footnote[credit: Marc Huertas-Company]

---
count: false
## Supervised learning

.center[
<img src="../img/classif2.png" style="width: 700px"/>
]

.footnote[credit: Marc Huertas-Company]

---
count: false
## Supervised learning

.center[
<img src="../img/classif3.png" style="width: 700px"/>
]

.footnote[credit: Marc Huertas-Company]

---
class: center, middle

.medium[There are many ML algorithms for supervised learning tasks, most of which you can find in the [**scikit-learn**][sklearn] library like random forests, boosted trees or support vector machines but in this lecture we will focus on the most flexible of all: .green[**neural networks**].]

[sklearn]: https://scikit-learn.org

---
## Zoo of neural networks
.center[
<img src="../img/nnzoo1.png" style="width: 700px"/>
]
.footnote[[Neural network zoo][nnzoo] - Fjodor van Veen (2016)]

---
## Zoo of neural networks

..center[
<img src="../img/nnzoo2.png" style="width: 700px"/>
]

.footnote[[Neural network zoo][nnzoo] - Fjodor van Veen (2016)]

[nnzoo]: http://www.asimovinstitute.org/neural-network-zoo/


---
class: middle, center
name: nn
# .red[Neural networks]

---
## A Neuron

A neuron is a .green[linear system] with two attributes
> the weight matrix $\mathbf{W}$  
> the linear bias $b$

It takes .green[multiple inputs] (from $\mathbf{x}$) and returns .green[a single output]
> $f(\mathbf{x}) = \mathbf{W} . \mathbf{x} + b $
.center[
  <img src="../img/neuron.svg" width="600px" />
]

---
## Linear layers

A linear layer is an .green[array of neurons].

A layer has .green[multiple inputs] (same $\mathbf{x}$ for each neuron) and returns .green[multiple outputs].

.center[<img src="../img/mlp.jpg" width="600px" vspace="50px" />]

.footnote[[cs231n.github.io](http://cs231n.github.io/)]

---
name: activation
## Activation functions

A network with several linear layers remains a .green[**linear system**].

To add non-linearities to the system, .red[activation functions] are introduced. 

.center[<img src="../img/artificial_neuron.svg" width="600px" />]

Activation layers .red[do not add] any .red[**depth**] to the network.

---
## Activation functions 

.center[<img src="../img/activations.png" width="750px" vspace="0px" />]


---
## A basic single neuron network


One neuron, one activation function.


.center[<img src="../img/artificial_neuron.svg" width="600px" />]

$$x \overset{neuron}\longrightarrow z(x) = wx + b\overset{activation}\longrightarrow g(z(x)) = y$$


---
## Supervised training

In .red[**supervised learning**], we train a neural network $f_{\vec w}$ with a set of weights $\vec w$ to approximate the target $\vec y$ (label, value) from the data $\vec x$ such that

$$f_{\vec w}(\vec x) = \vec y$$

For this simple network we have 

$$f_{\vec w}(x) = g(wx + b)\quad \text{with} \quad {\vec w} = \\{w, b\\}$$

In order to optimize the weight $\vec w$, we need to select a loss function $\ell$ depending on the category of problem and .red[minimize it with respect to the weights].

---
## Loss functions

Here are the most traditional loss functions.

.blue[**Regression**] : mean square error loss

$$\ell_\text{MSE} = \frac{1}{N}\sum_i\left[y_i - f_w(x_i)\right]^2$$

.blue[**Classification**] : binary cross-entropy loss (a.k.a. logistic regression loss)

$$\ell_\text{BCE} = -\frac{1}{N}\sum_i y_i\cdot\log\ f_w(x_i) + (1-y_i)\cdot\log\left(1-f_w(x_i)\right)$$

---
## Minimizing the loss

To tune the weights of the network, we use an iterative optimization procedure, based on .red[**gradient descent**], to minimize the loss function. 

--

For this to work, we need to be able to express .green[**the gradients of the loss $\ell$**] with respect to any of the weights and biases of the network.

$$ \dfrac{\partial \ell}{\partial w_i} \quad \text{and} \quad \dfrac{\partial \ell}{\partial b_i} $$

--
How do we compute these gradients ?

---
name: backprop
## Backpropagation

A .green[30-years old] algorithm (Rumelhart et al., 1986)

which is .red[key] for the re-emergence of neural networks today.

.center[<img src="../img/backpropagation.gif" width="800px" />]

.footnote[find a clear and more detailed explaination of backpropagation [here](https://www.jeremyjordan.me/neural-networks-training/)]

---
## Network update

1. feedforward and compute loss gradient on the output
$$ \nabla \ell(f_{\vec w}(\vec x)) $$

2. for each layer in the backward direction, 
  * .blue[receive] the gradients from the previous layer, 
  * .blue[compute] the gradient of the current layer
  * .blue[multiply] with the weights and .blue[pass] the results on to the next layer

3. for each layer, update their weight and bias using their own gradient, following the optimisation scheme (e.g. gradient descent)

---
## What to expect

The training must be stopped when reaching the .green[sweet spot]  
.small[(i.e. before .red[overfitting])].

.center[<img src="../img/overfitting.png" width="500px">]

---
## Typical architecture of a FCN

| parameter                |                    typical value |
| ------------------------ | -------------------------------: |
| input neurons            |            one per input feature |
| output neurons           |     one per prediction dimension |
| hidden layers            | depends on the problem (~1 to 5) |
| neurons in hidden layers |                       ~10 to 1e3 |
| loss function            |      MSE or binary cross-entropy |
| hidden activation        |                             ReLU |
| output activation        |                        see below |
    
| output activation |                  typical problem |
| ----------------- | -------------------------------: |
| `None`            |                       regression |
| `softplus`        |                 positive outputs |
| `softmax`         |        multiclass classification |
| `sigmoid/tanh`    | bounded outputs / classification |

---
## From ML to deep learning

Letting the network discover the most appropriate way to extract features from the raw data

.center[
  <img src="../img/ml_to_dl.png", width="700px", vspace="0px", hspace="0px"/>
]

---
## Questions

.big[
How do you feed an image to a FCN?]
<br/>
.center[
  <img src="../img/dc2.png", width="300px", vspace="30px", hspace="0px"/>
]

--
.big[
What issues would that create?
]

---
class: middle, center
name: cnn

# .red[Convolutional] Neural Networks

---
## Convolution in 1D

.center[<iframe width="720" height="450" src="../img/vid/1d-conv.mp4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>]

.footnote[[credit](https://www.youtube.com/watch?v=ulKbLD6BRJA)]

---
## Convolution in 2D

.center[<img src="../img/convolution_gifs/full_padding_no_strides_transposed.gif" width="400px"/>]

.footnote[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]

---
exclude:true
## Unrolling convolutions

.center[
  <img src="../img/unrolling-conv2d.png", width="500px", vspace="0px", hspace="0px"/>
]

.footnote[[credit](https://ychai.uk/notes/2019/08/28/NN/go-deeper-in-Convolutions-a-Peek/)]


---

## Convolutional Neural Networks


- succession of .blue[convolution and downsampling layers]  
  .small[(with some dense layers at the end)]
- the training phase tunes the .green[convolution kernels]
- each kernel is applied on the entire image/tensor at each step   
  => .red[preserves translational invariance]

.center[
  <img src="../img/blend_to_flux.png", height="300px", vspace="0px", hspace="0px"/>
]

---
## Convolution layer operations

.left-column[
- each input .blue[image channel] is convolved with a kernel
- the convoluted channels are summed element-wise to produce .green[output images]
- the same operation is repeated here 3 times
- the final output is a concatenation of the intermediate .green[output images]

.center[<img src="../img/convlayer2.jpg" width="65%"/>]
]

.right-column[
  .center[<img src="../img/convnet-explain-small.png" width="70%"/>]
]

---
## Strides & padding

.left-column.center[
<img src="../img/convolution_gifs/full_padding_no_strides_transposed.gif" width="350px"/>
<br>
strides 1, no padding
] 

.right-column.center[ 
<img src="../img/convolution_gifs/padding_strides.gif" />
<br>
strides 2, with padding
]

.footnote[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]

---
## Why downsampling ?

We just saw a convolution layer using a strides of 2, which is the equivalent of taking every other pixel from the convolution.  

This naturally downsamples the image by a factor of ~2*
.footnote[padding plays a role to adjust the exact shape].

There are .green[two main reasons] for downsampling the images

1. it .red[reduces the size] of the tensors .red[progressively] (i.e. until they can be passed to a dense network)
2. it allows the fixed-size kernels to explore .blue[larger scale correlations]  
  phenomenon also know as .red[increasing the *receptive field*] of the network

There are other ways of downsampling the images.

---
## Pooling layers

- common methods: **`MaxPooling`** or **`AvgPooling`**
- common strides: (2, 2)
- pooling layers do not have free parameters

.center[
  <img src="../img/maxpool.jpeg" width="600px" vspace="20px"/>
]
.footnote[[cs231n.github.io](http://cs231n.github.io/)]

---
## CNN activation layer : ReLU

</br>
.center[ 
<img src="../img/relu.jpeg" width="30%"  hspace="60px"/>
]
  </br>
- safe choice*: use .red[ReLU or [variants](https://homl.info/49)] (PReLU, [SELU](https://homl.info/selu), LeakyReLU) for the convolutional layers
- select the activation of the .red[last layer] according to your problem
.small[e.g. sigmoid for binary classification]
]
- checkout the available activation layers [here](https://keras.io/api/layers/activation_layers/)


---
class:middle, center

# Some recent .red[applications] in astrophysics and cosmology

---
## Convnets aka CNNs

Use of convolution layers to go from images / spectra to floats (summary statistics) for classification or regression

.center[
  <img src="../img/blend_to_flux.png", height="300px", vspace="0px", hspace="0px"/>
]

They had a .green[huge success] in 2017-2020

---
## CNNs are born classifiers

.left-column[
  State-of-the-art on strong lens classification benchmarks
  <br>
  <br>
  <img src="../img/cnn_lens_competition.png", width="100%", vspace="0px", hspace="0px"/>
  <br>
  <br>
  <br>
  <br>
  .right[[Metcalf+ 19](https://doi.org/10.1051/0004-6361/201832797)]
]

.right-column[
  .center[Supernovae classification with CNNs]
  <br>
  .center[<img src="../img/scone_qu_2021.png", width="80%" />]
  <br>
  .right[[Qu+ 21](https://doi.org/10.3847/1538-3881/ac0824)]
]

---

## Finding implicit information from multi-band data

.left-column[
  Photometric redshift estimation using pixel information
  <br>
  <br>
  <br>
  <img src="../img/photoz_pasquet_2018_small.png", width="100%" />
  <br>
  <br>
  .right[[Pasquet+ 19](https://doi.org/10.1051/0004-6361/201833617)]
]
.right-column[
  .center[Deblending galaxies with multiple instruments]
  <br>
  <br>
  .center[<img src="../img/arcelin_2020.png", width="100%" />]
  <br>
  .right[[Arcelin+ 2020](https://doi.org/10.1093/mnras/staa3062)]
]

<!-- 
.right-column[
  .center[Detection of strong lenses]
  <br>
  .center[<img src="../img/desilens_huang_2020.png", width="80%" />]
  <br>
  .right[[Huang+ 20](https://iopscience.iop.org/article/10.3847/1538-4357/ab7ffb/meta), [21](https://iopscience.iop.org/article/10.3847/1538-4357/abd62b/meta)]
] -->

---
## Dimensionality reduction

.left-column[
The encoder-decoder architecture is an efficient way of doing .blue[**non-linear dimensionality reduction**] through a compression (*encoding*) in a low dimensional space called the .red[**latent space**] followed by a reconstruction (*decoding*).
  
The network is trained to reconstruct the original input as output and in the process learns about the internal structure of the data. => useful for anomaly detection
]

.right-column[
  .center[
    <img src="../img/ae_simple.png", width="500px", vspace="0px", hspace="0px"/>
    <img src="../img/ae_visual.png", width="500px", vspace="0px", hspace="0px"/>
  ]
]

---
## Anomaly / outlier detection

.left-column[
  .center[Subaru HSC anomalous image visualizer]
  <br>
  .center[<img src="../img/weirdgalaxies.png", width="60%" />]
  <br>
  .right[[weirdgalaxi.es](https://weirdgalaxi.es/)]
]

.right-column[
  .center[SDSS spectra encoder used for finding outliers]
  <br>
  .center[<img src="../img/umap_melchior_2022.png", width="80%" />]
  .right[[Melchior+ 22](https://arxiv.org/abs/2211.07890), [Liang+ 23](https://arxiv.org/abs/2302.02496)]
]

---
exclude:true
## Probabilistic image segmentation <img src="../img/hubert.png" class="circle-image" alt="AB" style="float: right">

.left-column[
  <img src="../img/proba_unet.png", width="100%", vspace="0px", hspace="0px"/>
  
  <br>
  <br>
  <br>
  .right[Boucaud+19]
]

.right-column[
  .right.small[Hubert Bretonnière<br>now UC Santa Cruz]
  
  Non-binary object segmentation in the presence of blending. The main networks learns to classify the pixels as background, object, or overlapping objects, while a second network
]

.footnote[
  [Bretonnière, **Boucaud** & Huertas-Company 21](https://arxiv.org/abs/2111.15455)
]

---
class: center, middle
name: simulations

# ML-assisted .red[simulations]

---
## Variational Auto Encoder

Principle is simple: replace the deterministic latent space with a .green[**multivariate distribution**].

.center[
<img src="../img/vae_best.png" width="75%" />
]

This ensures that .red[close points] in latent space lead to .red[the same reconstruction] so it can be sampled.

.footnote[[Kingma & Welling +14](https://arxiv.org/abs/1312.6114)]

---
exclude: true
## 3D simulations

Finding dark matter halos in density fields

.center[
  <img src="../img/dmdensity.png", width="590px", vspace="0px", hspace="0px"/>
]

.footnote[Berger & Stein 18]

---
## Simulation of non-analytic galaxies profiles

.left-column[
  VAE trained on COSMOS galaxy images  

.blue[**conditioned**] on derived properties<br>
(ellipticity, magnitude, bulge ratio, etc.)

used to .blue[**simulate**] more realistic (i.e. non-analytic) galaxies.

Each single galaxy profile is sampled from a latent space distribution whose shape has been learned by a .green[normalising flow].
]

.right-column[
<img src="../img/flowvae_1.png" width="95%" vspace="0px", hspace="0px" />
<br>
<br>
<br>
<img src="../img/flowvae_2.png" width="95%" vspace="0px", hspace="0px" />
]

.footnote[
  [Lanusse+ 20](https://doi.org/10.1093/mnras/stab1214)<br>
  [Bretonnière, Huertas-Company, **Boucaud**+ 21](https://www.aanda.org/articles/aa/abs/2022/01/aa41393-21/aa41393-21.html)
]

---
## Accelerating N-Body simulations

Capturing accurately the displacement of particules in the cosmic structure

.center[
  <img src="../img/fastpm.png", width="80%", vspace="10px", hspace="0px"/>
]

.footnote[[He +18](https://doi.org/10.1073/pnas.1821458116)]

---
## N-Body emulation with GANs

Creating volumes of the universe in an autoregressive way using a GAN

.left-column[
<img src="../img/nbodygan1.png" width="100%" />
]

.right-column[
  <img src="../img/nbodygan2.png" width="100%" />
]

.footnote[[Perraudin+ 19](https://doi.org/10.1186/s40668-019-0032-1)]

---
## Painting baryons

.left-column[
  DM simulations are .green[**fast**], hydrodynamics ones are .red[**slow**].    
  Encode hydrodynamics with a NN to infer the baryonic properties of the dark matter simulations.
  <br>
  <br>
  <img src="../img/baryons.png" width="100%" />
  <br>
  <br>
  .right[[Horowitz+ 21]()]
]

.right-column[
  .center[First use of physics-informed NN to reduce scatter between baryon inpainting and actual simulations with stellar-to-halo mass relation
  <br>
  <br>
  <img src="../img/phy_informed_baryon_inpainting-dai-2023.png" width="50%" />]
  <br>
  <br>
  .right[[Dai+ 23](https://arxiv.org/abs/2303.14090)]
]

---
class: center, middle

# Score-based models

### gradients of the data likelihood

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
<img src="../img/ddpm.png" width="55%" />
]
.center[
<img src="../img/ddpm_img.png" width="65%" />
]

.footnote[Smith+21 + [GitHub](https://github.com/Smith42/astroddpm)]

---
# Take home message

<br>
.medium[Machine Learning is a .red[**powerful tool**] for physicists and gives state-of-the-art results for .green[detection] and .green[classification] tasks].
<br>
<br>
.medium[It can be used to .red[**explore and extract**] information from .green[high-dimensional datasets] (unsupervised learning) as well as .red[**approximate**] highly .green[non-linear models] (supervised learning).]
<br>
<br>
.medium[As any data driven method, it is very .red[**sensitive to biases**] in the dataset (e.g. selection bias) and one must take good care of .green[validating] the results.]

---
class: center, middle

# Thank .red[you] for your attention
</br>
</br>
Find this presentation at  
https://aboucaud.github.io/slides/2023/ai-seminar-centrale
</br>
</br>
</br>
</br>
.small[
  This presentation is licensed under a   
  [Creative Commons Attribution-ShareAlike 4.0 International License][cc]
]

[![](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)][cc]

[cc]: http://creativecommons.org/licenses/by-sa/4.0
