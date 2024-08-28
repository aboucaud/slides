class: center, middleclass: middle
<!-- background-image: url(../img/brain.png) -->

# Introduction to .red[deep learning] <br>for cosmology
#### Ecole d'été Rodolphe Clédassou 2024 – Cycle 2

.bottomlogo[<img src="../img/logo-ecole-euclid.svg" width='250px'>]
.footnote[ Alexandre Boucaud  &  Marc Huertas-Company]

[twitter]: https://twitter.com/alxbcd
---
exclude: true
## Alexandre Boucaud <img src="https://aboucaud.github.io/img/profile.png" class="circle-image" alt="AB" style="float: right">

Ingénieur de recherche at APC, CNRS

<!-- [@alxbcd][twitter] on twitter -->


.left-column[
  .medium.red[Background]

.small[**2010-2013** – PhD on WL with LSST]  
.small[**2013-2017** – IR on **Euclid** SGS pipeline]  
.small[**2017-2019** – IR on a [ML challenge platform](https://ramp.studio/) for researchers]  
.small[**since 2019** – permanent IR position]
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

[slides]: https://aboucaud.github.io/slides/2023/euclid-school-ml-cycle2

---
class: middle
# The goal for this cycle


Learn about the most useful .red[deep learning architectures] for cosmological data analysis
   - Convolutional neural networks (CNN)
   - Transformers
   - Graph neural networks (GNN)

<!-- 1. Use .blue[TensorFlow] and .blue[TensorFlow Probability] to play with these models and cosmological data -->

<!-- 3. Get started with .green[ML experiment tracking] using MLOps -->

#### 

#### 
   
---

## Program overview

<!-- ### .blue[This morning] -->

Quick recap on neural networks from Cycle 1

Probabilistic and Convolutional neural networks

<!-- -- -->

<!-- ### .green[Tuesday] -->

Transformers and Graph neural networks

TP tomorrow: Classification of galaxy types with Euclid data + Cosmology with one galaxy

---
class: center, middle

# This is an .green[interactive] lecture

### please .red[interrupt] if a notion needs to be explained ✋🏼️


---
class: center, middle

# 1. Recap on the neural networks introduction from cycle 1

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
## Increased complexity of data – e.g. Euclid sims

.center[<img src="../img/euclid_sim0.png" width="750px", vspace="0px">]

.footnote[credit: Marc Huertas-Company]

---
count: false
## Increased complexity of data – e.g. Euclid sims

.center[<img src="../img/euclid_sim1.png" width="750px", vspace="0px">]

.footnote[credit: Marc Huertas-Company]

---
count: false
## Increased complexity of data – e.g. Euclid sims

.center[<img src="../img/euclid_sim2.png" width="750px", vspace="0px">]

.footnote[credit: Marc Huertas-Company]

---
## And soon the first Euclid data release\\0/

.center[<img src="../img/euclid_NISP-commissioning-ESA.png" width="40%">]

.footnote[NISP instrument – credit: ESA press release July 31, 2023]

---
count: false
## Euclid NISP details

.center[<img src="../img/euclid_NISP-commissioning-ESA.png" width="100%">]


???

.center[
  .medium[Physical model .green[too complex] or dataset .green[too large], leading to convergence issues] 

  .big.red[speed]

  .medium[No suitable .green[physical model] available]

  .big.red[accuracy]

  .medium[There might be .green[hidden information] in the data, beyond the summary statistics we traditionally compute]
  
  .big.red[discovery]
]

---
## How can machine learning solve <br>our .green[data driven] problems ?

--

Most of the time it will use a technique called .red[**supervised learning**]

--

...which is in essence training a .green[very flexible] .small[(non linear)] function to .green[approximate the relationship] between the (raw) .green[data] and the .blue[target] task
  

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
## Multi-layer perceptron (MLP)


.center[<img src="../img/mlp.jpg" width="70%" vspace="50px" />]

.footnote[[cs231n.github.io](http://cs231n.github.io/)]

---
count: false
## Multi-layer perceptron (MLP)
.left-column[
```python
model = tfk.Sequential()

model.add(tfkl.Dense(4, input_dim=3))
model.add(tfkl.Dense(4))
model.add(tfkl.Dense(1))

# print model structure
model.summary()
```
]
.right-column[
<img src="../img/mlp.jpg" width="350px" vspace="30px" hspace="30px" />
]
.reset-column[
```
__________________________________________________
Layer (type)          Output Shape        Param #
==================================================
dense_1 (Dense)       (None, 4)           16
__________________________________________________
dense_2 (Dense)       (None, 4)           20
__________________________________________________
dense_3 (Dense)       (None, 1)           5        
==================================================
Total params: 41
Trainable params: 41
Non-trainable params: 0
```
]

---
## Dense neurons

A neuron is a .green[linear system] with two attributes
> the weight matrix $\mathbf{W}$  
> the linear bias $b$

It takes .green[multiple inputs] (from $\mathbf{x}$) and returns .green[a single output]
> $f(\mathbf{x}) = \mathbf{W} . \mathbf{x} + b $
.center[
  <img src="../img/neuron.svg" width="600px" />
]

---
## Activation functions 

.center[<img src="../img/activations.png" width="750px" vspace="0px" />]


---
## Activation layer

There are two different syntaxes whether the activation is seen as a .green[property] of the neuron layer

```python
model = tfk.Sequential()
model.add(tfkl.Dense(4, input_dim=3, activation='sigmoid'))
```

or as an .green[additional layer] to the stack

```python
model = tfk.Sequential()
model.add(tfkl.Dense(4, input_dim=3))
model.add(tfkl.Activation('tanh'))
```

The activation layer .red[does not add] any .red[depth] to the network.

---
## Loss functions

Here are the most traditional loss functions.

.blue[**Regression**] : mean square error

$$\text{MSE} = \frac{1}{N}\sum_i\left[y_i - f_w(x_i)\right]^2$$

.blue[**Classification**] : binary cross-entropy

$$\text{BCE} = -\frac{1}{N}\sum_i y_i\cdot\log\ f_w(x_i) + (1-y_i)\cdot\log\left(1-f_w(x_i)\right)$$

---
## Optimization

To optimize the weights of the networks, we use an iterative procedure, based on gradient descent, that minimizes the loss.

The most used optimizers are the `StochasticGradientDescent` (SGD) and `Adam` and its variants.

To do that we need to obtain the gradients of each of the MLP layers through the [backpropagation algorithm](https://www.jeremyjordan.me/neural-networks-training/). 

.footnote[[nice lecture](https://mlelarge.github.io/dataflowr-slides/X/lesson4.html) on optimizers]

---
## Loss and optimizer

Once your architecture (`model`) is ready, the [loss function](https://keras.io/losses/) and an [optimizer](https://keras.io/optimizers/) .red[must] be specified 
```python
model.compile(optimizer='adam', loss='mse')
```
or using specific classes for better access to optimization parameters
```python
model.compile(optimizer=tfk.optimizers.Adam(lr=0.01, decay=0.1), 
              loss='mse')
```

Choose both according to the data and target output.

---
## Optimizer learning rate

Must be chosen carefully

.center[<img src="../img/learning_rates.png" width="450px">]

---
## Monitoring the training loss

```python
history = model.fit(X_train, y_train, epochs=500, validation_split=0.3)  

import matplotlib.pyplot as plt
# Visualizing the training                    
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epochs'); plt.ylabel('loss'); plt.legend()
```

.center[<img src="../img/training-loss.png" width="450px">]

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
count: false
## But.. classic neural networks .red[**cannot**]<br>represent uncertainty

.center[
  <br>
  <br>
  <img src="../img/aleatoric_mse.png" width="85%">
]


---
## Which uncertainty are we talking about ?

**Aleatoric uncertainty** 

Uncertainty from noise in the observations or measurements and captures uncertainty that cannot be reduced .green[**even if more data was collected**]. It can be modeled by introducing a learned noise parameter per sample.


**Epistemic uncertainty** 

Uncertainty from a lack of knowledge and captures .blue[**uncertainty about the model parameters**] that can be reduced given more data. It can be modeled by introducing a prior over the model parameters and inferring a posterior distribution over them given the data.
In physics we usually call such uncertainty ***systematics***.

---
## Relation with the Mean Squared Error loss

To perform a regression on a dataset $\vec{x}$ to predict a variable $y$ using a neural network $f_w$ we do traditionnaly the training under a MSE loss which we aim at minimizing

$$\text{MSE} = \frac{1}{N}\sum_i\left(y_i - f_w(x_i)\right)^2$$

---
## Viewing neural networks as statistical models

.center[$f_w(x) = y$  ↔️️  $p_w(y|x)$]

We have a set of independant realizations $(x_i, y_i)$ $i=1,...,N$

We want to find the underlying probability distribution of $y$ given $x$: $p(y|x)$ 

We make the hypothesis that $p(y_i|x_i)$ is a Gaussian distribution with constant width $\sigma$

We can thus write the joint likelihood 

.center[$$p_\vec{w}(y|x) = L(\vec{w}, \sigma) \propto \prod_i^N \exp(-(y_i - f_w(x_i))^2/\sigma^2)$$]

---
## Likelihood maximisation

To find the parameters $\vec{w}$ that maximise the likelihood
.center[$$p_\vec{w}(y|x) = L(\vec{w}, \sigma) \propto \prod_i^N\exp(-(y_i - f_w(x_i))^2/\sigma^2)$$]

we usually work in log-space for simplification (since the $\log$ is monotonous).

Therefore, the optimization problem becomes minimising the .green[**negative log-likelihood**]

$$-\log p_\vec{w}(y|x) \propto \sum_i^N (y_i - f_w(x_i))^2/\sigma^2 + cst$$

---
## Relation with the Mean Squared Error loss

From the previous equation it appears that training a network with MSE loss on a dataset .blue[is equivalent to] maximising the joint probability of the data under the hypothesis that each data point is 
- an independant realisation
- is sampled from a Gaussian distribution of width $\sigma = 1$

The result of the optimisation is that the network will .red[model the mean value] of the distribution. 

---
## Modeling aleatoric uncertainty

.center[
  <br>
  <br>
  <br>
  <!-- <img src="../img/jax_logo.png" width="30%"> -->
  <img src="../img/pyro_logo.png" width="24%">
  <img src="../img/tfprob_logo.jpeg" width="35%">
  <br>
]

.medium.center[Probabilistic neural layers]

---

## Mixture density networks

.center[
  <img src="../img/mdn-archi.png" width="80%">
]


---

## Mixture density networks

.center[
  <img src="../img/mdn-principle.png" width="80%">
]

The neural network outputs the parameters of a distribution, on which one can backpropagate.

The network, thanks to the probabilistic layer, can now be trained under .red[**negative log-likelihood minimisation**].

---
class: center, middle

# Thank .red[you] for your attention
</br>
</br>
Find this presentation at  
https://aboucaud.github.io/slides/2024/euclid-school-ml-cycle2
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
