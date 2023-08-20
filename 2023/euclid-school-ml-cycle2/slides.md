class: center, middleclass: middle
<!-- background-image: url(../img/brain.png) -->

# Introduction to .red[deep learning] <br>for cosmology
#### Ecole d'été Rodolphe Clédassou 2023 – Cycle 2

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


1. Learn about the most useful .red[deep learning models] for cosmology   
   CNNs, Transformers, GNNs and probabilistic layers

2. Use .blue[TensorFlow] and .blue[TensorFlow Probability] to play with these models and cosmological data

3. Get started with .green[ML experiment tracking] using MLOps

#### 

#### 
   
---

## Program overview

### .blue[This morning]

Quick recap on neural networks from Cycle 1

Probabilistic and Convolutional neural networks

--

### .green[Tuesday]

Transformers and Graph neural networks

ML experiment tracking with MLflow

TP: Cosmology with one galaxy

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
## And soon Euclid data \\0/

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
class: center, middle

# Thank .red[you] for your attention
</br>
</br>
Find this presentation at  
https://aboucaud.github.io/slides/2023/euclid-school-ml-cycle2
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
