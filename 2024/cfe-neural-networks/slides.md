class: middle
background-image: url(../img/brain.png)

# Introduction to .red[Neural Networks]
###     with examples in <img src="../img/tf2.png" height="45px" style="vertical-align: middle" />

.bottomlogo[<img src="../img/CFE.png" width='250px'>]
.footnote[ Alexandre Boucaud]

---
class: middle
## Program overview

Understanding neural networks

Probabilistic networks

Convolutional Networks

---
## Zoo of neural networks #1

.center[
<img src="../img/nnzoo1.png" style="width: 700px"/>
]

.footnote[[Neural network zoo][nnzoo] - Fjodor van Veen (2016)]

---
## Zoo of neural networks #2

.center[
<img src="../img/nnzoo2.png" style="width: 900px"/>
]

.footnote[[Neural network zoo][nnzoo] - Fjodor van Veen (2016)]

[nnzoo]: http://www.asimovinstitute.org/neural-network-zoo/

---
<!-- class:middle -->
## Foreword: Python imports 🐍

The code snippets in these slides will use [Keras](https://keras.io) and [Tensorflow](https://tensorflow.org) (TF).  

.footnote[Keras is embedded inside TF since version 2.x.]

These snippets require some .red[preliminary Python imports] and .red[aliases] to be run beforehand.

```python
import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers
```
---
class: middle, center
name: nn
# What is a .red[neural network] made of ?

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

.center[
  <!-- <img src="../img/linear_layer.jpeg" width="450px" /> -->
  <img src="../img/mlp_bkg.svg" width="450px" vspace="40px"/>
]

---
## Hidden layers

All layers internal to the network (not input or output layer) are considered .green[hidden layers].

.center[<img src="../img/mlp.jpg" width="600px" vspace="50px" />]

.footnote[[cs231n.github.io](http://cs231n.github.io/)]

---
## Multi-layer perceptron (MLP)


.left-column[
```python
# initialize model
model = tfk.Sequential()

# add layers
model.add(tfkl.Dense(4, input_dim=3))
model.add(tfkl.Dense(4))
model.add(tfkl.Dense(1))
```
]

.right-column[
<img src="../img/mlp.jpg" width="350px" vspace="30px" hspace="30px" />
]

---
count: false
## Multi-layer perceptron (MLP)
.left-column[
```python
# initialize model
model = tfk.Sequential()

# add layers
model.add(tfkl.Dense(4, input_dim=3))
model.add(tfkl.Dense(4))
model.add(tfkl.Dense(1))
```
]
.right-column[
<img src="../img/mlp.jpg" width="350px" vspace="30px" hspace="30px" />
]

.hidden[aa]
.reset-column[]
.center[
.huge[QUESTION:]</br></br>
.big[How many .red[free parameters] has this model ?]
]

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
dense_1 (Dense)       (None, 4)           16         <=   W (3, 4)   b (4, 1)
__________________________________________________
```
]

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
dense_2 (Dense)       (None, 4)           20         <=   W (4, 4)   b (4, 1)
__________________________________________________
```
]

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
dense_3 (Dense)       (None, 1)           5          <=   W (4, 1)   b (1, 1)
==================================================
Total params: 41
Trainable params: 41
Non-trainable params: 0
```
]

---
exclude: True

## Multi-layer perceptron (MLP)

```python
# initialize model
model = tfk.Sequential()

# add layers
model.add(tfkl.Dense(4, input_dim=3))
model.add(tfkl.Dense(4))
model.add(tfkl.Dense(1))
```

--
exclude: True
```python
# print model structure
model.summary()
```

--
exclude: True
```bash
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 20
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 41
Trainable params: 41
Non-trainable params: 0
```

---
name: activation
## Adding non linearities

A network with several linear layers remains a .green[linear system].

--

To add non-linearities to the system, .red[activation functions] are introduced. 

.center[<img src="../img/artificial_neuron.svg" width="600px" />]

---
count: false
## Adding non linearities

A network with several linear layers remains a .green[linear system].

To add non-linearities to the system, .red[activation functions] are introduced. 

.center[<img src="../img/feedforwardnn.gif" width="400px" />]

.footnote[credit: Alexander Chekunkov]

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

--

or as an .green[additional layer] to the stack

```python
model = tfk.Sequential()
model.add(tfkl.Dense(4, input_dim=3))
model.add(tfkl.Activation('tanh'))
```

--
The activation layer .red[does not add] any .red[depth] to the network.

---
## Simple network


One neuron, one activation function.


.center[<img src="../img/artificial_neuron.svg" width="600px" />]

$$x \overset{neuron}\longrightarrow z(x) = wx + b\overset{activation}\longrightarrow g(z(x)) = y$$


---
## Supervised training

In .red[supervised learning], we train a neural network $f_{\vec w}$ with a set of weights $\vec w$ to approximate the target $\vec y$ (label, value) from the data $\vec x$ such that

$$f_{\vec w}(\vec x) = \vec y$$

For this simple network we have 

$$f_{\vec w}(x) = g(wx + b)\quad \text{with} \quad {\vec w} = \\{w, b\\}$$

In order to optimize the weight $\vec w$, we need to select a loss function $\ell$ depending on the category of problem and .red[minimize it with respect to the weights].

---
## Loss functions

Here are the most traditional loss functions.

.blue[**Regression**] : mean square error

$$\text{MSE} = \frac{1}{N}\sum_i\left(y_i - f_w(x_i)\right)^2$$

.blue[**Classification**] : binary cross-entropy

$$\text{BCE} = -\frac{1}{N}\sum_i y_i\cdot\log\left(f_w(x_i)\right) + (1-y_i)\cdot\log\left(1-f_w(x_i)\right)$$

---
## Minimizing the loss

To optimize the weights of the network, we use an iterative procedure, based on gradient descent, that minimizes the loss. 

--

For this to work, we need to be able to express the gradients of the loss $\ell$ with respect to any of the weight of the network.

In our single neuron example, we need
$$ \dfrac{\partial \ell}{\partial w} \quad \text{and} \quad \dfrac{\partial \ell}{\partial b} $$

--
Can we compute these gradients easily ?

---
name: backprop
## Backpropagation

A .green[30-years old] algorithm (Rumelhart et al., 1986)

which is .red[key] for the re-emergence of neural networks today.

.center[<img src="../img/backpropagation.gif" width="800px" />]

.footnote[credit: Alexander Chekunkov]

---
## Chain rule

Backpropagation works if networks are .green[differentiable].

.red[Each layer] must have an analytic derivative expression.

$$ x \overset{neuron}\longrightarrow z(x) = wx + b\overset{activation}\longrightarrow g(z(x)) = y$$

--
Since $w$ and $b$ are also variables
$$ z(x, w, b) = wx + b\,, $$
the gradients can be expressed as
$$ \dfrac{\partial z}{\partial x} = w\,, \quad \dfrac{\partial z}{\partial w} = x \quad\text{and}\quad \dfrac{\partial z}{\partial b} = 1\,. $$
---
count:false
## Chain rule

Backpropagation works if networks are .green[differentiable].

.red[Each layer] must have an analytic derivative expression.

$$ x \overset{neuron}\longrightarrow z(x) = wx + b\overset{activation}\longrightarrow g(z(x)) = y$$

Then the .red[chain rule] can be applied :

$$ \dfrac{\partial \ell}{\partial w} =
   \dfrac{\partial \ell}{\partial g} \cdot 
   \dfrac{\partial g}{\partial z} \cdot 
   \dfrac{\partial z}{\partial w} = \nabla \ell(y) \cdot g'(z) \cdot x $$
and
$$ \dfrac{\partial \ell}{\partial b} =
   \dfrac{\partial \ell}{\partial g} \cdot 
   \dfrac{\partial g}{\partial z} \cdot 
   \dfrac{\partial z}{\partial b} = \nabla \ell(y) \cdot g'(z)$$

---
## Network with more layers

Let's add one layer (with a single neuron) with the .green[same] activation $g$
<!-- $$ x \longrightarrow z_1 = w_1x + b_1\longrightarrow a_1 = g(z_1(x)) \downarrow$$ -->
<!-- $$  y = a_2 = g(z_2(x)) \longleftarrow z_2 = w_2a_1 + b_2$$ -->
$$ x \longrightarrow z_1 = w_1x + b_1\longrightarrow a_1 = g(z_1(x)) \rightarrow$$
$$ \rightarrow z_2 = w_2a_1 + b_2 \longrightarrow a_2 = g(z_2(x)) = y $$
--
.center.red[How do we compute the gradients of $w_1$ : $\dfrac{\partial\ell}{\partial w_1}$ ?]
--
.footnote[Hint: remember the algorithm is called .green[backpropagation]]

---
## Network with more layers

Let's add one layer (with a single neuron) with the .green[same] activation $g$
<!-- $$ x \longrightarrow z_1 = w_1x + b_1\longrightarrow a_1 = g(z_1(x)) \rightarrow$$
$$ \rightarrow z_2 = w_2a_1 + b_2 \longrightarrow a_2 = g(z_2(x)) = y $$ -->
$$ x \longrightarrow z_1 = w_1x + b_1\longrightarrow a_1 = g(z_1(x)) \rightarrow$$
$$ \rightarrow z_2 = w_2a_1 + b_2 \longrightarrow a_2 = g(z_2(x)) = y $$
We use the .red[chain rule]

$$ \dfrac{\partial \ell}{\partial w_1} =
   \dfrac{\partial \ell}{\partial a_2} \cdot 
   \dfrac{\partial a_2}{\partial z_2} \cdot 
   \dfrac{\partial z_2}{\partial a_1} \cdot 
   \dfrac{\partial a_1}{\partial z_1} \cdot 
   \dfrac{\partial z_1}{\partial w_1} $$
--
which simplifies to

$$ \dfrac{\partial \ell}{\partial w_1} =
   \nabla \ell(y) \cdot 
   g'(z_2) \cdot 
   w_2 \cdot 
   g'(z_1) \cdot 
   x $$
   <!-- = \dfrac{\partial \ell}{\partial a} \cdot g'(z) \cdot x $$ -->

---
## Network with more layers

From the latest expression

$$ \dfrac{\partial \ell}{\partial w_1} =
   \left(
     \nabla \ell(y) \cdot 
     g'(z_2) \cdot 
     w_2 \right)\cdot 
   g'(z_1) \cdot 
   x $$

one can derive the algorithm to compute .green[the gradient for a layer] $z_i$
$$ \dfrac{\partial \ell}{\partial z_i} =
\left(
\left(\nabla \ell(y) \cdot 
  g'(z_n) \cdot 
  w_n \right) \cdot 
  g'(z^{n-1}) * w^{n-1}\right)
  [\dots]  \cdot g'(z_i) $$

which can be re-written as .red[a recursion]

$$ \dfrac{\partial \ell}{\partial z_i} = \dfrac{\partial \ell}{\partial z^{i+1}} \cdot w^{i+1} \cdot g'(z_i) $$

--
.footnote[find a clear and more detailed explaination of backpropagation [here](https://www.jeremyjordan.me/neural-networks-training/)]

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

.footnote[[Nice lecture](https://mlelarge.github.io/dataflowr-slides/X/lesson4.html) on optimizers]

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
## Training

It's time to .green[train] your model on the data (`X_train`, `y_train`). 

```python
model.fit(X_train, y_train,
          batch_size=32,        
          epochs=50,  
          validation_split=0.3) # % of data being used for val_loss evaluation

```

- **`batch_size`**: .green[\# of images] used before updating the model<br/>
  32 is a very good compromise between precision and speed*
- **`epochs`**: .green[\# of times] the model is trained with the full dataset

After each epoch, the model will compute the loss on the validation set to produce the **`val_loss`**. 

.red[The closer the values of **`loss`** and **`val_loss`**, the better the training]. 

.footnote[*see [Masters et al. (2018)](https://arxiv.org/abs/1804.07612)]

---
## Training

It's time to .green[train] your model on the data (`X_train`, `y_train`). 

```python
model.fit(X_train, y_train,
          batch_size=32,        
          epochs=50,  
          validation_split=0.3) # % of data being used for val_loss evaluation

```

- **`batch_size`**: .green[\# of images] used before updating the model<br/>
  32 is a very good compromise between precision and speed*
- **`epochs`**: .green[\# of times] the model is trained with the full dataset

After each epoch, the model will compute the loss on the validation set to produce the **`val_loss`**. 

.red[The closer the values of **`loss`** and **`val_loss`**, the better the training]. 

.footnote[*see [Masters et al. (2018)](https://arxiv.org/abs/1804.07612)]

---
## Monitoring the training loss

```python
import matplotlib.pyplot as plt

history = model.fit(X_train, y_train, epochs=500, validation_split=0.3)  

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
## Learning rate

Must be chosen carefully

.center[<img src="../img/learning_rates.png" width="450px">]

---
## Callbacks

[Callbacks](https://keras.io/api/callbacks/) are methods that act on the model during training, e.g.

```python
# Save the weights of the model based on lowest val_loss value
chkpt = tfk.callbacks.ModelCheckpoint('weights.h5', save_best_only=True)
# Stop the model before 50 epochs if stalling for 5 epochs
early = tfk.callbacks.EarlyStopping(patience=5)

model.fit(X_train, y_train,
          epochs=50,
          callbacks=[chkpt, early])
```
--
- `ModelCheckpoint` saves the weights locally
  ```python
  model.load_weights('weights.h5')  # instead of model.fit()
  ```
- Many other interesting callbacks such as   
  `LearningRateScheduler`, `TensorBoard` or `TerminateOnNaN`

---
exclude: true
## Tensorboard

Use this callback to monitor your training live and compare the training of your models.

.center[<img src="../img/tensorboard_edward.png" width="600px">]

.footnote[credit: [Edward Tensorboard tuto](http://edwardlib.org/tutorials/tensorboard)]

---
exclude: true
## Multi-layer perceptron

The .green[classical] neural network, with an input vector $X_i$ where $i$ is the sample number.

.center[<img src="../img/mlp.jpg" width="600px" vspace="50px" />]

.footnote[[cs231n.github.io](http://cs231n.github.io/)]

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
class:center, middle

# Let's recap..

---

## Here is some random data from an experiment

.center[
  <br>
  <br>
  <img src="../img/aleatoric_data.png" width="85%">
  <br>
]

We can model it with a dense neural network..

---
## Regression with NN and MSE loss

.center[
  <br>
  <br>
  <img src="../img/aleatoric_data.png" width="85%">
]

.center.medium[Can you guess what it will look like ?]

---
count: false
## Regression with NN and MSE loss

.center[
  <br>
  <br>
  <img src="../img/aleatoric_mse.png" width="85%">
]

--

.center.medium[Are you .red[**satisfied**] with the result ?]

---
class: center, middle

# Classic neural networks .red[**cannot**]<br>represent uncertainty 😵️

---
## Which uncertainty are we talking about ?

**Aleatoric uncertainty** 

Uncertainty from noise in the observations or measurements and captures uncertainty that cannot be reduced .green[**even if more data was collected**]. It can be modeled by introducing a learned noise parameter per sample.


**Epistemic uncertainty** 

Uncertainty from a lack of knowledge and captures .blue[**uncertainty about the model parameters**] that can be reduced given more data. It can be modeled by introducing a prior over the model parameters and inferring a posterior distribution over them given the data.
In physics we usually call such uncertainty ***systematics***.

---

## Back on the Mean Squared Error loss

To perform a regression on a dataset $\vec{x}$ to predict a variable $y$ using a neural network $f_w$ we do traditionnaly the training under a MSE loss which we aim at minimizing

$$\text{MSE} = \frac{1}{N}\sum_i\left(y_i - f_w(x_i)\right)^2$$

.center.medium[What does it come from ?]

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
## Relation with the MSE

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
## Non-Gaussian scenario

.center[
  <img src="../img/mdn-multi.png" width="80%">
]


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

.center[<iframe width="560" height="315" src="../img/vid/1d-conv.mp4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>]

.footnote[[credit](https://www.youtube.com/watch?v=ulKbLD6BRJA)]

---
## Convolution in 2D

.center[<img src="../img/convolution_gifs/full_padding_no_strides_transposed.gif" width="400px"/>]

.footnote[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]

---
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
## Convolution layers

.left-column[
```python
model = tfk.Sequential()
  # First conv needs input_shape
model.add(
  tfkl.Conv2D(
    15,              # filter size 
    (3, 3),          # kernel size
    strides=1,       # default
    padding='valid', # default
    input_shape=(32, 32, 3)))
  # Next layers don't
model.add(
  tfkl.Conv2D(16, (3, 3), strides=2))
model.add(tfkl.Conv2D(32, (3, 3)))
```
]

.right-column[
<img src="../img/convlayer2.jpg" width="300px" vspace="40px", hspace="50px"/>
] 

.reset-columns[
  <br> <br> <br> <br/> <br/> <br/> <br> <br> <br> <br> 
- **kernel properties**: .green[size] and number of .green[filters]
- **convolution properties**: .green[strides] and .green[padding]
- output shape depends on .red[**all**] these properties
]

---
## Convolution layer operations

.left-column[
- each input .blue[image channel] is convolved with a kernel
- the convoluted channels are summed element-wise to produce .green[output images]
- the same operation is repeated here 3 times
- the final output is a concatenation of the intermediate .green[output images]
]

.right-column[
  .singleimg[![](../img/convnet-explain-small.png)]
]

---
## No strides, no padding

.left-column[
```python
model = tfk.Sequential()
model.add(
  tfkl.Conv2D(
    filters=1, 
    kernel_size=(3, 3), 
    strides=1,               # default
    padding='valid',         # default
    input_shape=(7, 7, 1)))
```
```python
model.summary()
```

```
_________________________________________
Layer (type)            Output Shape     
=========================================
conv2d (Conv2D)         (None, 5, 5, 1)  
=========================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________
```
] 
.right-column[
<img src="../img/convolution_gifs/full_padding_no_strides_transposed.gif" width="350px"/>
] 


.footnote[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]
---
## Strides (2,2) + padding

.left-column[
```python
model = tfk.Sequential()
model.add(
  tfkl.Conv2D(
    1, 
    (3, 3), 
*   strides=2, 
    padding='same', 
    input_shape=(5, 5, 1)))
```
```python
model.summary()
```

```
_________________________________________
Layer (type)            Output Shape     
=========================================
conv2d (Conv2D)         (None, 3, 3, 1)  
=========================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
_________________________________________
```
]
.right-column[ 
<img src="../img/convolution_gifs/padding_strides.gif" />
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
## Pooling layers

.left-column[
```python
model = tfk.Sequential()
model.add(
  tfkl.Conv2D(
    1, (3, 3), 
    strides=1, 
    padding='same', 
    input_shape=(8, 8, 1)))
model.add(tfkl.MaxPool2D((2, 2)))
```

```python
model.summary()
```

```
__________________________________________________
Layer (type)          Output Shape        Param #
==================================================
conv2d_1 (Conv2D)     (None, 8, 8, 1)     10
__________________________________________________
max_pooling2d_1 (MaxP (None, 4, 4, 1)     0
==================================================
Total params: 10
Trainable params: 10
Non-trainable params: 0
__________________________________________________
```
]
.right-column[ 
  <img src="../img/maxpool.jpeg" width="350px" vspace="50px" hspace="30px" />
]

---
## Activation

.left-column[
```python
model = tfk.Sequential()
model.add(
  tfkl.Conv2D(
    1, (3, 3), 
*   activation='relu'
    input_shape=(5, 5, 1)))
```
]

.right-column[ 
<img src="../img/relu.jpeg" width="250px"  hspace="60px"/>
]

.reset-columns[
  </br>
  </br>
  </br>
  </br>
  </br>
  </br>
- safe choice*: use .red[ReLU or [variants](https://homl.info/49)] (PReLU, [SELU](https://homl.info/selu), LeakyReLU) for the convolutional layers
- select the activation of the .red[last layer] according to your problem
.small[e.g. sigmoid for binary classification]
]
- checkout the available activation layers [here](https://keras.io/api/layers/activation_layers/)

<!-- .footnote[*not been proven (yet) but adopted empirically] -->
---
class: center, middle
# Convnets

---
## Convnets aka CNNs

Use of convolution layers to go from images / spectra to floats (summary statistics) for classification or regression

.center[
  <img src="../img/blend_to_flux.png", height="300px", vspace="0px", hspace="0px"/>
]

They had a .green[huge success] in 2017-2020

---
## Classification of strong lenses

.center[
  <img src="../img/cnn_res1.png", width="700px", vspace="0px", hspace="0px"/>
]

---
count: false
## Classification of strong lenses

.center[
  <img src="../img/cnn_res2.png", width="750px", vspace="0px", hspace="0px"/>
]

---
class: center, middle
# Image2Image networks

### a.k.a. feature extractors

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
  <img src="../img/ae_visual.png", width="700px", vspace="0px", hspace="0px"/>
]

.big.center[👉 cf. demo in Jupyter notebook]

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
class: center, middle

# Thank .red[you] for your attention
</br>
Find this presentation at  
https://aboucaud.github.io/slides/2024/cfe-neural-networks  
</br>
</br>
.small[
  This presentation is licensed under a   
  [Creative Commons Attribution-ShareAlike 4.0 International License][cc]
]

[![](https://i.creativecommons.org/l/by-sa/4.0/88x31.png)][cc]

[cc]: http://creativecommons.org/licenses/by-sa/4.0
