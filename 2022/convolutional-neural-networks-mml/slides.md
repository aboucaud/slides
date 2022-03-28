class: middle
background-image: url(img/brain.png)

# Hands on .red[deep learning]
#### MML - Cours 09

.footnote[ Alexandre Boucaud  -  [@alxbcd][twitter]]

[twitter]: https://twitter.com/alxbcd
---
exclude: true
class: middle
background-image: url(img/brain.png)
.hidden[aa]
# Hands on .red[deep learning]
.small[with [Keras][keras] examples]
.footnote[Alexandre Boucaud  -  [@alxbcd][twitter]]

---
exclude: true
<!-- class: center, middle -->
## deep learning vs. physics arXiv
<img src="img/arxiv-2019-04.png" , width="800px" / >

---
## What does "deep" means ?

.center[
<img src="img/imagenet.png" , width="700px" / >
]

.footnote[more on these common net architectures [here][archi]]

[archi]: https://www.jeremyjordan.me/convnet-architectures/

---
## Why this recent trend ?

- .medium[specialized .blue[hardware]] .right[e.g. GPU, TPU, Intel Xeon Phi]

--
- .medium[.blue[data] availability] .right[_big data era_]

--
- .medium[.blue[algorithm] research] .right[e.g. adversarial or reinforcement learning]

--
- .medium[.blue[open source] tools] .right[huge ecosystem right now]

---
## Graphics Processing Unit (GPU)

- **< 2000** : "graphics cards" (video edition + game rendering)

.center[<img src="img/msi.jpg" , width="450px">]

---
count: false
## Graphics Processing Unit (GPU)

- **< 2000** : "graphics cards" (video edition + game rendering)
- .hidden[< ]**1999** : Nvidia coins the term "GPU" for the first time

.center[<img src="img/cuda.jpg" , width="400px">]

---
count: false
## Graphics Processing Unit (GPU)

- **< 2000** : "graphics cards" (video edition + game rendering)
- .hidden[< ]**1999** : Nvidia coins the term "GPU" for the first time
- .hidden[< ]**2005** : programs start to be faster on GPU than on CPU

---
count: false
## Graphics Processing Unit (GPU)

- **< 2000** : "graphics cards" (video edition + game rendering)
- .hidden[< ]**1999** : Nvidia coins the term "GPU" for the first time
- .hidden[< ]**2005** : programs start to be faster on GPU than on CPU
- .hidden[< ]**2016** : GPUs are part of our lives (phones, computers, cars, etc..)

.center[<img src="img/nvidia_v100.jpg" , width="300px", vspace="50px"/ >]
.footnote[credit: Nvidia Tesla V100]

---
## Computational power 
GPU architectures are .blue[excellent] for the kind of computations required by the training of NN

.center[<img src="img/tensor_core2.png" , width="600px", vspace="20px">]

| year | hardware | computation (TFLOPS) | price (K$) |
|------|:------:|:-----------:|:----:|
| 2000 | IBM ASCI White | 12 | 100 000 K |
| 2005 | IBM Blue Gene/L | 135 | 40 000 K |
| 2018 | Nvidia Tesla V100 | > 100 | 10 K |
<!-- | 2017 | Nvidia Titan V | > 100 | 3 K | -->

<!-- .footnote[[Wikipedia: Nvidia GPUs](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units)] -->

---
## Deep learning software ecosystem

.center[
  <img src="img/frameworks.png" width="800px" vspace="30px"/>
]

---
## Deep learning today

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
  <img src="img/dl_ex1.png" width="700px" vspace="30px"/>
]

---
## Deep learning today

.center[
  <img src="img/dl_ex2.png" width="800px"/>
]

---
## DL today: speech synthesis

.center[
<img src="img/WaveNet.gif" style="width: 500px;" vspace="80px" />
]

.footnote[[WaveNet][wavenet] - TTS with sound generation - DeepMind (2017)]

[wavenet]: https://deepmind.com/blog/high-fidelity-speech-synthesis-wavenet/

---
## DL today: image colorisation

.center[
<img src="img/imgbw.png" style="width: 166px"/>
]
.center[
<img src="img/imgcolor.png" style="width: 500px"/>
]

.footnote[[Real-time image colorization][deepcolor] (2017)]

[deepcolor]: https://richzhang.github.io/ideepcolor/

---
## DL today: strategic game bots

.center[
<img src="img/starcraft-alpha_star.png" style="width: 700px"/>
]

.footnote[[AlphaStar][alphastar] - Starcraft II AI - DeepMind (2019)]

[alphastar]: https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/

---
## DL today: data science world

.center[
 <img src="img/rapids.png" style="width: 200px"/>
]

.center[
<img src="img/rapids-desc.png" style="width: 650px"/>
]

.footnote[[rapids.ai][rapids] - Nvidia (2019)]

[rapids]: https://rapids.ai

---
exclude: true
class: center, middle

### The sucess of ML applications is blatant,

#### BUT

### we are still .red[far]* from "Artificial Intelligence".

.footnote[*see [nice post][mjordanmedium] by M. Jordan - Apr 2018]

[mjordanmedium]: https://medium.com/@mijordan3/artificial-intelligence-the-revolution-hasnt-happened-yet-5e1d5812e1e7 

---

# Outline

.medium[.green[Neural nets]] -  .small[(dernier cours)]

> hidden layers - activation - backpropagation - optimization

--

.medium[[Convolutional Neural Networks (CNN)](#cnn)]

> kernels - strides - pooling - loss - training


--
.medium[[In practice](#practice)]

> step-by-step - monitoring your training


--
.medium[[Common optimizations](#optim)]

> data augmentation - dropout - batch normalisation

---
## ⚠️  Important side note ⚠️

As a consequence of a massive code change in TensorFlow between versions 1.x and 2.x, with versions of **TensorFlow** above 2.x, .red[all Python imports] from the code examples in this lecture .red[need to be modified] as follows:  .red[`keras`] become .red[`tensorflow.keras`]

**Example**
```python
from keras.module import Toto
```
needs to be changed to
```python
from tensorflow.keras.module import Toto
```
otherwise the examples might not work.

---
## Multi-layer perceptron

The .green[classical] neural network, with an input vector $X_i$ where $i$ is the sample number.

.center[<img src="img/mlp.jpg" width="600px" vspace="50px" />]

.footnote[[cs231n.github.io](http://cs231n.github.io/)]

---
## Typical architecture

* .blue[input neurons]: one per input feature
* .blue[output neurons]: one per prediction dimension
* .blue[hidden layers]: depends on the problem (~ 1 to 5)
* .blue[neurons in hidden layers]: depends on the problem (~10 to 100)
* .blue[loss function]: Mean Squared Error (MSE)
* .blue[hidden activation]: ReLU
* .blue[output activation]: 
    - None
    - softplus (positive outputs)
    - softmax (multiclass classification)
    - sigmoid/tanh (bounded outputs)

---
class: middle, center

## QUESTION:

### How would you feed .red[images] to a MLP ?

---
class: middle, center
name: cnn

# .red[Convolutional] Neural Networks

---
exclude: true
## Zoo of neural networks #1
.singleimg[![](img/nnzoo1.png)]

.footnote[[Neural network zoo][nnzoo] - Fjodor van Veen (2016)]

---
exclude: true
## Zoo of neural networks #2

.singleimg[![](img/nnzoo2.png)]

.footnote[[Neural network zoo][nnzoo] - Fjodor van Veen (2016)]

[nnzoo]: http://www.asimovinstitute.org/neural-network-zoo/

---
class: middle
### Classical convolutional net (CNN)

.center[
  <img src="img/blend_to_flux.png", width="800px", vspace="20px", hspace="0px"/>
]

---
class: middle
### Fully convolutional net (FCNN)

.center[
  <img src="img/U_net.png", width="700px", vspace="10px", hspace="0px"/>
]

---

## Convolutional Neural Networks

- elegant way of passing .green[tensors] to a network
- perform convolutions with .green[3D kernels] 
- training optimizes kernels, not neuron weights

.center[
  <img src="img/cnn_overview2.png", width="800px", vspace="30px", hspace="0px"/>
]

---
## Convolutional layers

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
# First conv needs input_shape
# Shape order depends on backend
model.add(
    Conv2D(15,       # filter size 
           (3, 3),   # kernel size
           strides=1,       # default
           padding='valid', # default
           input_shape=(32, 32, 3)))
# Next layers don't
model.add(Conv2D(16, (3, 3) strides=2))
model.add(Conv2D(32, (3, 3)))
```
]

.right-column[
<img src="img/convlayer2.jpg" width="300px" vspace="50px", hspace="50px"/>
] 

.reset-columns[
  <br/> <br/> <br/> <br/> <br/> <br/> <br/> <br/> <br/>  <br/> 
- **kernel properties**: .green[size] and number of .green[filters]
- **convolution properties**: .green[strides] and .green[padding]
- output shape depends on .red[**all**] these properties
]


---
## No strides, no padding

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(
    Conv2D(1, (3, 3), 
           strides=1,        # default
           padding='valid',  # default
           input_shape=(7, 7, 1)))
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
<img src="img/convolution_gifs/full_padding_no_strides_transposed.gif" width="350px"/>
] 


.footnote[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]
---
## Strides (2,2) + padding

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(1, (3, 3), 
*                strides=2, 
                 padding='same', 
                 input_shape=(5, 5, 1)))
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
<img src="img/convolution_gifs/padding_strides.gif" />
]

.footnote[[arXiv:1603.07285](https://arxiv.org/abs/1603.07285)]

---
## Pooling layers

- reduces the spatial size of the representation (downsampling)<br/>
=> less parameters & less computation
- common method: **`MaxPooling`** or **`AvgPooling`**
- common strides: (2, 2)

.center[
  <img src="img/maxpool.jpeg" width="600px" vspace="20px"/>
]
.footnote[[cs231n.github.io](http://cs231n.github.io/)]

---
## Pooling layers

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D

model = Sequential()
model.add(Conv2D(1, (3, 3), 
                 strides=1, 
                 padding='same', 
                 input_shape=(8, 8, 1)))
model.add(MaxPool2D((2, 2)))
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
  <img src="img/maxpool.jpeg" width="350px" vspace="50px" hspace="30px" />
]

---
## Activation

.left-column[
```python
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(1, (3, 3), 
*                activation='relu'
                 input_shape=(5, 5, 1)))
```
]

.right-column[ 
<img src="img/relu.jpeg" width="250px"  hspace="60px"/>
]

.reset-columns[
  </br>
  </br>
  </br>
  </br>
  </br>
  </br>
- safe choice*: use .red[ReLU or [variants](https://homl.info/49)] (PReLU, [SELU](https://homl.info/selu)) for the convolutional layers
- select the activation of the .red[last layer] according to your problem
.small[e.g. sigmoid for binary classification]
]

<!-- .footnote[*not been proven (yet) but adopted empirically] -->
.footnote[*not been proven (yet) but adopted empirically]

---
class: center, middle

# EXERCICE
.medium[on your own time, .red[write down the model] for the following architecture ]

<img src="img/cnn_overview2.png", width="700px", vspace="30px", hspace="0px"/>

.medium[how many .red[free parameters] does this architecture have ?]

---
exclude: true
# SOLUTION

.left-column[

```python
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

model = Sequential()

model.add(Conv2D(10, (5, 5), 
          input_shape=(32, 32, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(25, (5, 5)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(100, (4, 4)))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(10))

model.summary()
```]

.right-column[
```
 ___________________________________________
  Layer        Output Shape          Param #
 ===========================================
  Conv2D       (None, 28, 28, 10)    260
 ___________________________________________
  MaxPool2D    (None, 14, 14, 10)    0
 ___________________________________________
  Conv2D       (None, 10, 10, 25)    6275
 ___________________________________________
  MaxPool2D    (None, 5, 5, 25)      0
 ___________________________________________
  Conv2D       (None, 2, 2, 100)     40100
 ___________________________________________
  MaxPool2D    (None, 1, 1, 100)     0
 ___________________________________________
  Flatten      (None, 100)           0
 ___________________________________________
  Dense        (None, 10)            1010
 ===========================================
  Total params: 47,645
  Trainable params: 47,645
  Non-trainable params: 0
 ___________________________________________
```
]

---
exclude: true
# SOLUTION (bis)

.left-column[

```python
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

input = Input((32, 32, 1))
x = Conv2D(10, (5, 5))(input)
x = MaxPool2D((2, 2))(x)
x = Conv2D(25, (5, 5))(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(100, (4, 4))(x)
x = MaxPool2D((2, 2))(x)
x = Flatten()(x)
output = Dense(10)(x)

model = Model(input, output)
model.summary()
```]

.right-column[
```
 ___________________________________________
  Layer        Output Shape          Param #
 ===========================================
  Conv2D       (None, 28, 28, 10)    260
 ___________________________________________
  MaxPool2D    (None, 14, 14, 10)    0
 ___________________________________________
  Conv2D       (None, 10, 10, 25)    6275
 ___________________________________________
  MaxPool2D    (None, 5, 5, 25)      0
 ___________________________________________
  Conv2D       (None, 2, 2, 100)     40100
 ___________________________________________
  MaxPool2D    (None, 1, 1, 100)     0
 ___________________________________________
  Flatten      (None, 100)           0
 ___________________________________________
  Dense        (None, 10)            1010
 ===========================================
  Total params: 47,645
  Trainable params: 47,645
  Non-trainable params: 0
 ___________________________________________
```
]

---
## Loss and optimizer

Once your architecture (`model`) is ready, a [loss function](https://keras.io/losses/) and an [optimizer](https://keras.io/optimizers/) .red[must] be specified 
```python
model.compile(optimizer='adam', loss='mse')
```
or with better access to optimization parameters
```python
from keras.optimizers import Adam
from keras.losses import MSE

model.compile(optimizer=Adam(lr=0.01, decay=0.1), 
              loss=MSE)
```

Choose both according to the target output.

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
## Callbacks

[Callbacks](https://keras.io/callbacks/) are methods that act on the model during training, e.g.

```python
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# Save the weights of the model based on lowest val_loss value
chkpt = ModelCheckpoint('weights.h5', save_best_only=True)
# Stop the model before 50 epochs if stalling for 5 epochs
early = EarlyStopping(patience=5)

model.fit(X_train, y_train,
          epochs=50,
          callbacks=[chkpt, early])
```
--
- ModelCheckpoint saves the weights, which can be reloaded
  ```python
  model.load_weights('weights.h5')  # instead of model.fit()
  ```
- EarlyStopping saves the planet.


---
exclude: true
class: center, middle
name: architecture

# Common .red[architectures]

---
class: center, middle
name: practice

# In .red[practice]

---
## The right architecture
<!-- class: middle -->

There is currently .red[no magic recipe] to find a network architecture 
that will solve your particular problem.

.center[
  # `¯\_(ツ)_/¯`
]

But here are some advice to guide you in the right direction and/or get you out of trouble.

---
## A community effort

.center.medium[The Machine Learning community has long been </br>a fervent advocate of </br></br>.big.red[open source]</br></br>which has fostered the spread of very recent developements even from big companies like Google. </br></br> Both .green[code and papers] are generally available </br>and can be found .green[within a few clicks].]

---
## Start with existing (working) models

- look for a relevant architecture for your problem  
  (arXiv, blogs, websites)

.center[
<img src="img/ssd.png" width="600px" />
]

---
count: false
## Start with existing (working) models

- look for a relevant architecture for your problem  
  (arXiv, blogs, websites)
- find an implementation on [GitHub][gh]  
  (often the case if algorithm is efficient)

.center[
<img src="img/ssd_keras.png" width="700px" /> 
]

---
count: false
## Start with existing (working) models

- look for a relevant architecture for your problem  
  (arXiv, blogs, websites)
- find an implementation on [GitHub][gh]  
  (often the case if algorithm is efficient)
- play with the examples and adjust to your inputs/outputs

--
- use [pretrained nets][kerasapp] for the  pre-processing of your data

--
- start tuning the model parameters..

[gh]: https://github.com/
[kerasapp]: https://keras.io/applications/

---
## Plot the training loss

```python
import matplotlib.pyplot as plt

history = model.fit(X_train, y_train, validation_split=0.3)  

# Visualizing the training                    
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epochs'); plt.ylabel('loss'); plt.legend()
```

.center[<img src="img/loss.png" width="500px">]

---

## Plot the training loss

And look for the training .green[sweet spot] (before .red[overfitting]).

.center[<img src="img/overfitting.png" width="550px">]

---
## Plot other metrics

```python
import matplotlib.pyplot as plt

model.compile(..., metrics=['acc'])  # computes other metrics, here accuracy

history = model.fit(X_train, y_train, validation_split=0.3)

# Visualizing the training                    
plt.plot(history.history['acc'], label='training')
plt.plot(history.history['val_acc'], label='validation')
plt.xlabel('epochs'); plt.ylabel('accuracy'); plt.legend()
```

.center[<img src="img/accuracy.png" width="450px">]

---
## Tensorboard

There is a [Keras callback](https://keras.io/callbacks/) to use Tensorboard

.center[<img src="img/tensorboard_edward.png" width="650px">]

.footnote[credit: [Edward Tensorboard tuto](http://edwardlib.org/tutorials/tensorboard)]

---
class: center, middle
name: optim
# Common optimizations

.medium["avoiding overfitting"]

---
## Data is key

Deep neural nets need .red[a lot of data] to achieve good performance.

.center[<img src="img/dl_perf.jpg", width="600px"/>]

---
count: false
## Data is key

Deep neural nets need .red[a lot of data] to achieve good performance.

Use .red[data augmentation].

.center[<img src="img/data_augment.png", width="600px"/>]

---
count: false
## Data is key

Deep neural nets need .red[a lot of data] to achieve good performance.

Use .red[data augmentation].

Choose a training set .red[representative] of your data.


--
If you cannot get enough labeled data, use simulations or turn to [transfer learning](https://arxiv.org/abs/1411.1792) techniques.

---
## Dropout

A % of random neurons are .green[switched off] during training  
it mimics different architectures being trained at each step 

.center[<img src="img/dropout.png" width="500 px" />]
.footnote[[Srivastava et al. (2014)](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)]

---
## Dropout

```python
...
from keras.layers import Dropout

dropout_rate = 0.25

model = Sequential()
model.add(Conv2D(2, (3, 3), input_shape=(9, 9, 1)))
*model.add(Dropout(dropout_rate))
model.add(Conv2D(4, (3, 3)))
*model.add(Dropout(dropout_rate))
...
```

- regularization technique extremely effective
- .green[prevents overfitting]

**Note:** dropout is .red[not used during evaluation], which accounts for a small gap between **`loss`** and **`val_loss`** during training.


.footnote[[Srivastava et al. (2014)](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)]

---
## Batch normalization

```python
...
from keras.layers import BatchNormalization
from keras.layers import Activation

model = Sequential()
model.add(Conv2D(..., activation=None))
*model.add(BatchNormalization())
model.add(Activation('relu'))
```

- technique that .green[adds robustness against bad initialization]
- forces activations layers to take on a unit gaussian distribution at the beginning of the training
- must be used .red[before] non-linearities

.footnote[[Ioffe & Szegedy (2015)](http://arxiv.org/abs/1502.03167)]

---
## to go further..

Here are some leads (random order) to explore if your model do not converge:
- [data normalization](https://www.jeremyjordan.me/batch-normalization/)
- [weight initialization](https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94)
- [choice of learning rate](http://www.bdhammel.com/learning-rates/)
- [gradient clipping](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/)
- [various regularisation techniques](https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/)

---
exclude: true
## Next ?

.medium[ML developments are happening at a high pace,  
.red[stay tuned] !  

A .green[curated list] of inspirations for this presentation  
can be found [here][refs].
]

[refs]: https://github.com/aboucaud/slides/blob/master/2018/hands-on-deep-learning/references.md

---
class: center, middle

La semaine prochaine :
# .red[Generative] neural networks

---
exclude: true
class: center, middle

# Thank .red[you]
</br>
</br>
.medium[Contact info:]  
[aboucaud.github.io][website]  
@aboucaud on GitHub, GitLab  
[@alxbcd][twitter] on Twitter

[website]: https://aboucaud.github.io
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
