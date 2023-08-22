class: center, middleclass: middle
<!-- background-image: url(../img/brain.png) -->

# An introduction to .red[MLOps]
#### Ecole Rodolphe Clédassou – August 2023

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
class: middle, center
## Acknowledgement

.big.center[The inspiration for this talk comes from a seminar  
by .green[**Pierre-Marc Jodoin**] from Université de Sherbrooke  
in November 2022.]

---

## Reproducibility in research

As scientists, you must account for reproducibility.

It is often done with code such as in https://paperswithcode.com/

.center[
<img src="../img/paper-with-code.png" width="75%" />
]

---

## Software development cycle

.center[
<img src="../img/dev-cycle.png" width="95%" />
]

---

## Software development cycle

.left-column[
- Stage 1 – Development
    - Planning
    - Coding
- Stage 2 – Integration
    - Small rapids tests
    - Build
- Stage 3 – Testing
    - More in-depth tests
    - Validation
    - Release
]

.right-column[
- Stage 4 – Delivery
    - Final packaging
    - Deploy (and test) on an operating server
- Stage 5 – Monitoring
    - Collect data, monitor each function, and spot errors
    - End user feedback
]

.center[
<img src="../img/dev-cycle.png" width="45%" />
]

---

## .red[Automated] software development cycle

.left-column[
- Stage 1 – .red[**Continuous**] Development
    - Planning
    - Coding
- Stage 2 – .red[**Continuous**] Integration
    - Small rapids tests
    - Build
- Stage 3 – .red[**Continuous**] Testing
    - More in-depth tests
    - Validation
    - Release
]

.right-column[
- Stage 4 – .red[**Continuous**] Delivery
    - Final packaging
    - Deploy (and test) on an operating server
- Stage 5 – .red[**Continuous**] Monitoring
    - Collect data, monitor each function, and spot errors
    - End user feedback
]

---

## DevOps 101


.center[
<img src="../img/ci-cd-github.png" width="85%" />
]

.footnote[https://resources.github.com/ci-cd/]

---

## DevOps

.center[
  <img src="../img/devops-trident.png" width="50%" />
]

.footnote[https://aws.amazon.com/devops/what-is-devops]

---

## ML cycle


.center[
<img src="../img/mlcycle.jpg" width="75%" />
]


.footnote[credit: [ml-ops.org](https://ml-ops.org/content/motivation)]

---

## ML life cycle 101

.medium[
1. **.blue[Data extraction]**: go fetch data
2. **.blue[Data analysis]**: Understand the nature of the data (what nnUNet does)
3. **.blue[Date preparation]**: Cleaning and splitting the data
4. **.green[Model training]**: Training, validation, hyper-parameter tuning : output is a model and/or a packaged pipeline
5. **.green[Model evaluation]**: Testing: output is a set of metrics to assess the quality of the model.
6. **.green[Model validation]**: The model is confirmed to be adequate for deployment—that its predictive performance is better
than a certain baseline.
1. **.green[Model serving]**: The validated model is compiled/encapsulated and deployed to a target environment to serve
predictions.
1. **.green[Model monitoring]**: The model predictive performance is monitored to potentially invoke a new iteration in the ML
process.
]

---

## MLOps

.center[ 
<img src="../img/mlops-org.png" width="50%">

**MLOps = DevOps principles applied to ML systems**
]
- **CI** is no longer only about testing and validating code, but also testing and validating data, data schemas, and models.
- **CD** is no longer about a single software package, but a system (an ML pipeline) that should automatically deploy another service.
- **CT** (Continuous training) unique to ML systems : automatically retraining and serving the models
- **CM** (Continuous monitoring) : model decay tracking, prediction trigger

.footnote[https://ml-ops.org]

---

## MLOps workflow example

.center[
<img src="../img/mymlops-workflow.png" width="75%" />
]

.footnote[https://mymlops.com]

---
class: middle
## Today's menu


1. [**Optuna**](https://optuna.org/) : hyperparameter search engine

2. [**MLflow**](https://mlflow.org/) : ML tracking and monitoring

#### 

#### 

---
class: middle

<img src="../img/optuna-logo.png" width="40%">  

Hyperparameter .red[**search**] made easy

Test a bunch of .blue[**architectures**] on your problem and find .green[**the best**] one!

.footnote[[docs](https://optuna.readthedocs.io/) | [tutorials](https://optuna.readthedocs.io/en/stable/tutorial/index.html)]

---
## Hyperparameter Optimization with Optuna

```python
import optuna
import tensorflow as tf

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)  # <----------------- PROVIDE A RANGE TO EXPLORE
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), # <--- USE THE SAMPLED lr
                  loss='binary_crossentropy', metrics=['accuracy']) 
                
    model.fit(x_train, y_train, 
              epochs=10, 
              validation_data=(x_test, y_test))
              
    return model.evaluate(x_test, y_test, verbose=0)[1] 
  
study = optuna.create_study()  # <------------------------------------------ CREATE A STUDY
study.optimize(objective, n_trials=100)  # <-------------------------------- RUN OPTIMIZATION IN //

print(study.best_trial)
```
---
## Optimizing the model architecture

```python
import optuna
import tensorflow as tf

def create_model(trial):  # <------------------------------------------- NEED TO WRITE MODEL PROGRAMATICALLY 
    num_layers = trial.suggest_int("num_layers", 1, 3)
    first_layer_size = trial.suggest_categorical("first_layer", [32, 64, 128])
    
    model = tf.keras.Sequential()
    for i in range(num_layers):
        if i == 0: 
            model.add(tf.keras.layers.Dense(first_layer_size, activation='relu'))
        else:
            model.add(tf.keras.layers.Dense(32, activation='relu')) d
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='bce', optimizer='adam', metrics=['accuracy'])
    return model

def objective(trial):
    model = create_model(trial)  # <------------------------------------ MODEL WILL BE DYNAMICALLY CREATED
    model.fit(x_train, y_train, epochs=3)  #                             AND TRAINED BY OPTUNA
    return model.evaluate(x_test, y_test)[1]

study = optuna.create_study()
study.optimize(objective, n_trials=100) 

print(study.best_trial)
```

---
## Integrate with Keras Callbacks for tracking metrics

```python
import optuna
from tensorflow.keras.callbacks import Callback

class OptunaCallback(Callback):

  def __init__(self, study):
    self.study = study

  def on_epoch_end(self, epoch, logs=None):
    self.study.report(logs['val_acc'], epoch)  # <-------------------------- REPORT METRICS TO OPTIMIZER

study = optuna.create_study()

model.fit(x_train, y_train, 
          callbacks=[OptunaCallback(study)])  # <--------------------------- WILL BE RUN IN THE BACKGROUND ON TRAIN
```

---
class: middle

<img src="../img/mlflow-logo.png" width="30%">  

Run ML experiments = environment + data + code.

Log .red[**hyperparameters**] + .green[**results**] + .blue[**plots**].

.footnote[[docs](https://mlflow.org/docs) | [tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)]


---

## MLFlow Setup

- Install MLFlow with pip

```bash
pip install mlflow
```
--
- Set the tracking URI to record run information

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
``` 
--
- Set an experiment name to organize runs

```python
mlflow.set_experiment("ExperimentABC")
```
---
## Experiment Tracking

```python 
import mlflow

with mlflow.start_run() as run:
    lr = 0.1
    mlflow.log_param("learning_rate", lr)

    ...

    mlflow.log_metric("accuracy", 0.92) 
```

- Each run will log code, parameters, metrics to MLFlow Tracking

- Can set run name and nested runs

---
## Log parameters, metrics and models

```python
mlflow.log_param("num_layers", 3)
mlflow.log_params({"learning_rate": 0.001, "epochs": 20})
```

- Key-value pairs for hyperparameters, settings, etc.

```python
mlflow.log_metric("loss", 0.45)
```

- Record evaluation metrics like loss, accuracy
- These metrics can be plotted in the UI so you can compare runs

```python
mlflow.tensorflow.log_model(model, "Name_for_model")
```
- Record the architecture and weights of the model in the [`MLModel` format](https://mlflow.org/docs/latest/models.html).

---
## Full example with manual logging (best results)

```python
import tensorflow.keras as tfk

import mlflow
import mlflow.tensorflow

mlflow.set_experiment("ExperimentABC")

model = tfk.models.Sequential()
model.add(tfk.layers.Dense(64, activation='relu', input_shape=(10,))) 
model.add(tfk.layers.Dense(1, activation='sigmoid'))

learning_rate = 0.01

model.compile(
  optimizer=tfk.optimizers.Adam(lr=learning_rate, decay=0.1), 
  loss='binary_crossentropy', 
  metrics=['accuracy']
)

mlflow.tensorflow.log_model(model, "model_name")  # <------------------------ ARCHITECTURE

with mlflow.start_run() as run: 
    model.fit(X_train, y_train, epochs=5)

    mlflow.log_metric("loss", model.evaluate(CX_test, y_test)[0])  # <------- SCORE
    mlflow.log_param("learning_rate", learning_rate)  # <-------------------- HYPERPARAMETER
```

---
## If you are lazy (like me) => `autolog`

```python
import mlflow

mlflow.tensorflow.autolog()

model = ...
model.compile(...)
model.fit(...)

```

`autolog` will store

- Metrics and Parameters
   - training loss; validation loss; user-specified metrics
   - `fit()` or `fit_generator()` parameters; optimizer name; learning rate; epsilon
- Artifacts
   - model summary on training start
   - `MLflow Model` (Keras model)
   - TensorBoard logs on training end

⚠️️ Only compatible with 2.3.0 <= tensorflow <= 2.13.0 

---
## Best feature of MLFlow : logging plots \\0/

```python
import mlflow
import tensorflow as tf
import matplotlib.pyplot as plt

mlflow.set_experiment("ExperimentABCwithPlots")  # <------------- SET EXPERIMENT

with mlflow.start_run() as run:

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10)

    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Model Loss Progress'); plt.ylabel('Loss'); plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('loss_plot.png')

  mlflow.log_artifact('loss_plot.png')  # <---------------------- THE PLOT WILL BE STORED IN THE DB 
                                        #                         AND ASSOCIATE IT WITH THE EXPERIMENT RUN
                                        #                         IT CAN NOW BE VIEWED IN THE MLFLOW UI
```

---
## Take home message

<br>
Machine Learning is a .red[**powerful tool**] for physicists and gives state-of-the-art results  
for .green[**detection**] and .green[**classification**] tasks.
<br>
<br>
<br>
Research should be .red[**reproducible**] and .green[**open source**], and therefore if your research includes ML, you should care about using dedicated tools to make your experiments reproducible.
<br>
<br>
<br>
Incorporating MLOps in .green[**your daily workflow**] will come with a lot of .red[**benefits**]  
(gain of time, engineering expertise, trust among peers, etc.).

---
class: center, middle

# Thank .red[you] for your attention
</br>
</br>
Find this presentation at  
https://aboucaud.github.io/slides/2023/euclid-school-mlops
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
