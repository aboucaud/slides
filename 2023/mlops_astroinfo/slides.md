class: center, middleclass: middle
<!-- background-image: url(../img/brain.png) -->

# An introduction to .red[MLOps]
#### Ecole CNRS AstroInfo2023 – June 2023

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


1. [**Hydra**](https://hydra.cc/) : ML experimentation

2. [**MLflow**](https://mlflow.org/) : ML tracking and monitoring

3. [**DVC**](https://dvc.org/) : versioning data like code

#### 

#### 

---
class: middle

.center[
<img src="../img/hydra.png" width="100%">  
]

Hyperparameter .red[**configuration**] made easy for ML workflows

👉 see [slides](https://docs.google.com/presentation/d/12xlz64yDC3lQfBacp0vvUM2EX6EitmM1O9clCNa2-HI/edit?usp=sharing) .small[_from Club des développeurs – March 2023_]

---
class: middle
## MLflow

Run ML experiments = environment + data + code.

Log .red[**hyperparameters**] + .green[**results**] + .blue[**plots**].
<br>
<br>
.center[see **DEMO**]

---
class: middle
## DVC (data version control)

In the data-driven world, data should be treated like code. 

DVC allows you to version and keep a full .red[**history of data changes**].
<br>
<br>
.center[see **DEMO from Julien at 2:00 pm**]

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
https://aboucaud.github.io/slides/2023/mlops_astroinfo
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
