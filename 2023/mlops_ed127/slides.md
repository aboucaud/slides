class: center, middleclass: middle
<!-- background-image: url(../img/brain.png) -->

# Introduction to .red[ML reproducibility]
#### Cours de ML – ED 127 – June 2023

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

## Reproducibility in research

As scientists, you must account for reproducibility.

It is often done with code such as in https://paperswithcode.com/


---
## ML cycle


.center[
<img src="../img/mlcycle.jpg" width="75%" />
]


.footnote[credit: [ml-ops.org](https://ml-ops.org/content/motivation)]

---
class: middle
## Today's menu


1. [Hydra](https://hydra.cc/) : ML script configuration

2. [MLflow](https://mlflow.org/) : tracking ML experiments and monitoring changes

3. [DVC](https://dvc.org/) : versoning data like code

#### 

#### 

---
class: middle
## Hydra

Hyperparameter .red[**configuration**] made easy for ML jobs

👉 see [slides](https://docs.google.com/presentation/d/12xlz64yDC3lQfBacp0vvUM2EX6EitmM1O9clCNa2-HI/edit?usp=sharing) .small[git p_from Club des développeurs – March 2023_]

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
.center[see **DEMO**]

---

## Take home message

<br>
Machine Learning is a .red[**powerful tool**] for physicists and gives state-of-the-art results for .green[**detection**] and .green[**classification**] tasks.
<br>
<br>
<br>
Research should be .red[**reproducible**] and .green[**open source**], and therefore if your research includes ML, you should care about using dedicated tools to make your experiments reproducible.
<br>
<br>
<br>
Incorporating MLops in .green[**your daily workflow**] as a PhD will come with a lot of .red[**benefits**] (e.g. help with manuscript, boost career outside of academia).

---
class: center, middle

# Thank .red[you] for your attention
</br>
</br>
Find this presentation at  
https://aboucaud.github.io/slides/2023/mlops_ed127
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
