class: middle
background-image: url(../img/brain.png)

# Machine learning for cosmology

#### International Conf. of the Physics of the 2 Infinites – University of Kyoto – March 28 2023 
#### [Alexandre Boucaud](mailto:aboucaud@apc.in2p3.fr) – APC/IN2P3/CNRS

.footnote[
  <img src="../img/apc_logo_transp.png" height="100px" alt="Astroparticule et Cosmologie" style="float: left">
  ]

---
exclude: true

.footnote[<a href="https://arxiv.org/abs/2210.01813" target="_blank"><img src="https://img.shields.io/badge/astro--ph.IM-arXiv%3A2210.01813-B31B1B.svg" class="plain" style="height: 25px"/></a>]

---
## Importance of ML in physics today

.left-column[
  ML has become .red[ubiquitous] in physics and for some aspects completely .red[uncanny].


  3 dedicated plenary talks in this conference

  - **ML for Gravitational Physics**   
    Tuesday 28 at 9:00 by _Jonathan Gair_  
  - **ML for HEP**  
    Thursday 30 at 9:25 by _Tobias Golling_
]

.right-column[
.center[Keywords frequency on _arXiv:astroph_]
<img src="../img/arxiv-neural-2022.png" , width="560px" />
</br>
</br>
.right[Huertas-Company+22]
]

---
## Upcoming very large imaging surveys

.left-column[
  <img src="../img/big_surveys.png" , width="580px", vspace="0px">
  </br>
  </br>
  .right[credit: LSST DC2 simulation]
]

.right-column[
  .center[Euclid imaging footprint vs. time]
  .right[
    <iframe width="400" height="300" src="../img/vid/hstvseuclid.mp4" title="Euclid vs HST" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
    </br>
    </br>
    courtesy Jarle Brinchmann]
]

---
## with an increased complexity

.left-column[<img src="../img/complex_surveys.png" , width="680px", vspace="0px">]

.right-column[
  .right[<img src="../img/complex_simulations3.png", width="55%", vspace="0px">]
]

---
## Where can data driven solutions help ?

--

.medium[Physical model .green[**too complex**] or dataset .green[**too large**], leading to convergence issues] 

.right.big[➡️ .red[**speed**]]

--

.medium[No suitable .green[**physical model**] available]

.right.big[➡️ .red[**accuracy**]]

--

.medium[There might be .green[**hidden information**] in the data, .green[**beyond the summary statistics**] we traditionally compute]
  
.right.big[➡️ .red[**discovery**]]

---
## Outline

This presentation will focus on some of the successes of ML in cosmology, in particular through three wide themes

1. .medium[Performant feature extraction on images]
2. .medium[Non-linear (and differentiable) simulations]
3. .medium[Field-level inference]

</br>
</br>
</br>
.small[⚠️ please keep in mind that, for brevity, the chosen examples in the reminder of this presentation are quite subjective and do not reflect the full diversity of the literature. For a much broader overview and extensive references, I highly recommend the recent review from Huertas-Company & Lanusse +22]
.footnote[<a href="https://arxiv.org/abs/2210.01813" target="_blank"><img src="https://img.shields.io/badge/astro--ph.IM-arXiv%3A2210.01813-B31B1B.svg" class="plain" style="height: 25px"/></a>]

---
class: middle, center
# .red[Feature extraction] with </br>Convolutional Neural Networks

---
## Convnets aka CNNs

Neural networks with convolution layers that learn the mapping between .blue[**images or spectra**] and .red[**summary statistics**] (floats) used for tasks such as detection, classification or regression

.center[
  <img src="../img/blend_to_flux.png", width="100%", vspace="0px", hspace="0px"/>
]

.footnote[Boucaud+ 19]

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
name: density
# Field-level .red[inference] <br>with .red[neural density estimation]

---
## Cosmological field constraints

.center[
  <img src="../img/cnn_res4.png", width="90%", vspace="0px", hspace="0px"/>
]

---

.center[
  <img src="../img/francois_marginal_like.png", width="95%", vspace="0px", hspace="0px"/> 
]

.footnote[courtesy F. Lanusse]

---
exclude: true
## Neural density estimation

.center[<iframe width="590" height="390" src="../img/vid/NF_explanation_justine.mp4" title="Justine NF explanation" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>]

.footnote[credit: Justine Zeghal]

---
## Cosmological constraints from SBI

.left-column[
<img src="../img/sbicosmo.png" width="140%" />
]

.right-column[
1. maps are optimally compressed
2. 
]

.footnote[courtesy: M. Huertas-Company]

---
## Accelerating SBI with gradients <img src="../img/justine.png" class="circle-image" alt="JZ" style="float: right">

.left-column[
<img src="../img/npe_nle_lv_justine.png" width="90%" />
]

.right-column[
  .right.small[Justine Zeghal<br> PhD student @ APC]
  .medium[On a classical SBI benchmark our preliminary results indicate that incorporating gradients in the density estimation process (NLE or NPE) yields a gain in the number of simulations needed for a given metric.  
  Method is currently being validated on cosmological inference ($\sigma_8-\Omega_c$) from simulated WL log-normal mass maps.]  
]

---
## Take home messages

In this presentation we have seen that in cosmology, deep learning techniques

- can very well .blue[**extract implicit information**] from multi-band images / spectra to perform tasks such as .green[**detection or classification**].
.right[.red[***Caveats***] : _model response very non-linear, needs a lot of calibration for regression_]

- can help .blue[**scale up numerical simulations**] by providing a .green[**non-linear approximation of the physical process**] otherwise very expensive to compute.
.right[.red[***Caveats***] : _often biased in some regimes, need representative training_]

- can be used to .blue[**optimally compress**] the cosmological field information. Simulation-based inference schemes can then leverage that information to provide .green[**amortized inference**] through neural density estimation.
.right[.red[***Caveats***] : _black-box process, requires lots of simulations, need to check systematics_]