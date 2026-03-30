class: center, middleclass: middle

# .red[IA générative] pour le développement

#### [Journées développements logiciels IN2P3](https://indico.in2p3.fr/event/39054/) – 30 mars 2026
#### [Alexandre Boucaud][mail] – AstroParticule & Cosmologie

.bottomlogo[<img src="../img/apc_logo_transp.png" height="100px" alt="Astroparticule et Cosmologie" style="float: left">]

[mail]: mailto:aboucaud@apc.in2p3.fr

---

# Un sujet hautement sensible

Comme nous le verrons à travers les résultats du sondage, ce sujet est aussi clivant au sein de l'IN2P3 que dans la société française.

Parmi les adeptes, les demandes sont

- la .green[**sécurité des données**] et la mise en place d'.green[**outils institutionnels**] avec un .green[**usage raisonné**]
- la nécessité d'avoir .green[**accès aux outils de pointe**], .blue[**par API**], pour rester compétitifs
- la .blue[confiance] reste .blue[limitée] et la peur d'un .red[**appauvrissement intellectuel**] est présente

Les raisons principales des oppositions fortes,

- un .red[gouffre énergétique et matériel] à l'impact écologique .red[non soutenable]
- une .red[source de précarisation du travail] et d'affaiblissement
- un .blue[**choix**] technologique qui nécessite un .blue[**débat collectif**]

.footnote[Manifeste d'objecteurs de conscience de l'IAg https://atecopol.hypotheses.org/13082]

---
background-image: url(../img/cianum_consultation2026.png)

.hidden[toto]  
.hidden[tototototototototototo]
👉 [Lire les résultats de la consultation][consult]

[consult]: https://www.conseil-ia-numerique.fr/le-numerique-est-une-affaire-collective-decouvrez-les-resultats-de-la-consultation-citoyenne-du

---

# Et qui amène des questions légitimes

.center[
    <img src="../img/claude-cc.png" width="60%" />
]

.center[
    <img src="../img/claude-cc-answer.png" width="60%" />
]

_"Je souligne que la plupart des IA connaissent parfaitement le code des pipeline Rubin-LSST (US). Ca aide énormément à interroger les différentes plateformes qui fournissent des données."_

.right.small[Un répondant du sondage]

---
class: middle
# Outline

1. Motivations

2. Capacités des modèles fin Mars 2026

3. Résultats du sondage

4. Quelques initiatives à noter

5. Discussion

---

# L'IA générative au CNRS: Emmy

Mi-décembre 2025, lancement d’.blue[**Emmy**], l’IA générative .red[pour les agents du CNRS], dont les capacités sont

-	traduction de textes en toutes langues ;
-	synthèses de documents ;
-	aide à la reformulation ;
-	aide à la réflexion ;
-	recherche sur le web ;
-	reconnaissance de textes et d’images ;
-	mode « raisonnement » : l’IA traite la question de l’utilisateur étape par étape afin de donner une réponse plus pertinente et plus complète ;
-	collections de documents

Cet outil résulte d'un accord passé avec l’entreprise française Mistral AI pour 35 000 utilisateurs pour l'usage de leur offre .blue[**Le Chat Entreprise**].

.footnote[https://emmy.cnrs.fr/]

---

# Voeux 2026 du président du CNRS

_"L’objectif [de la mise en place de Emmy] est de permettre à chacun et chacune de mieux comprendre les opportunités que peut lui offrir l’IA dans son cadre professionnel, tout en identifiant aussi ses risques et ses limites. Et bien entendu, ces expérimentations doivent se faire avec modération, .green[**l’empreinte carbone de l’utilisation de l’IA étant un enjeu majeur**]."_
.right[– Antoine Petit]

--

=> Pas de positionnement, mais plutôt à chacun de juger.

--

=> L'usage doit se faire avec modération, sans plus de précision.

---

# Newsletter CNRS de février 2026..

« _[Emmy] a été testé en interne par .blue[**près de 600 agents**] volontaires qui nous ont fait part de leurs retours d’expérience. Grâce à ce travail de preuve de concept, .blue[**nous avons ainsi pu identifier les besoins des agents**] et mesurer quelles étaient les problématiques techniques associées à la généralisation d’un tel outil. Ce type de déploiement à grande échelle est en effet inédit dans le monde de la recherche en France._ »

.footnote[[Newsletter CNRS 3 février 2026](https://www.cnrs.fr/fr/actualite/le-cnrs-lance-avec-mistral-ai-un-outil-dia-generative-securise-mis-disposition-de-ses)]

--

« _En signant ce partenariat avec Mistral AI, nous sommes assurés que les données des agents du CNRS qui utilisent l’outil Emmy seront hébergées au sein de datacenters situés sur le sol européen notamment soumis au RGPD et à l’IA Act, garantissant la sécurité de l’usage des données et leur intégrité. Elle conservera alors une trace des informations qui lui ont été fournies afin d’apporter la réponse la plus adaptée et la plus pertinente possible étant précisé qu’.green[**Emmy n’exploite ni ne réutilise les prompts et les documents saisis par les utilisateurs dans le but d’entraîner les modèles d’intelligence artificielle et autres produits développés par Mistral AI**]._ »

---

# ..qui vient avec des directives

« _Nous avons également rédigé un document présentant les conditions d’utilisation d’Emmy au CNRS et .red[proscrivant l’utilisation de toute autre IA généraliste grand public]._ »

.footnote[[Newsletter CNRS 3 février 2026](https://www.cnrs.fr/fr/actualite/le-cnrs-lance-avec-mistral-ai-un-outil-dia-generative-securise-mis-disposition-de-ses)]

--

réitérée une seconde fois

« _[...] le déploiement de cet agent conversationnel sécurisé au sein du CNRS implique .red[l’interdiction de l’usage d’autres outils aux fonctionnalités similaires, dans le cadre du travail des agents]._ »

--

avant de parler d'une charte à venir sans échéance

« _Une charte .blue[encadrant l’utilisation de l’IA au sein de l’organisme] est aussi en cours de préparation._ »

---

# Plus de questions que de réponses

En plus de confondre "IA généraliste" et "IA générative", le CNRS proscrit à ses agents l'utilisation de .blue[**toute autre IA (que Emmy)**], puis insiste sur le fait que cette restriction s'applique à l'usage d'autres outils .blue[**aux fonctionnalités similaires**].

--

Emmy fournissant seulement un accès à une interface web, sans API permettant de faire de l'inférence par requête, on voit que .red[les besoins pour le développement n'ont pas été pris en compte] dans "l'identification des besoins des agents".

--

Cette interdiction s'applique à tous les agents ayant accès à Emmy. Or bon nombre de laboratoires du CNRS sont des UMR et accueillent une masse importante de personnel universitaire qui

- n'a pas accès à Emmy
- ne sont donc pas soumis à ces interdictions

---

Le manque de feuille de route claire sur le sujet de l'IA générative au CNRS (et dans l'ESR), de débat ou de transparence sur la stratégie, associé à cette interdiction, a pour effet

- de .red[créer des inégalités] entre les personnels des laboratoires (CNRS vs Université), en particulier les profils juniors qui doivent faire face à "la compétition"

--

- de .red[favoriser l'absence de transparence] (_acknowledgement_) propre au métier de la recherche, lors de l'utilisation d'IA générative dans le cadre du travail

--

- de .red[renforcer l'isolement des personnels] qui sont livrés à eux mêmes face à ces questions et qui profitent des outils "interdits" par d'autres intermédiaires : une collaboration scientifique (CERN), une offre pour l'éducation (Copilot) ou une souscription personnelle

--

.blue[**L'échelle de l'IN2P3**] est sans doute plus appropriée que celle du CNRS pour assurer l'inclusitivé de l'ensemble des personnels CNRS + Université, d'où la décision de faire ce sondage.

---
class: middle

# Etat de l'art et impact sur notre métier

---

# L'IA agentique en 1 slide

Les modèles de langage de 2024 sont devenus en 2025 des .blue[modèles de raisonnement] (chain-of-thought), permettant de travailler par étapes et sur de plus larges données d'entrée.

--

Puis ils ont été dotés d'outils permettant d'accomplir des calculs ou de faire des actions (récupérer le contenu d'une page internet, écrire, exécuter un script puis lire le résultat).

--

En leur donnant la capacité de s'évaluer à la fin d'une tâche, pour éventuellement recommencer autrement, ils sont devenus des .blue[_agents_].

--

Les agents se sont ensuite spécialisés à des tâches précises (ex. coder, lire un pdf, chercher sur internet), ce qui a créé le concept de .blue[multi-agents]. Un agent orchestrateur qui gère des agents spécialisés et qui agrège le résultat.

--

Jusqu'à fin 2025, ces agents .red[faisaient beaucoup d'erreurs ou étaient inutilement très verbeux en code], d'où un sentiment d'outil peu fiable.

---

# Le Model Context Protocol

Anthropic a mis en 2025 au point un protocol de communication pour que les agents puissent utiliser très facilement les outils à travers leur API, ce qui permet de retourner directement des résultats dans le bon format.

.center[
  <img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Agents/image_35.png" width="70%">
]

.footnote[Credit https://lbourdois.github.io/blog/LLM_Agents/]

---

# La personnalisation des agents

La personnalisation des agents s'est normalisée autour d'un fichier .red[`AGENT.md`] qui pour un projet donné contient tout le contexte qu'il doit savoir et peut servir de mémoire statique. 

--

Afin d'éviter de répéter régulièrement les mêmes ordres à un agent pour qu'il accomplisse des tâches spécifiques, des fichiers de compétences .red[`SKILLS.md`] ont vu le jour, permettant d'augmenter et de focaliser leur capacités.

--

La miniaturisation et démocratisation des .blue[outils de vision] (OCR) permet également aux agents d'obtenir une description très bonne de n'importe quelle image, ce qui rajoute la capacité de "voir" tout ce que voit un humain sur son écran (screenshots)

---

# Début 2026 changement de paradigme

.left-column[

  Un bond très important dans les performances depuis décembre 2025 et un deuxième en février 2026.
  
  Cet effet s'est notamment vu d'abord avec la sortie de Google Gemini 3 Pro, puis des modèles Anthropic Claude Opus 4.6 (large) et Claude Sonnet 4.6 (médium).

  .green[La fiabilité et la focalisation des modèles fait un **énorme** bond en avant] et les rend beaucoup plus utiles aux tâches de développement.

]

.right-column[
  .center[
  <img src="../img/claude-shipping-feb2026.jpeg" width="80%">
  ]
]

---

# Benchmark sur la détection d'erreurs

.center[
<img src="../img/bullshitbenchv2-march2026.png" width="80%">
]

.footnote[https://petergpt.github.io/bullshit-benchmark]

---

# Impact sur notre métier

.left-column[
Divers études récentes (souvent biaisées car montées par les grands groupes eux mêmes) indiquent le pourcentage des tâches classiques d’un corps de métier que l’IA serait en capacité d’accomplir.

On voit que notre métier fait partie de ceux qui vont subir une pression forte de l'IA, qu’on le veuille ou non.
]

.right-column[
  .center[
    <img src="../img/anthropic-labor-impact.png" width="85%" />
  ]
  
]

.footnote[https://www.anthropic.com/research/labor-market-impacts - March 2026]

---

##  Clawdbot / OpenClaw (aparté)

En parallèle, à partir de fin Novembre 2025, un projet d'IA agentique indépendante voit le jour.

.center[<img src="https://imgs.search.brave.com/GLssrqcoxMIafoEOGcEtWCrEHznGg0GOfa-q73oq-oY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9kb2lt/YWdlcy5ueWMzLmNk/bi5kaWdpdGFsb2Nl/YW5zcGFjZXMuY29t/LzAwOEFydGljbGVJ/bWFnZXMvV2hhdC1J/cy1PcGVuQ2xhdy9X/aGF0JTIwaXMlMjBP/cGVuQ2xhdy5wbmc" width="40%">]

Il conceptualise la mise à disposition d’une machine entière à un super-agent (équivalent Jarvis dans IronMan ©) qu'il va piloter de manière autonome : contrôle de la boîte mail, gestion de tous les fichiers sur la machine, pilotage à distance par messagerie (WhatsApp / Telegram). On commence par lui donner une âme .red[`SOUL.md`] et on le laisse faire ses tâches.  
[Et ces agents deviennent des bullies..](https://simonwillison.net/2026/Feb/12/an-ai-agent-published-a-hit-piece-on-me/)

---

## Le concept d'AI scientist (aparté)

.left-column[
.center[
  <img src="../img/denario.png" width="100%">
]
]

.right-column[
  .center[
    <img src="../img/denario-steps.png" width="100%">
  ]
]

.footnote[https://arxiv.org/pdf/2510.26887 discuté par Julien Zoubian lors des réunions RI3-RAGLABS]

---

## Un essai intéressant qui ramène les rapports humains au centre de l'équation (aparté)

.left-column[
  .center[
    <img src="../img/hogg-why-astro.png" width="100%" />
  ]
  .credits[https://arxiv.org/abs/2602.10181 - February 2026]
]

.right-column[
  Pitch:
  - l'arrivée des IA génératives puissantes arrive
  - plusieurs options s'offrent à nous en tant que scientifiques
  - deux sont particulièrement mauvaises (l'acceptation du "tout IA") et le rejet complet de leur utilisation par "flicage" des chercheurs
  - il en profite pour redire les fondamentaux qui font qu'on aime ce travail et ouvre quelques perspectives sur notre manière de travailler avec l'IA
]

---
class: middle

# Résultats du sondage

#### https://machine-learning.pages.in2p3.fr/llm-survey-2026

---
# Robustesse des statistiques

Première analyse réalisée après 4 jours de sondage avec 291 réponses.

Deuxième analyse avec 451 réponses après la fin du sondage le 28/03.

Les statistiques sont .green[**équivalentes**] alors que certains labos avaient bien moins de répondants, indiquant une .green[**bonne représentativité des résultats**].

---
class: middle
.center[ ![](../img/taches_developpement.svg) ]
---
class: middle
.center[ ![](../img/accompagnement_par_statut.svg) ]
---
class: middle
.center[ ![](../img/niveau_confiance.svg) ]
---
class: middle
.center[ ![](../img/freins_usage.svg) ]
---
class: middle
.center[ ![](../img/freins_par_categorie.svg) ]
---
class: middle
.center[ ![](../img/freins_users_vs_nonusers.svg) ]
---
class: middle
.center[![](../img/score_par_statut.svg)]
---
class: middle
.center[![](../img/score_par_frequence.svg)]
---
class: middle
.center[![](../img/taux_paiement_par_statut.svg)]
---

Julien Zoubian a réalisé un sondage au CPPM en Novembre 2025 avec des résultats similaires.

.center[
<iframe src="https://jzoubian.pages.in2p3.fr/slides/20251218_ia_sondage_cppm" width="800px" height="450px"></iframe>
]

.footnote[https://jzoubian.pages.in2p3.fr/slides/20251218_ia_sondage_cppm]

---

# Remarques - morceaux choisis

_"Le plaisir du développement et de la réflexion disparait en promptant"_

*"I am afraid of becoming stupid, loosing my brain power"*

*"Financement de ces outils au meme titre que n'importe quel logiciel de CAO"*

*"Restreindre l'usage de l'IA au vu de ses implications sociales, éthiques et environnementales délétères."*

*"Un rappel clair et précis sur les effets sociaux de l'IA générative"*

*"Les personnels universitaires ne peuvent pas bénéficier des mêmes outils que les personnels CNRS dans une UMR, ce qui est préjudiciable."*

---

# Remarques - morceaux choisis (suite)

*"Notre rôle de scientifique est d'être des dépositaires humains de la connaissance, si nos compétences et notre capacité à raisonner sont dépendantes d'outils qui nous privent de notre réflexion propre, je pense que notre intégrité est menacée. Il est clair que l'usage de LLM a des effets très négatifs sur une partie de la population, je pense que nous jouons un rôle d'exemplarité dans la prudence vis-a-vis de ces outils."*

*"all shall use AI, from all scietifique domains .. it is not a choice ..."*

*"Je n'utilise pas l'IA par fierté. J'estime être en capacité de produire de moi même ce qui pourrait concerner mes demandes. Bien évidement dans le cadre de mon travail. Cependant comme tout outil j'ai bel et bien conscience de son utilité et de son efficacité."*

---

# Volontés qui ressortent majoritairement

- Un **cadre réglementaire** clair

- Service d’inférence .green[**sécurisé et souverain**] .blue[**pour tous**] (CNRS + Universitaires)  
=> l'API de MistralAI est très demandée ou une plateforme intégrée au CC-IN2P3
  
- Privilégier les .green[**modèles open source**] et les modèles auditables

- S’assurer d’avoir accès aux .green[**modèles de pointe**] si on veut remporter l’adhésion et éviter l'utilisation des modèles

- L’impact écologique étant un frein globalement partagé, .red[**définir des quotas**] (projet, labo) tout comme on a sur le calcul + mettre en place un .green[**calcul de la consommation**]

---

# Volontés qui ressortent majoritairement

- Favoriser une utilisation modérée pour des tâches identifiées car l'usage très régulier de l'IA risque de créer une .red[perte globale de connaissances]  
=> le développement est un savoir-faire et il faut .green[**le préserver**]

- La .blue[transparence sur l'utilisation] est nécessaire

- Faire .red[attention aux licences]  
=> voir présentation de Philippe

---
class: middle

# Quelques initiatives dans l'ESR

---

# Chartes IA

Discuté dans Reprises

- **Cadre légal** : RGPD, droits d’auteur, conformité sectorielle (ex. : santé, éducation).
- **Principe d’éthique** : Transparence, responsabilité, équité, respect de la vie privée.
- **Bonnes pratiques** :
  - limites d’usage
  - validation humaine des résultats
  - traçabilité des contenus générés
  - impact écologique

--

.right.medium[Doit être pédagogique !]

Exemples:

- [portail des chartes IA dans l'administration](https://alliance.numerique.gouv.fr/cartographie/portail-des-chartes-ia-dans-ladministration/)
- [template KairoiAI utilisé comme base par le LIP6](https://github.com/KairoiAI/Resources/blob/main/Template-ChatGPT-policy.md)
- [charte personnelle d'un doctorant](https://kilianrouge.github.io/posts/2026/2_AI_Charter)

---

# Usage de l'IAg en parallèle d'un cours

Usage des llm dans le cadre d'un cours d'astrophysique à Harvard.

Les points clés

- gros travail de préparation en amont sur les prompts (tout est partagé dans l'article)
- entraînement en RAG sur un document de cours
- restriction de l'IA à de courtes réponses .blue[**avec citation du cours**] et un non-engagement de l'étudiant dans une discussion
- indications de quand l'utilisation est autorisée et quand elle est fortement déconseillée  
  **=> a beaucoup plu aux étudiant**

.footnote[Stubbs et al. 2026 - https://arxiv.org/abs/2602.04389]

---

# Ecole thématique Labobots / AISSAI

Un second semestre thématique du [centre interdisciplinaire du CNRS AISSAI][aissai] est organisé cette année en partenariat avec l'IN2P3.
Dans ce cadre, une école / ANF intitulée Labobots sera proposée à la rentrée (29 septembre au 2 octobre 2026 à Saint-Rémy-lès-Chevreuse) par l'équipe du RI3-RAGLABS 

.left-column[
- Françoise Bouvet (IJCLab)
- David Rousseau (IJCLab)
- David Chamont (IJCLab)
- Hugo Bacard (IJCLab)]

.right-column[
- Sébastien Gadrat (CCIN2P3)
- Imed Magroune (CEA)
- Anne-Laure Méalier (Centrale Mediterrannée)
- Alexandre Boucaud (APC)]

Annonce à venir..

.footnote[
  <img src="../img/aissai-logo.png" height='80px' alt="AISSAI"> 
]

[aissai]: https://aissai.cnrs.fr/en/

---

# Inférence as a Service dans l'ESR

.left-column[
  ILaaS

  _Une fédération mutualisée visant une IA générative de confiance, robuste, éthique, et sobre_

  Met à disposition une API d'inférence vers des modèles open-source.

  .blue[Voir si ça aurait un sens que l'IN2P3 contribue à cette fédération ?]
]

.right-column[
.center[
    <img src="../img/ilaas-service.png" width="100%" />
  ]
]

.footnote[https://ilaas.fr]

---

# Inférence as a Service dans l'ESR

.left-column[
  .center[
    Albert – API de la DINUM
    <img src="../img/albert-dinum.png" width="100%" />
  ]
  .credits[https://albert.sites.beta.gouv.fr]
]

.right-column[
  .center[
    Claude Code + Albert API = `le-claude`  
    <img src="https://raw.githubusercontent.com/EiffL/le-claude/main/assets/le-claude.png" width="90%" />
  ]
  ```bash
  npx le-claude
  ```
  .credits[https://github.com/EiffL/le-claude]
]

---
class: middle

# Place aux discussions

---

.hidden[toto]
### Charte IA
  
quels usages veut-on proscrire à l'IN2P3 ?  
conséquences en cas de non respect ?  
prévalence entre charte CNRS et chartes labo / institut ?

--

### Formation des personnels

but de ces formations ?  
a quelle échelle (labo, in2p3, cnrs) ?  
sensibiliser aux questions éthiques et à la souveraineté des données  
temps 

---
.hidden[toto]
### API institutionnelle sécurisée

demander accès à l'API Mistral au CNRS
possibilité d'utiliser la ferme locale pour de l'inférence (à travers les notebooks par exemple)  
mise en place de quotas utilisateurs  
mise en commun des moyens pour l'ESR

