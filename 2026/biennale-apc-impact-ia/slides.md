class: titre

<!-- .gold-stripe[] -->

# Agentic AI impact on our activities
.gold-bar[]
## Philippe Bacon - Paul Zakharov - Alexandre Boucaud
### Biennale APC - Juin 2026

.apc-logo[]

---
class: deux-colonnes

# Un sujet hautement sensible

Comme nous le verrons à travers les résultats du sondage, ce sujet est particulièrement clivant au sein de l'APC, de l'IN2P3 ou bien dans la société française.

.colonnes[
.col-gauche[

## Les adoptants

Parmi les adeptes, les demandes sont

- .navy[**sécurité des données**] et la mise en place d'.terra[**outils institutionnels**] avec un .navy[**usage raisonné**]
- la nécessité d'avoir .navy[**accès aux outils de pointe**], .terra[**par API**], pour rester compétitifs
- la .navy[**confiance**] reste .navy[**limitée**] et la peur d'un .red[**appauvrissement intellectuel**] est présente
]
.col-droite[

## Les détracteurs

Les raisons principales des oppositions fortes

- un .red[gouffre énergétique et matériel] à l'impact écologique .red[non soutenable]
- une .red[source de précarisation du travail] et d'affaiblissement
- un .blue[**choix**] technologique qui nécessite un .blue[**débat collectif**]
]
]

.footnote[Manifeste d'objecteurs de conscience de l'IAg https://atecopol.hypotheses.org/13082]

---
class: fullbleed
background-image: url(../img/cianum_consultation2026.png)
background-size: cover

.hidden[toto]  
.hidden[tototototototototototo]
👉 [Lire les résultats de la consultation][consult]

[consult]: https://www.conseil-ia-numerique.fr/le-numerique-est-une-affaire-collective-decouvrez-les-resultats-de-la-consultation-citoyenne-du

---
class: citation

.quote-mark["]

> L’objectif [de la mise en place de Emmy] est de permettre à chacun et chacune de mieux comprendre les opportunités que peut lui offrir l’IA dans son cadre professionnel, tout en identifiant aussi ses risques et ses limites. Et bien entendu, ces expérimentations doivent se faire avec modération, .navy[**l’empreinte carbone de l’utilisation de l’IA étant un enjeu majeur**].

.cite-bar[]
.source[Antoine Petit · Voeux 2026]

---

# Purpose of this Working Group

- Collect discussion on the use of **AI in research**.
- Identify key areas of focus through a **charter or guidelines**.

.terra[**Disclaimer:** Our role is not to attack nor defend AI, but to examine how to use it effectively in research.]

## Agentic AI & LLMs

- Overview of **agentic AI** and **Large Language Models (LLMs)**.
- Focus on **practical applications** in research contexts.

---
class: outline

# Outline

1. Agentic AI as of June 2026

2. IN2P3 / APC survey results

3. Feedback from the discussion sessions

4. Major topics for discussion

5. Discussion time

---
class: section

<!-- .section-num[01] -->
<!-- .arc[] -->
.section-label[Section 01]
.section-bar[]

# Agentic AI as of June 2026

## to put everything in context

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

  .terra[**La fiabilité et la focalisation des modèles fait un énorme bond en avant**] et les rend beaucoup plus utiles aux tâches de développement.

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

# Clawdbot / OpenClaw (aparté)

En parallèle, à partir de fin Novembre 2025, un projet d'IA agentique indépendante voit le jour.

.center[<img src="https://imgs.search.brave.com/GLssrqcoxMIafoEOGcEtWCrEHznGg0GOfa-q73oq-oY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9kb2lt/YWdlcy5ueWMzLmNk/bi5kaWdpdGFsb2Nl/YW5zcGFjZXMuY29t/LzAwOEFydGljbGVJ/bWFnZXMvV2hhdC1J/cy1PcGVuQ2xhdy9X/aGF0JTIwaXMlMjBP/cGVuQ2xhdy5wbmc" width="40%">]

Il conceptualise la mise à disposition d’une machine entière à un super-agent (équivalent Jarvis dans IronMan ©) qu'il va piloter de manière autonome : contrôle de la boîte mail, gestion de tous les fichiers sur la machine, pilotage à distance par messagerie (WhatsApp / Telegram). On commence par lui donner une âme .red[`SOUL.md`] et on le laisse faire ses tâches.  

.right[[Et ces agents deviennent des bullies..](https://simonwillison.net/2026/Feb/12/an-ai-agent-published-a-hit-piece-on-me/)]

---

# Le concept d'AI scientist (aparté)

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
class: deux-colonnes
# Un essai qui insiste sur les rapports humains dans nos métiers

.colonnes[
.gauche[
  .center[
<img src="../img/hogg-why-astro.png" width="100%" />

https://arxiv.org/abs/2602.10181 - February 2026
  ]
]

.droite[
## Pitch

- l'arrivée des IA génératives puissantes arrive
- plusieurs options s'offrent à nous en tant que scientifiques
- deux sont particulièrement mauvaises (l'acceptation du "tout IA") et le rejet complet de leur utilisation par "flicage" des chercheurs
- il en profite pour redire les fondamentaux qui font qu'on aime ce travail et ouvre quelques perspectives sur notre manière de travailler avec l'IA
]
]

---
class: section

<!-- .section-num[02] -->
<!-- .arc[] -->
.section-label[Section 02]
.section-bar[]

# Survey results

## https://machine-learning.pages.in2p3.fr/llm-survey-2026

---
class: middle, center

# Utilisation par les développeurs

![](../img/taches_developpement.svg)

---
class: middle, center

# Souhaits des développeurs

![](../img/accompagnement_par_statut.svg)
---
class: middle, center

# Indice de confiance

![](../img/niveau_confiance.svg)

---
class: middle, center

# Freins à l'usage de l'IAg

![](../img/freins_usage.svg)

---
class: middle, center

# Freins à l'usage en fonction de la fréquence d'utilisation

![](../img/freins_par_categorie.svg)

---
class: middle, center

# Jugement de productivité en fonction du statut

![](../img/score_par_statut.svg)

---
class: middle, center

# Jugement de productivité en fonction de la fréquence d'utilisation

![](../img/score_par_frequence.svg)

---
class: middle, center

# % de personnes prêtes à payer un abonnement en fonction du statut

![](../img/taux_paiement_par_statut.svg)

---
class: titre

# Quelques citations choisies

---
class: citation

.quote-mark["]

> Le plaisir du développement et de la réflexion disparait en promptant

.cite-bar[]

> I am afraid of becoming stupid, loosing my brain power

.cite-bar[]

> Financement de ces outils au meme titre que n'importe quel logiciel de CAO

.cite-bar[]

> Restreindre l'usage de l'IA au vu de ses implications sociales, éthiques et environnementales délétères.

---
class: citation

.quote-mark["]

> Un rappel clair et précis sur les effets sociaux de l'IA générative

.cite-bar[]

> Les personnels universitaires ne peuvent pas bénéficier des mêmes outils que les personnels CNRS dans une UMR, ce qui est préjudiciable.

.cite-bar[]

> Je n'utilise pas l'IA par fierté. J'estime être en capacité de produire de moi même ce qui pourrait concerner mes demandes. Bien évidement dans le cadre de mon travail. Cependant comme tout outil j'ai bel et bien conscience de son utilité et de son efficacité.

---
class: citation

.quote-mark["]

> Notre rôle de scientifique est d'être des dépositaires humains de la connaissance, si nos compétences et notre capacité à raisonner sont dépendantes d'outils qui nous privent de notre réflexion propre, je pense que notre intégrité est menacée. Il est clair que l'usage de LLM a des effets très négatifs sur une partie de la population, je pense que nous jouons un rôle d'exemplarité dans la prudence vis-a-vis de ces outils.

.cite-bar[]

> all shall use AI, from all scientific domains .. it is not a choice ...

---
class: section

.section-label[Section 03]
.section-bar[]

# Use cases

## Collected during the preparation meetings and the survey

---
class: deux-colonnes

# Use cases (non-exhaustive)

.colonnes[
.col-gauche[

## Docs/Postdocs

- **StackOverflow-like** assistance.
- **Brainstorming** (not limited to code).
- **Bibliographic research**.
]
.col-droite[

## Faculty/Researchers

- **Explain concepts** to students (reformulation).
- **Theme generation** for research topics.
- **Tentative use for grading** (mixed results).
- **Synthesize documents** (papers).
- **Grant writing** support.
- **Code writing assistance**.
- **Bibliographic research** (Consensus, ChatGPT, Sight).
]
]

---
class: deux-colonnes

# Use cases (non-exhaustive) 2/2

.colonnes[
.col-gauche[

## Administrative/HR

- **Translation**
- **Writing documents** (posts on social networks, APC website)
- **Rephrasing of message**
- **Abstract of thesis**
- **Explanation of complex concepts in scientific literature**
- **Interview preparation** (questions)
]

.col-droite[

## IT/Engineers

- **Code assistant** .big.red[vague].
- **Propose code architecture & software design**.
- **Recommend algorithms** (pros/cons).
- .big.red[Input from Walter (to be added).]
]
]

---
class: section

.section-num[03]
.section-label[Section 03]
.section-bar[]

# Major themes

---

# Environmental Impact

- **Rapid advancement** of generative AI (~month) vs. research timelines.
- **Knowledge gap**: State-of-the-art and capabilities evolve monthly.
- **Need for flexibility** and a **local experimentation platform**.

## Challenges

- **Hard to quantify** environmental impact of individual AI usage.
- **Rising RAM costs** + **hardware availability** issues.
- **Global resource reallocation** toward AI.

---

# Environmental Impact

## Local Observations

- Laboratory members are **reasonable** in usage.

.big.red[Add some metrics taken from poll ?]

## Best Practices

- **Rate-limited inference services** (e.g., Albert, ILaaS).
- **Shared computing resources**.
- **Recycling hardware** (Environmental charter - Art. 1.1, 3.2, 3.3)
- **Token allocation** (like compute time requests).
- **Carbon footprint tracking**:
  - Include in **annual lab reports**.
  - Add to **job submission metrics**.
- **Local Platform**: Small-scale testing (not for scaling demand). \newline
$\rightarrow$ Goal: **Experiment and design**.

---

# Transparency & Security

- **CNRS-Mistral contract (Emmy)**: Data remains **private** and **not used for training**.
- **UPC**: ILaaS runs on **French inference servers**.
- **Prohibited usage**: Any other tool must comply with **RGPD**.
  - **No sharing** of personal data (e.g., student lists) without **explicit consent**.
  - **Anonymize data** if necessary.

## Best Practices

- **Always apply critical thinking** to AI responses. \newline
  Models can fail unpredictably (e.g., after 10 exchanges, AI hallucination).
- **Expertise first**: Use AI as a **tool**, not a replacement.
  - **Improve/correct** AI outputs with human input.

## Academic Integrity

- **arXiv ban**: Risk of training on **non-peer-reviewed drafts** (data poisoning.
- **Declare AI use**: https://declare-ai.org.
- **Adapt policies** for interns/labs (e.g. https://mammouth.ai).

---

# Futures of the Profession

- **Adaptation**: AI is here to stay—**how will it change our work?**
  Dependecy develops with publication pressure (docs/postdocs)
- **Deskilling risk**: Preserve **expertise** and avoid tool dependency.
- **Reflection on change:**
  AI is here to stay, even if usage details evolve. We must consider how it will transform our lives, professions, and teaching. \newline 

**Example: [Denario](https://astropilot-ai.github.io/DenarioPaperPage)**

\textit{...serve as a scientific research assistant. Denario can perform many different tasks, such as generating ideas, checking the literature, developing research plans, writing and executing code, making plots, and drafting and reviewing a scientific paper. The system has a modular architecture, allowing it to handle specific tasks, such as generating an idea, or carrying out end-to-end scientific analysis...}

\vspace{0.5cm}

## Best Practices

- Avoid **dependency** ("IA fatigue").
- Re-evaluate the role of scientific publications as the primary means of assessing research. 

---

# Teaching

- **AI Charter**: Underway at UPC (aligned with other labs).
- **\href{https://www.ilaas.fr/}{\color{blue}\texttt{Projet ILaaS}}**: UPC will provide **inference resources** for ESR members (faculty + students).

**Example:**
At a conference (UPCité, ~2 years ago), an English teacher (middle/high school) instead of banning ChatGPT for translations, organized sessions to *analyze ChatGPT translations* and compare them with manual translations.\newline

--> There are no ready-made answers. Instead of saying *"this is wrong,"* we should ask: \newline "Things are different now—we need to adapt."

## Best Practices

- **Student responsability**: *"This is your training - you are learning"*
- **Adapt the way skills * knowledge are evaluated**

---
class: conclusion

# Merci pour votre attention

.gold-bar[]

## Time for discussions

.apc-logo[]

---
class: conclusion

# Backup

---

# L'IA générative au CNRS: Emmy

Mi-décembre 2025, lancement d’.blue[**Emmy**], l’IA générative .red[pour les agents du CNRS], dont les capacités sont

- traduction de textes en toutes langues ;
- synthèses de documents ;
- aide à la reformulation ;
- aide à la réflexion ;
- recherche sur le web ;
- reconnaissance de textes et d’images ;
- mode « raisonnement » : l’IA traite la question de l’utilisateur étape par étape afin de donner une réponse plus pertinente et plus complète ;
- collections de documents

Cet outil résulte d'un accord passé avec l’entreprise française Mistral AI pour 35 000 utilisateurs pour l'usage de leur offre .blue[**Le Chat Entreprise**].

.footnote[https://emmy.cnrs.fr/]

---
class: deux-colonnes

# Inférence as a Service dans l'ESR

.colonnes[
.gauche[

  ## ILaaS

  > Une fédération mutualisée visant une IA générative de confiance, robuste, éthique, et sobre

  Met à disposition une API d'inférence vers des modèles open-source.

  .terra[UPC va contribuer à cette fédération]
]

.droite[ .center[
<img src="../img/ilaas-service.png" width="100%" />

https://ilaas.fr
]]
]
---
class: deux-colonnes

# Inférence as a Service dans l'ESR

.colonnes[
.gauche[  .center[
## Albert – API de la DINUM
<img src="../img/albert-dinum.png" width="100%" />

https://albert.sites.beta.gouv.fr
]]

.droite[ .center[
## Claude Code + Albert API = `le-claude`

<img src="https://raw.githubusercontent.com/EiffL/le-claude/main/assets/le-claude.png" width="90%" />

https://github.com/EiffL/le-claude
]
```bash
  $ npx le-claude
```
]
]
---

- de .red[favoriser l'absence de transparence] (_acknowledgement_) propre au métier de la recherche, lors de l'utilisation d'IA générative dans le cadre du travail

--

- de .red[renforcer l'isolement des personnels] qui sont livrés à eux mêmes face à ces questions et qui profitent des outils "interdits" par d'autres intermédiaires : une collaboration scientifique (CERN), une offre pour l'éducation (Copilot) ou une souscription personnelle

--

.blue[**L'échelle de l'IN2P3**] est sans doute plus appropriée que celle du CNRS pour assurer l'inclusitivé de l'ensemble des personnels CNRS + Université, d'où la décision de faire ce sondage.

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

---
.hidden[toto]
### API institutionnelle sécurisée

demander accès à l'API Mistral au CNRS
possibilité d'utiliser la ferme locale pour de l'inférence (à travers les notebooks par exemple)  
mise en place de quotas utilisateurs  
mise en commun des moyens pour l'ESR  
impact environnemental de l'utilisation

--

### Risques sociétaux

perte de sens dans le monde de la recherche – [essai](https://davidbessis.substack.com/p/letter-to-a-phd-student)  
enseignement / formation des jeunes générations aux métiers  
nouveaux risques psychosociaux – [exemple](https://siddhantkhare.com/writing/ai-fatigue-is-real)
