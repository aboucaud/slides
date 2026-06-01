class: cover

<!-- .gold-stripe[] -->

# Agentic AI impact on our activities
.gold-bar[]
## Philippe Bacon - Paul Zakharov - Alexandre Boucaud
### Biennale APC - June 2026

.apc-logo[]

---
class: twocol

# A Highly Divisive Topic

As we will see through the survey results, this topic is particularly divisive within APC, IN2P3, and in French society more broadly.

.columns[
.left-col[

## Adopters

Among enthusiasts, the main requests are

- .navy[**data security**] and the deployment of .terra[**institutional tools**] with .navy[**responsible usage**]
- the need for .navy[**access to state-of-the-art tools**], .terra[**via API**], to stay competitive
- .navy[**trust**] remains .navy[**limited**] and the fear of .red[**intellectual impoverishment**] is present
]
.right-col[

## Critics

The main reasons for strong opposition

- an .red[energy and material sinkhole] with an .red[unsustainable] ecological impact
- a .red[source of job precarity] and skills erosion
- a .blue[**technological choice**] that requires .blue[**collective debate**]
]
]

.footnote[Manifeste d'objecteurs de conscience de l'IAg https://atecopol.hypotheses.org/13082]

---
class: fullbleed
background-image: url(../img/cianum_consultation2026.png)
background-size: cover

.hidden[toto]  
.hidden[tototototototototototo]
👉 [Read the consultation results][consult]

[consult]: https://www.conseil-ia-numerique.fr/le-numerique-est-une-affaire-collective-decouvrez-les-resultats-de-la-consultation-citoyenne-du

---
class: citation

.quote-mark["]

> L’objectif [de la mise en place de Emmy] est de permettre à chacun et chacune de mieux comprendre les opportunités que peut lui offrir l’IA dans son cadre professionnel, tout en identifiant aussi ses risques et ses limites. Et bien entendu, ces expérimentations doivent se faire avec modération, .navy[**l’empreinte carbone de l’utilisation de l’IA étant un enjeu majeur**].

.cite-bar[]
.source[Antoine Petit · Voeux 2026]

---

# Purpose of this Working Group

.stamp[Preliminary]

- Collect discussion on the use of **AI in research**.
- Identify key areas of focus through a **charter or guidelines**.

.highlight-box[**Disclaimer:**

Our role is not to attack nor defend AI, but to examine how to use it effectively in research.
]

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

# Agentic AI in 1 Slide

The 2024 language models became .blue[reasoning models] (chain-of-thought) in 2025, enabling step-by-step processing of larger inputs.

--

They were then equipped with tools to perform calculations or take actions (retrieve web page content, write and run a script, then read the result).

--

By giving them the ability to self-evaluate at the end of a task — and potentially retry differently — they became .blue[_agents_].

--

Agents then specialized in specific tasks (e.g. coding, reading PDFs, web search), giving rise to the concept of .blue[multi-agent] systems — an orchestrator agent managing specialized agents and aggregating results.

--

Until late 2025, these agents .red[made many errors or produced needlessly verbose code], leading to a perception of unreliable tools.

---

# The Model Context Protocol

In 2025, Anthropic developed a communication protocol enabling agents to easily use tools via their API, returning results directly in the correct format.

.center[
  <img src="https://raw.githubusercontent.com/lbourdois/blog/refs/heads/master/assets/images/Agents/image_35.png" width="70%">
]

.footnote[Credit https://lbourdois.github.io/blog/LLM_Agents/]

---

# Agent Customization

Agent customization has converged around a .red[`AGENT.md`] file that, for a given project, holds all the context the agent needs and serves as static memory.

--

To avoid repeatedly issuing the same instructions for specific tasks, .red[`SKILLS.md`] skill files emerged, allowing agents' capabilities to be extended and focused.

--

The miniaturization and democratization of .blue[vision tools] (OCR) also gives agents the ability to produce accurate descriptions of any image, adding the capacity to "see" everything a human sees on their screen (screenshots).

.gold-sep[]

---

# Early 2026: A Paradigm Shift

.left-column[

  A major leap in performance since December 2025, followed by a second one in February 2026.
  
  This was first seen with the release of Google Gemini 3 Pro, then Anthropic's Claude Opus 4.6 (large) and Claude Sonnet 4.6 (medium).

  .terra[**Model reliability and focus made an enormous leap forward**], making them far more useful for development tasks.

]

.right-column[
  .center[
  <img src="../img/claude-shipping-feb2026.jpeg" width="80%">
  ]
]

---

# Error Detection Benchmark

.center[
<img src="../img/bullshitbenchv2-march2026.png" width="80%">
]

.footnote[https://petergpt.github.io/bullshit-benchmark]

---

# Impact on Our Profession

.left-column[
Various recent studies (often biased, as they are commissioned by the tech giants themselves) report the percentage of typical job tasks that AI would be capable of handling.

It is clear that our profession is among those that will face strong AI-driven pressure, whether we like it or not.
]

.right-column[
  .center[
    <img src="../img/anthropic-labor-impact.png" width="85%" />
  ]
  
]

.footnote[https://www.anthropic.com/research/labor-market-impacts - March 2026]

---

# Clawdbot / OpenClaw (aside)

In parallel, starting in late November 2025, an independent agentic AI project emerged.

.center[<img src="https://imgs.search.brave.com/GLssrqcoxMIafoEOGcEtWCrEHznGg0GOfa-q73oq-oY/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9kb2lt/YWdlcy5ueWMzLmNk/bi5kaWdpdGFsb2Nl/YW5zcGFjZXMuY29t/LzAwOEFydGljbGVJ/bWFnZXMvV2hhdC1J/cy1PcGVuQ2xhdy9X/aGF0JTIwaXMlMjBP/cGVuQ2xhdy5wbmc" width="40%">]

The concept is to hand an entire machine over to a super-agent (the Jarvis equivalent from IronMan ©) to pilot autonomously: email inbox control, full file management, remote control via messaging apps (WhatsApp / Telegram). You start by giving it a soul .red[`SOUL.md`] and let it handle its tasks.  

.right[[And these agents can turn into bullies..](https://simonwillison.net/2026/Feb/12/an-ai-agent-published-a-hit-piece-on-me/)]

---

# The AI Scientist Concept (aside)

.left-column[
  .img-full[![Denario](../img/denario.png)]
]

.right-column[
  .img-full[![Denario steps](../img/denario-steps.png)]
]

---
class: twocol

# An Essay Emphasizing the Human Dimension of Our Work

.columns[
.gauche[

.img-full[![David Hogg - Why Astro](../img/hogg-why-astro.png)]

.center[https://arxiv.org/abs/2602.10181 - February 2026]
]

.droite[

## Pitch

- the arrival of powerful generative AI is here
- several options are available to us as scientists
- two are particularly bad: full acceptance ("all-AI") and outright rejection through policing of researchers
- he uses it to revisit the fundamentals of why we love this work and opens perspectives on how to work alongside AI
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

# Usage by Developers

![](../img/taches_developpement.svg)

---
class: middle, center

# What Developers Want

![](../img/accompagnement_par_statut.svg)
---
class: middle, center

# Confidence Index

![](../img/niveau_confiance.svg)

---
class: middle, center

# Barriers to Agentic AI Usage

![](../img/freins_usage.svg)

---
class: middle, center

# Barriers to Usage by Frequency of Use

![](../img/freins_par_categorie.svg)

---
class: middle, center

# Productivity Assessment by Status

![](../img/score_par_statut.svg)

---
class: middle, center

# Productivity Assessment by Frequency of Use

![](../img/score_par_frequence.svg)

---
class: middle, center

# % of People Willing to Pay a Subscription by Status

![](../img/taux_paiement_par_statut.svg)

---
class: cover

# Selected Quotes

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
class: twocol

# Use cases (non-exhaustive)

.columns[
.left-col[

## Docs/Postdocs

- **StackOverflow-like** assistance.
- **Brainstorming** (not limited to code).
- **Bibliographic research**.
]
.right-col[

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
class: twocol

# Use cases (non-exhaustive) 2/2

.columns[
.left-col[

## Administrative/HR

- **Translation**
- **Writing documents** (posts on social networks, APC website)
- **Rephrasing of message**
- **Abstract of thesis**
- **Explanation of complex concepts in scientific literature**
- **Interview preparation** (questions)
]

.right-col[

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
class: cover

# Merci pour votre attention

.gold-bar[]

## Time for discussions

.apc-logo[]

---
class: cover

# Backup

---

# Generative AI at CNRS: Emmy

Mid-December 2025, launch of .blue[**Emmy**], the generative AI .red[for CNRS staff], whose capabilities include

- text translation in all languages;
- document summarization;
- rephrasing assistance;
- reasoning support;
- web search;
- text and image recognition;
- "reasoning" mode: the AI processes the user's question step by step to provide a more relevant and comprehensive answer;
- document collections

This tool results from a deal between CNRS and the Mistral IA company for 35,000 licenses of their .blue[**Le Chat Entreprise**] offer.

.footnote[https://emmy.cnrs.fr/]

---
class: twocol

# Inference as a Service in French Academia

.columns[
.gauche[

  ## ILaaS

  > A shared federation aiming for trustworthy, robust, ethical, and frugal generative AI

  Provides an inference API for open-source models.

  .terra[UPC will contribute to this federation]
]

.droite[ .center[
.img-full[![ILaaS service](../img/ilaas-service.png)]

https://ilaas.fr
]]
]
---
class: twocol

# Inference as a Service in French Academia

.columns[
.gauche[  .center[
## Albert – DINUM API
.img-full[![Albert DINUM](../img/albert-dinum.png)]

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

- to .red[encourage a lack of transparency] (_acknowledgement_) inherent to research practice, when using generative AI at work

--

- to .red[reinforce the isolation of staff] who are left to navigate these questions alone and who access "banned" tools through other channels: a scientific collaboration (CERN), an educational subscription (Copilot), or a personal one

--

.blue[**The IN2P3 scale**] is arguably more appropriate than the CNRS scale to ensure inclusivity for all staff (CNRS + University), hence the decision to conduct this survey.

---

# Top Requests from the Survey

- A clear **regulatory framework**

- An inference service that is .green[**secure and sovereign**] .blue[**for all**] (CNRS + University staff)  
=> MistralAI API is highly requested, or a platform integrated with CC-IN2P3
  
- Prioritize .green[**open-source models**] and auditable models

- Ensure access to .green[**state-of-the-art models**] to gain buy-in and deter use of off-platform models

- Ecological impact being a widely shared concern, .red[**define quotas**] (project, lab) as with compute time + implement .green[**consumption tracking**]

---

# Top Requests from the Survey

- Encourage moderate usage for identified tasks, as very frequent AI use risks causing .red[global knowledge erosion]  
=> development is a craft and must .green[**be preserved**]

- .blue[Transparency about AI usage] is essential

- Be .red[mindful of licenses]  
=> see Philippe's presentation

---
class: middle

# Some Initiatives in French Academia

---

# AI Charters

Discussed in Reprises sessions

- **Legal framework**: GDPR, copyright, sector-specific compliance (e.g. health, education).
- **Ethics principles**: Transparency, accountability, fairness, privacy.
- **Best practices**:
  - usage limits
  - human validation of results
  - traceability of generated content
  - ecological impact

--

.right.medium[Must be educational!]

Examples:

- [AI charter portal in public administration](https://alliance.numerique.gouv.fr/cartographie/portail-des-chartes-ia-dans-ladministration/)
- [KairoiAI template used as a basis by LIP6](https://github.com/KairoiAI/Resources/blob/main/Template-ChatGPT-policy.md)
- [personal charter of a PhD student](https://kilianrouge.github.io/posts/2026/2_AI_Charter)

---

# Using Agentic AI Alongside a Course

Use of LLMs in an astrophysics course at Harvard.

Key points

- substantial upfront preparation work on prompts (everything is shared in the paper)
- RAG fine-tuning on a course document
- AI restricted to short answers .blue[**with course citations**] and no extended student–AI dialog
- guidelines on when use is authorized and when it is strongly discouraged  
  **=> very well received by students**

.footnote[Stubbs et al. 2026 - https://arxiv.org/abs/2602.04389]

---

# Ecole thématique Labobots / AISSAI

A second thematic semester of the [CNRS interdisciplinary center AISSAI][aissai] is being organized this year in partnership with IN2P3.
As part of this, a school / ANF called Labobots will be held in the fall (September 29 – October 2, 2026, Saint-Rémy-lès-Chevreuse) by the RI3-RAGLABS team.

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

Announcement coming soon..

.footnote[
  <img src="../img/aissai-logo.png" height='80px' alt="AISSAI"> 
]

[aissai]: https://aissai.cnrs.fr/en/

---

.hidden[toto]
### AI Charter
  
what uses do we want to prohibit at IN2P3?  
consequences in case of non-compliance?  
precedence between the CNRS charter and lab/institute charters?

--

### Staff Training

what is the purpose of these training sessions?  
at what scale (lab, in2p3, cnrs)?  
raise awareness of ethical issues and data sovereignty  

---
.hidden[toto]
### Secure Institutional API

request access to the Mistral API through CNRS
possibility of using the local cluster for inference (e.g. via notebooks)  
implement user quotas  
pool resources for French academia  
environmental impact of usage

--

### Societal Risks

loss of meaning in the research world – [essay](https://davidbessis.substack.com/p/letter-to-a-phd-student)  
teaching / training younger generations for the profession  
new psychosocial risks – [example](https://siddhantkhare.com/writing/ai-fatigue-is-real)
