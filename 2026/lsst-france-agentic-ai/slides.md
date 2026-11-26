class: cover

# Access to agentic AI in France:<br>where do we stand?

.grade[LSST France meeting — 10-minute presentation followed by an open discussion]

.meta[**Alexandre Boucaud** <a class="ico" href="https://aboucaud.github.io" target="_blank"><i class="fa-solid fa-globe"></i></a><a class="ico" href="https://github.com/aboucaud" target="_blank"><i class="fa-brands fa-github"></i></a><br>
AstroParticule et Cosmologie (APC), CNRS/IN2P3<br>
LSST France, November 26th 2026]

---

# Context: working with AI within DESC

- Ongoing discussions at the collaboration level, including the collaboration council
- New **DESC agentic AI group**, coordinated by François Lanusse and Corentin Ravoux
- Most people interested on daily work: **data analysis, software development, documentation, pipeline exploration**

The key question today: **what are we allowed to use, where, and on which data?**

---

# The IN2P3 generative AI working group

.kicker[Coordinated by Alexandre Boucaud (APC) and Thomas Vuillaume (LAPP)]

.cols[
.col[
#### Goal

Bring **clarity** to a fast-moving field, and show that generative AI and **frugality** can go together through targeted uses.

#### Current work

- Needs assessment
- Tools usable under CNRS constraints
- Open LLM solution (ILaaS)
- Cost-effectiveness of commercial offers
- Impact on software development jobs
]
.col[
#### Resources

- A guide in progress
- An agentic AI glossary
- A survey on usage

.note[We want to hear your needs, please **take the survey** (if not already)<br>→ **[sondage-ia-generative.in2p3.fr](https://sondage-ia-generative.in2p3.fr)**<br>.muted.small[Officially closed on 25 November, still open to latecomers until tomorrow, 27 November]]
]
]

---

# Today in France

.cols-even[
.col[
#### Emmy — for CNRS members

.muted.lien[[emmy.cnrs.fr](https://emmy.cnrs.fr)]

- The AI service reserved for **CNRS members**
- Mostly chatbot
- Recent upgrade:
  - **access to latest models** (Mistral Large 4)
  - new features incl. dedicated app
]
.col[
#### Albert API – for everyone

.muted.lien[[Albert API](https://ia.numerique.gouv.fr/outils-ia/albert-api/)]

- Service from the DINUM (Direction du Numérique)
- Few GPUs for many users => low quotas
- **Good enough for assisted programming**
- **Not for agentic work**
]
]

---

# Coming soon

.cols-even[
.col[
#### Emmy — for CNRS members

.muted.lien[[emmy.cnrs.fr](https://emmy.cnrs.fr)]

- **API access opening soon**
- Quotas still to be announced
- Only for **CNRS employees**
]
.col[
#### CC-IN2P3 local hosting

- CC-IN2P3 is setting up a **locally hosted** solution and will join the **ILaaS federation**
- Unlike Emmy: **open to all staff**, not only CNRS personnel
- Timeline: in preparation
]
]

---

# New CNRS exemption

.kicker[Clarification sent to lab directors by the director of CC-IN2P3]

- Since Nov 13, a **exemption** exists for **Anthropic (Claude)** and **OpenAI (Codex)** pro accounts
- **This does not legitimate the use of free or personal accounts !**
- On your laptop install **only the professional account** which by contract does not use data for training models.
- Any other commercial provider (beyond Anthropic or OpenAI) requires a **specific exemption request**

Key message: if you need one of these accounts, **request it from your lab director** (*directeur d'unité*)

---

# The Rubin constraint

Rubin **forbids using commercial AI** to process its data **until the data are public**

Consequence: the CNRS exemption on the use of commercial models is **not sufficient** to work on Rubin data

To use AI for Rubin analysis, we need: models **hosted by public research infrastructure**

---

# Available today: ALCF at CC-IN2P3

.kicker[Inference provided bt DOE infrastructure for DESC members]

Argonne Leadership Computing Facility is providing tokens access for DESC members through a NERSC account

- **Easy to setup** locally or on a computing center, see the setup guide below
- Works with **agentic workflows**
- Use of ALCF inference **has been authorized at CC-IN2P3** (still quite new so it might still be a bit painful)

.note[Project: **[LSSTDESC/agentic-research](https://github.com/LSSTDESC/agentic-research)** — setup guide and examples for DESC members]

---

# Open discussion

.kicker[Led by Marina Ricci - we want to hear the early career scientists]

<div class="piliers">
<div><span class="n">What has generative AI changed in your work?</span></div>
<div><span class="n">How do you feel about AI — your needs, your fears?</span></div>
<div><span class="n">What can we do to support you better?</span></div>
</div>
