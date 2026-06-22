class: cover

.masthead-rule[]
# LightCone Research
## Theme Component Showcase
### Alexandre Boucaud · APC · JDev 2026

.lc-logo[]

---
class: outline

# Outline

1. Typography
2. Layout & Cards
3. Terminal & YAML
4. Workflow components
5. Process & Navigation
6. Stats, Animations & Misc

---
class: section

.section-label[01]
# Typography

---

# Typography — Eyebrow, Headline, Body, Takeaway

<p class="eyebrow">LightCone Research · 2026</p>
<p class="headline">Grounding AI decisions in scientific evidence</p>
<p class="body-text">LightCone is an AI-native research environment that records, justifies,
and reproduces scientific work at any scale — from a single notebook to a multi-year collaboration.</p>
<p class="takeaway">Every claim is traceable. Every decision is grounded.</p>

---

# Typography — Text colours

.left-column[
**Palette**

.text-primary[.text-primary — Blue Ink]

.text-secondary[.text-secondary — Slate Blue]

.text-accent[.text-accent — Pine Green]

.text-warm[.text-warm — Antique Gold]

.text-highlight[.text-highlight — Wax Red]

.text-muted[.text-muted — Graphite]
]
.right-column[
.primary[.primary]

.warm[.warm]

.highlight[.highlight]

.secondary[.secondary]

.muted[.muted — slightly smaller]

**Size helpers**

.small[.small — 0.82em] / .medium[.medium] / .large[.large] / .huge[.huge — 1.5em]
]
.reset-column[]

---

# Typography — Pills & Tags

<br>

.pill.pill-primary[Primary]
.pill.pill-secondary[Secondary]
.pill.pill-accent[Accent]
.pill.pill-warm[Warm]
.pill.pill-highlight[Highlight]
.pill.pill-muted[Muted]

<br><br>

**Section label chip** (used on section slides and inside components):

<span class="section-label">LightCone Research</span>
&nbsp;
<span class="section-label">v2.1.0</span>
&nbsp;
<span class="section-label">Alpha</span>

<br>

**Gold separator**

.gold-sep[]

Some content below the separator.

---
class: section

.section-label[02]
# Layout & Cards

---

# Layout — Two and three columns

.cols[
.card[
**Left column**

Use `.cols` for a 50/50 grid.
Each child becomes one column.

.text-muted[Works well for comparing two things side by side.]
]
.card[
**Right column**

Pair with `.card`, `.card-glow`,
or any block element.

.text-muted[No floats needed — CSS grid under the hood.]
]
]

<br>

.cols-3[
.highlight-box[**Step 1** — Record]
.highlight-box[**Step 2** — Decide]
.highlight-box[**Step 3** — Run]
]

---

# Cards — .card, .card-glow, .highlight-box

.cols[
<div class="card">
<p class="eyebrow">Standard card</p>
<p class="headline">Zero dependencies</p>
<p class="body-text">Use <code>.card</code> for panels that need to stand off the slide background with a soft ambient shadow.</p>
</div>

<div class="card-glow">
<p class="eyebrow">Glow variant</p>
<p class="headline">Slightly more emphasis</p>
<p class="body-text">Use <code>.card-glow</code> when the card is the primary focus of the slide.</p>
</div>
]

<br>

<div class="highlight-box">
<strong>Highlight box</strong> — left navy border, tinted background. Great for key takeaways or warnings inline with text content.
</div>

---

# Cards — with left-border accents

.cols[
<div class="card" style="border-left: 3px solid var(--lc-warm);">
<p class="eyebrow">Decision</p>
<p class="headline">Use transformer-based embeddings</p>
<p class="body-text">Chosen over bag-of-words after benchmarking on the internal corpus.</p>
</div>

<div class="card" style="border-left: 3px solid var(--lc-accent);">
<p class="eyebrow">Result</p>
<p class="headline">+14 pp retrieval accuracy</p>
<p class="body-text">Measured on 200 held-out queries against the Rubin DESC knowledge base.</p>
</div>
]

<br>

.cols[
<div class="card" style="border-top: 3px solid var(--lc-primary);">
<p class="eyebrow">Note — border-top variant</p>
<p class="body-text">Use <code>style="border-top: 3px solid var(--lc-*)"</code> for a top-accent card.</p>
</div>

<div class="card" style="border-left: 3px solid var(--lc-highlight);">
<p class="eyebrow">Warning</p>
<p class="body-text">Wax-red border signals caution or a known limitation.</p>
</div>
]

---

# Layout — Walkthrough panel

<div class="walkthrough-panel">
  <div class="walkthrough-copy">
    <p class="eyebrow">Step 01 · Record</p>
    <p class="headline">Start the scientific record</p>
    <p class="body-text">Every experiment begins by declaring its context — inputs, decisions, and environment — before a single line of code runs.</p>
    <p class="takeaway">Reproducibility starts at inception, not at publication.</p>
  </div>
  <div class="walkthrough-visual">
    <div class="vis-card" style="max-width: 22rem;">
      <div class="vis-card__chrome">
        <span class="vis-dot"></span>
        <span class="vis-dot"></span>
        <span class="vis-dot"></span>
        <span class="vis-card__title">terminal</span>
        <span class="vis-card__tag">ASTRA</span>
      </div>
      <div class="vis-card__body">
        <div class="vis-terminal">
          <div class="vis-terminal__line">
            <span class="vis-terminal__prompt">$</span>lc record start
          </div>
          <div class="vis-terminal__line--out vis-terminal__line">✓ Record #a3f8 opened</div>
          <div class="vis-terminal__line--out vis-terminal__line">✓ Environment snapshotted</div>
        </div>
      </div>
    </div>
  </div>
</div>

---
class: section

.section-label[03]
# Terminal & YAML

---

# Vis-card — Chrome card with macOS header bar

<div style="display: flex; gap: 1.5em; align-items: flex-start;">
  <div class="vis-card" style="max-width: 26rem;">
    <div class="vis-card__chrome">
      <span class="vis-dot"></span>
      <span class="vis-dot"></span>
      <span class="vis-dot"></span>
      <span class="vis-card__title">terminal</span>
      <span class="vis-card__tag">ASTRA</span>
    </div>
    <div class="vis-card__body">
      <div class="vis-terminal">
        <div class="vis-terminal__line">
          <span class="vis-terminal__prompt">$</span>lc run pipeline.yaml
        </div>
        <div class="vis-terminal__line--out vis-terminal__line">✓  Loaded 3 scripts</div>
        <div class="vis-terminal__line--out vis-terminal__line">✓  preprocess  0.8 s</div>
        <div class="vis-terminal__line--out vis-terminal__line">✓  train       42.3 s</div>
        <div class="vis-terminal__line--out vis-terminal__line">✓  evaluate    2.1 s</div>
      </div>
    </div>
  </div>

  <div class="vis-card" style="max-width: 22rem;">
    <div class="vis-card__chrome">
      <span class="vis-dot"></span>
      <span class="vis-dot"></span>
      <span class="vis-dot"></span>
      <span class="vis-card__filename">pipeline.yaml</span>
      <span class="vis-card__tag--accent vis-card__tag">DECISION</span>
    </div>
    <div class="vis-card__body">
      <div class="vis-yaml">
        <div class="vis-yaml__row"><span class="vis-yaml__key">model:</span> transformer</div>
        <div class="vis-yaml__row"><span class="vis-yaml__key">dim:</span> 512</div>
        <div class="vis-yaml__row"><span class="vis-yaml__key">epochs:</span> 40</div>
        <div class="vis-yaml__row"><span class="vis-yaml__key">lr:</span> 3e-4</div>
        <div class="vis-yaml__row"><span class="vis-yaml__key">batch:</span> 128</div>
        <div class="vis-yaml__row"><span class="vis-yaml__key">seed:</span> 42</div>
      </div>
    </div>
  </div>
</div>

---

# YAML document viewer

<div class="vis-card" style="max-width: 38rem; margin: 0 auto;">
  <div class="vis-card__chrome">
    <span class="vis-dot"></span>
    <span class="vis-dot"></span>
    <span class="vis-dot"></span>
    <span class="vis-card__filename">experiment.yaml</span>
    <span class="vis-card__tag">CONFIG</span>
  </div>
  <div class="vis-yaml-doc">
    <div class="yaml-line yaml-line--section">experiment:</div>
    <div class="yaml-line">  <span class="yaml-key">name:</span>  <span class="yaml-val">galaxy-morphology-v3</span></div>
    <div class="yaml-line">  <span class="yaml-key">record:</span> <span class="yaml-val">a3f8c91</span></div>
    <div class="yaml-line yaml-line--empty"></div>
    <div class="yaml-line yaml-line--section">model:</div>
    <div class="yaml-line">  <span class="yaml-key">arch:</span>   <span class="yaml-val">efficientnet-b3</span></div>
    <div class="yaml-line">  <span class="yaml-key">pretrained:</span> <span class="yaml-val">true</span></div>
    <div class="yaml-line yaml-line--empty"></div>
    <div class="yaml-line yaml-line--section">training:</div>
    <div class="yaml-line">  <span class="yaml-key">epochs:</span> <span class="yaml-val">80</span></div>
    <div class="yaml-line">  <span class="yaml-key">lr:</span>     <span class="yaml-val">1e-3</span></div>
    <div class="yaml-line">  <span class="yaml-key">scheduler:</span> <span class="yaml-val">cosine</span></div>
  </div>
</div>

---
class: section

.section-label[04]
# Workflow components

---

# Script pipeline

<div style="display: flex; gap: 2em; align-items: flex-start; justify-content: center;">
  <div class="vis-scripts" style="width: 14rem;">
    <span class="vis-scripts__cli"><span class="vis-scripts__prompt">$</span> lc run</span>
    <div class="vis-script"><span class="vis-script__name">preprocess</span></div>
    <div class="vis-script"><span class="vis-script__name">embed</span></div>
    <div class="vis-script vis-script--anchor">
      <span class="vis-script__name">train</span>
      <span class="vis-script__dot"></span>
    </div>
    <div class="vis-script"><span class="vis-script__name">evaluate</span></div>
  </div>

  <div style="max-width: 28rem;">
    <p class="eyebrow">Script pipeline</p>
    <p class="headline">Sequential execution with provenance</p>
    <p class="body-text">Each script is a versioned, named step. Arrows are auto-generated by <code>+ .vis-script</code> sibling selectors — no extra markup needed.</p>
    <br>
    <p class="body-text">Use <code>.vis-script--anchor</code> with <code>.vis-script__dot</code> to mark a step that receives an external input connection.</p>
  </div>
</div>

---

# Decision card — options + evidence

<div class="vis-decision" style="max-width: 26rem; margin: 0 auto;">
  <div class="vis-card vis-decision__card">
    <div class="vis-card__chrome">
      <span class="vis-dot"></span><span class="vis-dot"></span><span class="vis-dot"></span>
      <span class="vis-decision__chrome-title">Embedding strategy</span>
      <span class="vis-card__tag">DECISION</span>
    </div>
    <div class="vis-card__body">
      <ul class="vis-decision__options">
        <li class="vis-decision__option"><span class="vis-radio"></span>Bag of words</li>
        <li class="vis-decision__option is-selected"><span class="vis-radio"></span>Transformer embeddings</li>
        <li class="vis-decision__option"><span class="vis-radio"></span>Sparse TF-IDF</li>
      </ul>
    </div>
  </div>
  <div class="vis-decision__bridge">
    <span class="vis-decision__bridge-line"></span>
    <span class="vis-decision__bridge-label">supported by</span>
    <span class="vis-decision__bridge-line vis-decision__bridge-line--arrow"></span>
  </div>
  <div class="vis-card vis-decision__evidence">
    <div class="vis-card__chrome">
      <span class="vis-dot"></span><span class="vis-dot"></span><span class="vis-dot"></span>
      <span class="vis-card__filename">Devlin et al., 2019</span>
      <span class="vis-card__tag--accent vis-card__tag">EVIDENCE</span>
    </div>
    <div class="vis-card__body">
      <p class="vis-decision__quote">"BERT representations substantially outperform
      feature-based approaches on semantic similarity tasks."</p>
      <p class="vis-decision__cite">NAACL 2019 · Best Paper</p>
      <p class="vis-decision__doi">doi:10.18653/v1/N19-1423</p>
      <div class="vis-decision__check">
        <span class="vis-check">✓</span> Verified on internal benchmark
      </div>
    </div>
  </div>
</div>

---

# Inspector rows — key/value metadata panel

<div style="display: flex; gap: 2em; align-items: flex-start;">
  <div class="vis-card" style="max-width: 32rem; flex: 1;">
    <div class="vis-card__chrome">
      <span class="vis-dot"></span><span class="vis-dot"></span><span class="vis-dot"></span>
      <span class="vis-card__filename">record a3f8c91</span>
      <span class="vis-card__tag">INSPECT</span>
    </div>
    <div class="vis-card__body">
      <div class="vis-inspect__rows">
        <div class="vis-inspect__row">
          <span class="vis-inspect__label">Record ID</span>
          <code>a3f8c91d</code>
        </div>
        <div class="vis-inspect__row">
          <span class="vis-inspect__label">Author</span>
          <span>Alexandre Boucaud</span>
        </div>
        <div class="vis-inspect__row">
          <span class="vis-inspect__label">Dataset</span>
          <code>HSC-S21-morphology</code>
        </div>
        <div class="vis-inspect__row">
          <span class="vis-inspect__label">Accuracy</span>
          <span class="text-accent">91.4 %</span>
        </div>
        <div class="vis-inspect__row">
          <span class="vis-inspect__label">Duration</span>
          <span>45.2 s</span>
        </div>
        <div class="vis-inspect__row">
          <span class="vis-inspect__label">Status</span>
          <span><span class="vis-check" style="width:0.8rem;height:0.8rem;font-size:0.6rem;">✓</span> Reproducible</span>
        </div>
      </div>
    </div>
  </div>

  <div style="flex: 1;">
    <p class="eyebrow">Inspector rows</p>
    <p class="headline">Structured metadata at a glance</p>
    <p class="body-text">Use <code>.vis-inspect__rows</code> inside any <code>.vis-card__body</code> to present key/value pairs in a clean grid layout.</p>
    <br>
    <p class="body-text">Labels are auto-uppercased via CSS. Values accept inline <code>code</code>, colour utilities, or any inline element.</p>
  </div>
</div>

---
class: section

.section-label[05]
# Process & Navigation

---

# Step tabs — multi-step indicator strip

<div class="step-tabs">
  <div class="step-tab is-active">
    <span class="step-tab__num">01</span>Record
  </div>
  <div class="step-tab">
    <span class="step-tab__num">02</span>Decide
  </div>
  <div class="step-tab">
    <span class="step-tab__num">03</span>Run
  </div>
  <div class="step-tab">
    <span class="step-tab__num">04</span>Inspect
  </div>
</div>

<br>

Add `.is-active` to the tab that corresponds to the current slide's step.
Duplicate the strip across consecutive slides and shift the active class to
show progress through a multi-step sequence.

<br>

<div class="step-tabs">
  <div class="step-tab"><span class="step-tab__num">01</span>Record</div>
  <div class="step-tab"><span class="step-tab__num">02</span>Decide</div>
  <div class="step-tab is-active"><span class="step-tab__num">03</span>Run</div>
  <div class="step-tab"><span class="step-tab__num">04</span>Inspect</div>
</div>

---

# Step row — horizontal process flow

<br>

<div class="step-row">
  <div class="step">
    <div class="step__icon" style="background: rgba(78,90,112,0.1);">📋</div>
    <p class="step__label">Declare<br>context</p>
  </div>
  <span class="step__arrow">→</span>
  <div class="step">
    <div class="step__icon" style="background: rgba(166,124,60,0.1);">🧠</div>
    <p class="step__label">Ground<br>decisions</p>
  </div>
  <span class="step__arrow">→</span>
  <div class="step">
    <div class="step__icon" style="background: rgba(24,64,28,0.1);">▶</div>
    <p class="step__label">Run<br>scripts</p>
  </div>
  <span class="step__arrow">→</span>
  <div class="step">
    <div class="step__icon" style="background: rgba(66,107,120,0.1);">🔍</div>
    <p class="step__label">Inspect<br>chain</p>
  </div>
</div>

<br>

.cols[
<div class="card" style="font-size:0.85em;">
Use <code>.step-row</code> with <code>.step__icon</code> (circular container) and <code>.step__label</code>. Set the icon background colour with an inline <code>style</code> using any <code>--lc-*</code> variable at low opacity.
</div>
<div class="card" style="font-size:0.85em;">
Pair with <strong>step tabs</strong> on consecutive slides to build an animated walkthrough: step tabs show which step is active, the step row gives the overall picture.
</div>
]

---
class: section

.section-label[06]
# Stats, Animations & Misc

---

# Stats / KPI blocks

<br>

.cols-3[
<div class="stat">
  <div class="stat-number">91.4%</div>
  <div class="stat-label">Classification accuracy</div>
</div>
<div class="stat">
  <div class="stat-number">3.2×</div>
  <div class="stat-label">Faster than baseline</div>
</div>
<div class="stat">
  <div class="stat-number">10 k</div>
  <div class="stat-label">Labelled galaxies</div>
</div>
]

<br>

.cols[
<div class="card" style="border-top: 3px solid var(--lc-warm);">
  <div class="stat">
    <div class="stat-number" style="font-size:1.8em;">47 ms</div>
    <div class="stat-label">Median inference time</div>
  </div>
</div>
<div class="card" style="border-top: 3px solid var(--lc-accent);">
  <div class="stat">
    <div class="stat-number" style="font-size:1.8em;">100%</div>
    <div class="stat-label">Runs reproduced end-to-end</div>
  </div>
</div>
]

---

# Animations — entrance keyframes

Elements below each use `.anim-rise` / `.anim-fade` / `.anim-pop` with delay modifiers.

<div class="cols" style="margin-top: 1em;">
<div>
<div class="card anim-rise anim-delay-1" style="margin-bottom:0.8em;">
  <p class="eyebrow">anim-rise + delay-1</p>
  <p class="headline">Slides up and fades in</p>
</div>
<div class="card anim-rise anim-delay-2" style="margin-bottom:0.8em;">
  <p class="eyebrow">anim-rise + delay-2</p>
  <p class="headline">Offset by 0.4 s</p>
</div>
<div class="card anim-rise anim-delay-3">
  <p class="eyebrow">anim-rise + delay-3</p>
  <p class="headline">Offset by 0.6 s</p>
</div>
</div>
<div>
<div style="text-align:center; margin-top:1em;">
  <div class="pill pill-warm anim-fade anim-delay-1" style="margin:0.3em;">anim-fade</div>
  <div class="pill pill-primary anim-fade anim-delay-2" style="margin:0.3em;">delay-2</div>
  <div class="pill pill-accent anim-fade anim-delay-3" style="margin:0.3em;">delay-3</div>
</div>
<br>
<div style="display:flex; gap:1em; justify-content:center; margin-top:1.2em;">
  <div class="vis-check anim-pop anim-delay-2" style="width:2rem;height:2rem;font-size:1rem;">✓</div>
  <div class="vis-check anim-pop anim-delay-3" style="width:2rem;height:2rem;font-size:1rem;">✓</div>
  <div class="vis-check anim-pop anim-delay-4" style="width:2rem;height:2rem;font-size:1rem;">✓</div>
</div>
<p class="footnote">anim-pop with delay-2/3/4</p>
</div>
</div>

---

# Misc — Stamp, CTA buttons, Footnote

<br>

.cols[
<div>
<div class="stamp">PRELIMINARY</div>
<br>
<p class="body-text center">Use <code>.stamp</code> for a rotated watermark label. Place it anywhere on the slide — it centres itself in its container.</p>
</div>
<div>
<p class="eyebrow" style="margin-bottom:1em;">Call-to-action buttons</p>
<p>
  <span class="cta">Primary CTA</span>
  &nbsp;
  <span class="cta-outline">Outline CTA</span>
</p>
<br>
<p class="body-text">Use <code>.cta</code> for filled and <code>.cta-outline</code> for outlined buttons — both use Quattrocento, matching the heading style.</p>
</div>
]

.footnote[.footnote — anchored bottom-right, 0.6em, muted weight]

---
class: blanc

# Slide type — .blanc (white background)

This slide uses `class: blanc`. The background switches to pure white (`--lc-surface-lowest`) while the navy header, body text, and brand mark remain unchanged.

<br>

.cols[
<div class="card">
Good for slides where a white background helps images or diagrams read more cleanly without the warm parchment tint.
</div>
<div class="highlight-box">
All other styling (typography, brand mark, cards) continues to work identically on `.blanc` slides.
</div>
]

---
class: cover

.masthead-rule[]
# Thank you

.lc-logo[]
