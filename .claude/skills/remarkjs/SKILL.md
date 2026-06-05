---
name: remarkjs
description: Create and edit Remark.js HTML presentations in the slides repository. Use when creating a new presentation, adding or modifying slides, or when working with the APC theme. Covers the two-file layout (index.html + slides.md), Remark.js Markdown syntax, APC theme slide types, and screenshot workflow.
metadata:
    skill-author: Alexandre Boucaud
---

# Remark.js Presentations (APC Theme)

Presentations in this repository are two-file bundles: `index.html` (loads Remark.js and CSS) + `slides.md` (Markdown content). No build step.

## Directory convention

```
YEAR/short-kebab-case-title/
├── index.html
├── slides.md
└── img/          # optional, local images
```

Depth from the root is always `YEAR/name/`, so stylesheets and JS are referenced as `../../css/...` and `../../js/...`.

---

## New presentation: index.html template

Use this exact template. Adjust `<title>`, `<meta name="description">`, and `highlightStyle` (options: `zenburn`, `monokai`, `tomorrow-night`). Add the `apc-theme.css` line only for APC-branded presentations.

```html
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="utf-8">
  <title>TITLE HERE</title>
  <meta name="author" content="Alexandre Boucaud">
  <meta name="description" content="DESCRIPTION HERE">
  <link rel="stylesheet" type="text/css" href="../../css/slides.css">
  <link rel="stylesheet" type="text/css" href="../../css/apc-theme.css">
</head>
<body>
  <style TYPE="text/css">
    code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
  </style>
  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre']
    }
    });
    MathJax.Hub.Queue(function() {
      var all = MathJax.Hub.getAllJax(), i;
      for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
      }
    });
  </script>
  <script type="text/javascript" async
    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
  </script>
  <script type="text/javascript" src="../../js/remark-latest.min.js"></script>
  <script type="text/javascript">
    var slideshow = remark.create({
      sourceUrl: 'slides.md',
      ratio: '16:9',
      navigation: {
        scroll: false,
        touch: true,
        click: false,
      },
      slideNumberFormat: '%current%/%total%',
      countIncrementalSlides: false,
      highlightStyle: 'zenburn',
      highlightLines: true,
      highlightSpans: true,
    });
  </script>
</body>
</html>
```

---

## New presentation: slides.md template (APC theme)

```markdown
class: cover

# Presentation Title
## Subtitle or context
### Author · Institution · Date

.gold-bar[]
.apc-logo[]

---
class: outline

# Outline

1. First part
2. Second part
3. Third part

---
class: section

.section-num[01]
.section-label[Part one]
.section-bar[]
# Section Title
.arc[]

---

# Slide Title

Content goes here.

- Bullet one
- Bullet two

---
class: cover

# Thank you

.gold-bar[]
.apc-logo[]
```

---

## Remark.js Markdown syntax

### Slide separator
```
---
```
Every `---` on its own line starts a new slide.

### Slide class
```markdown
---
class: cover
```
Place immediately after `---`, before any content.

### Inline class notation — `.classname[content]`
Wraps content in a `<span class="classname">`. Used for colours, sizes, and theme components:
```markdown
This is .red[important] and this is .navy[**bold navy**].
.footnote[Credit: ESA / Hubble]
.gold-bar[]
.apc-logo[]
```
When the brackets are empty (`.gold-bar[]`, `.apc-logo[]`, `.arc[]`), it renders a CSS-generated decoration.

### Background image (fullbleed slides)
```markdown
---
class: fullbleed
background-image: url(img/photo.jpg)
background-size: cover
---
```

### Speaker notes
```markdown
# Slide Title

Visible content.

???
These notes only appear in presenter mode (press `p`).
```

### Slide properties (front matter per slide)
```markdown
---
name: my-slide-id
count: false

# Hidden or unnumbered slide
```

`count: false` excludes the slide from the total count.  
`name:` lets you link to the slide: `[go there](#my-slide-id)`.

### Incremental reveal
```markdown
# Incremental

- First point

--

- Second point (appears on next key press)
```

### Code highlighting
Fence with language tag. Highlight lines with `*` prefix, inline spans with `{{ }}`:

````markdown
```python
def foo():
*   return 42   # highlighted line
```
````

---

## APC theme: slide types

Full CSS reference is in `CLAUDE.md`. Quick reference for `class:` values:

| class | Background | Use for |
|-------|-----------|---------|
| *(default)* | rose-light | Regular content slide with navy header |
| `cover` | navy | Title slide, conclusion slide |
| `section` | navy | Section interlude |
| `outline` | navy | Table of contents |
| `citation` | rose-light | Pull quote |
| `twocol` | rose-light + header | Two-column layout via `.columns[.left-col[…].right-col[…]]` |
| `blanc` | white | Content slide without rose sidebar |
| `fullbleed` | black/image | Full-bleed photo slide, no header |

See CLAUDE.md for complete component list (`.stamp[]`, `.highlight-box`, `.gold-sep`, etc.) and colour utility classes (`.navy`, `.gold`, `.terra`, `.rose`, `.muted`).

---

## Auto-footer on content slides

To automatically add `.apc-footer` to every default, twocol, blanc, and citation slide (without touching `slides.md`), add this script block to `index.html` immediately after `remark.create({…});`:

```html
<script>
  // Auto-inject .apc-footer on default/twocol/blanc/citation slides.
  // Remove this block to disable; manually placed .apc-footer[] are respected.
  (function () {
    document.querySelectorAll(
      '.remark-slide-content:not(.cover):not(.section):not(.outline):not(.fullbleed)'
    ).forEach(function (slide) {
      if (!slide.querySelector('.apc-footer')) {
        var el = document.createElement('div');
        el.className = 'apc-footer';
        slide.appendChild(el);
      }
    });
  })();
</script>
```

- Cover, section, outline, and fullbleed slides are excluded automatically.
- Slides that already have `.apc-footer[]` in the Markdown are not doubled.
- The citation slide CSS includes `padding-bottom: var(--footer-h)` to prevent overlap.

---

## Preview (local server)

MathJax and `sourceUrl` require HTTP, not `file://`. Serve from the repo root:

```bash
python3 -m http.server 8000
# open http://localhost:8000/YEAR/presentation-name/
```

Or from the presentation directory:
```bash
python3 -m http.server 8000
# open http://localhost:8000/
```

Press `p` for presenter mode, `c` to clone the window (for dual-screen presenting), `?` for all shortcuts.

---

## Screenshots / export to PDF

Remark.js uses `position: fixed` for the slide container, which breaks Chrome's PDF print rendering (content gets mispositioned/clipped). The reliable workflow is:

1. Capture per-slide PNGs with decktape
2. Assemble them into a PDF with `img2pdf`

**Step 1 — capture screenshots** (run in background; 52 slides ≈ 2 min):

```bash
mkdir -p /tmp/slides-screenshots
npx decktape remark --size 1920x1080 \
  --screenshots --screenshots-directory /tmp/slides-screenshots --screenshots-size 1920x1080 \
  http://localhost:8000/YEAR/presentation-name/ slides-export.pdf
```

**Step 2 — assemble PDF** (requires `img2pdf`; install once via `uv`):

```bash
uv venv /tmp/slides-venv --clear --quiet
uv pip install img2pdf --python /tmp/slides-venv/bin/python --quiet
```

```python
# Run with: /tmp/slides-venv/bin/python script.py
import glob, re, img2pdf

files = sorted(
    glob.glob('/tmp/slides-screenshots/slides-export_*_1920x1080.png'),
    key=lambda x: int(re.search(r'_(\d+)_', x).group(1))
)

# 1920×1080 px at 96 DPI → 1440×810 pt (16:9)
layout = img2pdf.get_layout_fun((img2pdf.in_to_pt(20), img2pdf.in_to_pt(11.25)))

with open('YEAR/presentation-name/output.pdf', 'wb') as f:
    f.write(img2pdf.convert(files, layout_fun=layout))
```

Review a sample screenshot with the Read tool to catch layout issues before assembling.

**Requires the local server to be running. Do NOT use `xargs convert … output.pdf` — xargs appends inputs after the PDF path, inverting input/output.**
