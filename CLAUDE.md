# Slides — CSS reference

**Always invoke the `remarkjs` skill** (via the Skill tool) before creating or editing presentations in this repository.

This repository uses [Remark.js](https://remarkjs.com/) to generate HTML presentations from Markdown files.
Each presentation loads two stylesheets:

```html
<link rel="stylesheet" href="../../css/slides.css">    <!-- shared base -->
<link rel="stylesheet" href="../../css/apc-theme.css"> <!-- APC theme -->
```

---

## slides.css — shared base

Stylesheet shared by all presentations in the repository.

### Typography

| Selector | Size | Notes |
|----------|------|-------|
| `h2` | 140% | line-height 150% |
| `h3` | 120% | line-height 140% |
| `ul` | 120% | line-height 140% |
| `ol li` | 100% | line-height 140% |
| `li` | 90% | line-height 140% |
| `li > p` | 1em | Neutralises the `li(90%) × p(120%)` compounding in loose Remark lists |
| `p` | 120% | line-height 140% |

Font: **Ubuntu** (400 / 300 / 100) + **Ubuntu Mono** for code.

### Layout classes

| Class | Behaviour |
|-------|-----------|
| `.left-column` / `.right-column` | Two 49% floating columns |
| `.reset-column` | Clears floats |
| `.middle-left` | 50% float left, vertical-align middle |
| `.middlebelowheader` | 500 px table-cell, content centred vertically |
| `.widespace` | `h2` with line-height 200% |

### Text classes

| Class | Behaviour |
|-------|-----------|
| `.small` | 90% |
| `.medium` | 120% |
| `.big` | 150% |
| `.huge` | 180% |
| `.grey` / `.dark-grey` / `.red` / `.blue` / `.green` | Text colours |
| `.credits` | Italic 70%, float bottom-right |
| `.footnote` | 80%, light weight, anchored bottom-right |
| `.hidden` | `visibility: hidden` |

### Code classes

| Class | Behaviour |
|-------|-----------|
| `.big .remark-code` | Code at 200% |
| `.medium .remark-code` | Code at 120% |
| `.mmedium .remark-code` | Code at 99% |

Code font: Ubuntu Mono. Line highlight: `remark-code-line-highlighted` (cyan background).

### Image classes

| Class | Behaviour |
|-------|-----------|
| `.singleimg img` | Centred, max 90% width, max 600 px height |
| `.circle-image` | Round image 170 × 170 px |
| `.bunchoflogos img` | Logos in a row, max 100 px height |
| `.bottomlogo` | Logo anchored bottom-left |

---

## apc-theme.css — APC theme

Official Astroparticule et Cosmologie theme. **Requires** `slides.css` loaded first.

### Colour palette

| Variable | Value | Usage |
|----------|-------|-------|
| `--apc-navy` | `#013975` | Main background for dark slides |
| `--apc-navy-mid` | `#012D61` | Watermark, overlays |
| `--apc-navy-light` | `#E6EDF6` | Secondary text on navy background |
| `--apc-terracotta` | `#B5563D` | Strong accents |
| `--apc-terracotta-mid` | `#C4674F` | Common accents, decorative bars |
| `--apc-rose` | `#B89D98` | Left sidebar on content slides |
| `--apc-rose-light` | `#F5EEEC` | Content slide background (default) |
| `--apc-gold` | `#E8A838` | Gold accent, numbers, header bar |
| `--apc-white` | `#FFFFFF` | Text on dark backgrounds |
| `--apc-gray-bdr` | `#E8DFDC` | Borders, separators |
| `--apc-text-dark` | `#2D3848` | Main text |
| `--apc-text-mid` | `#556070` | Secondary text |
| `--apc-text-light` | `#8B95A8` | Subtle text, page numbers |

### Structural dimensions

| Variable | Value | Usage |
|----------|-------|-------|
| `--header-h` | `5.3rem` | Header bar height |
| `--footer-h` | `1.5rem` | Footer height |
| `--gold-accent` | `1.6rem` | Gold stripe width (right side of header) |
| `--pad-x` | `2.5rem` | Horizontal content padding |
| `--pad-y` | `1.2rem` | Vertical content padding (below header) |

### Slide types

Declare the class right after `---` in Markdown:

```markdown
---
class: NAME
```

#### (default) — Content slide

Background `--apc-rose-light`. Navy header bar at the top with a gold stripe on the right. `h1` displayed in the bar. Rose left sidebar.

---

#### `class: cover`

Cover slide (title or conclusion). Navy background, flex centred layout. Large `h1` (3.12 rem), terracotta `h2`, navy-light `h3`. Logo bottom-right via `.logo[]`. Thin gold stripe at top via `.gold-stripe[]`. Gold horizontal bar via `.gold-bar[]`. Page number hidden.

```markdown
---
class: cover

# Main title
## Subtitle
### Author · Institution · Date

.gold-bar[]
.logo[![](logo_apc_white.png)]
```

---

#### `class: section`

Section interlude slide. Navy background. Large section number inside a decorative arc (top-right corner) via `.section-num[]`. Uppercase label via `.section-label[]`. Terracotta bar via `.section-bar[]`. Decorative arc via `.arc[]`.

```markdown
---
class: section

.section-num[01]
.section-label[Part]
.section-bar[]
# Section title
.arc[]
```

---

#### `class: outline`

Outline slide. Navy background. Two-column grid (2fr / 3fr): `h1` on the left with terracotta border, numbered `ol` on the right. Automatic gold zero-padded numbers (`decimal-leading-zero` via CSS counter).

```markdown
---
class: outline

# Outline

1. First part
2. Second part
3. Third part
```

---

#### `class: citation`

Quote slide. Background `--apc-rose-light`. Giant quotation mark watermark via `.quote-mark[]`. Text in `blockquote` (Lora italic). Terracotta bar via `.cite-bar[]`. Source via `.source[]` or `cite`. Has `padding-bottom: var(--footer-h)` so it is compatible with the auto-footer script.

```markdown
---
class: citation

.quote-mark["]
> The quote text here.
.cite-bar[]
.source[Author, Year]
```

---

#### `class: twocol`

Inherits the content style (navy header). Use `.columns[]` with `.left-col[]` and `.right-col[]` for the 50/50 grid. Left `h2` are navy, right `h2` are terracotta.

---

#### `class: blanc`

Content slide on a white background. Removes the rose sidebar. Keeps the navy header.

---

#### `class: fullbleed`

Full-bleed slide with no header. No padding, black background. Any `img` placed in the slide fills the entire surface (`object-fit: cover`). Page number in semi-transparent white. Optional caption via `.caption[]`.

```markdown
---
class: fullbleed

![](./img/photo.jpg)

.caption[Credit: ESA]
```

Also works with Remark's native `background-image` property:

```markdown
---
class: fullbleed
background-image: url(./img/photo.jpg)
background-size: cover
---
```

---

### Reusable components

| Class/Element | Usage |
|---------------|-------|
| `.stamp[]` | Decorative stamp, centred, rotated −18°, terracotta colour (e.g. `.stamp[PRELIMINARY]`) |
| `.footer[]` | Slide footer (logo + text) |
| `.apc-footer` | Automatic APC footer (logo + "Astroparticule et Cosmologie") |
| `.apc-logo` | White APC logo (dark backgrounds), anchored bottom-right |
| `.highlight-box` | Navy-light box with left navy border |
| `.gold-sep` | Gold horizontal separator (12 px) |
| `.img-full img` | Full-width image |
| `.cols` | Generic 2-column grid 1fr/1fr |
| `.cols-3` | Generic 3-column grid 1fr/1fr/1fr |

### Text utility classes

| Class | Effect |
|-------|--------|
| `.navy` / `.gold` / `.terra` / `.rose` | APC text colours |
| `.muted` | `--apc-text-mid` colour, 0.88em |
| `.small` | 0.82em |
| `.large` | 1.25em |
| `.center` / `.right` | Alignment |
| `.bold` | font-weight 700 |
