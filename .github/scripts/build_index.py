#!/usr/bin/env python3
"""Generate index.html for the slides repository.

Run from anywhere — uses the script's own location to find the repo root.
On first run also seeds topics.yml with keyword-inferred topics.
"""

import html
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
TEMPLATE  = Path(__file__).parent / 'index_template.html'

# ── Text cleanup ─────────────────────────────────────────────────

_REMARK_INLINE = re.compile(r'\.\w+\[([^\]]*)\]')
_MD_LINK_INLINE = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_MD_LINK_REF    = re.compile(r'\[([^\]]+)\]\[[^\]]*\]')
_BR_TAG         = re.compile(r'<br\s*/?>', re.I)
_HTML_TAG       = re.compile(r'<[^>]+>')
_EXTRA_WS       = re.compile(r'\s+')


def _clean(text: str) -> str:
    text = _REMARK_INLINE.sub(r'\1', text)
    text = _MD_LINK_INLINE.sub(r'\1', text)
    text = _MD_LINK_REF.sub(r'\1', text)
    text = _BR_TAG.sub(' ', text)
    text = _HTML_TAG.sub('', text)
    return _EXTRA_WS.sub(' ', text).strip()


# ── Metadata extraction from slides.md ───────────────────────────

def _extract(slides_md: Path) -> tuple[str, str]:
    """Return (title, venue) from the cover slide."""
    first_slide = slides_md.read_text(encoding='utf-8', errors='replace').split('\n---\n')[0]
    title = venue = h3_candidate = ''

    for raw_line in first_slide.splitlines():
        line = raw_line.strip()
        if not title and line.startswith('# '):
            title = _clean(line[2:])
        elif line.startswith('#### '):
            candidate = _clean(line[5:])
            if candidate and '@' not in candidate and 'mailto' not in candidate.lower():
                venue = candidate
                break
        elif not h3_candidate and line.startswith('### '):
            candidate = _clean(line[4:])
            if candidate and not re.match(r'with\s+examples', candidate, re.I) and '@' not in candidate:
                h3_candidate = candidate

    return title, venue or h3_candidate


# ── topics.yml ───────────────────────────────────────────────────

_TOPIC_KEYWORDS = [
    ('MLOps',  r'mlops|reproducib'),
    ('ML',     r'neural|deep.learn|euclid.school|machine.learn|cosmo|cfe|generative|asterics|hands.on|ml_for|mml|journee_rt'),
    ('AI',     r'impact.ia|aissai|ai.semi|ia.g|journee_dev|generative'),
    ('Python', r'python'),
    ('Git',    r'\bgit\b|gitlab'),
    ('Tools',  r'cafe.info|html.present'),
]


def _infer_topic(year: str, folder: str) -> str:
    key = f'{year}/{folder}'.lower()
    for topic, pattern in _TOPIC_KEYWORDS:
        if re.search(pattern, key):
            return topic
    return ''


def _load_topics(path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    if not path.exists():
        return topics
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and ':' in line:
            k, _, v = line.partition(':')
            topics[k.strip()] = v.strip()
    return topics


def _seed_topics(path: Path, presentations: list[dict]) -> None:
    lines = ['# Flat map: YYYY/folder-name → topic', '# Seeded by build_index.py — edit freely.', '']
    prev_year = None
    for p in presentations:
        topic = _infer_topic(p['year'], p['folder'])
        if not topic:
            continue
        if p['year'] != prev_year:
            if prev_year is not None:
                lines.append('')
            prev_year = p['year']
        lines.append(f"{p['year']}/{p['folder']}: {topic}")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ── Presentation discovery ────────────────────────────────────────

_YEAR_RE = re.compile(r'^\d{4}$')


def _discover(root: Path) -> list[dict]:
    presentations = []
    for year_dir in sorted(root.iterdir(), reverse=True):
        if not year_dir.is_dir() or not _YEAR_RE.match(year_dir.name):
            continue
        for pres_dir in sorted(year_dir.iterdir()):
            if not pres_dir.is_dir() or pres_dir.name == 'img':
                continue
            if not (pres_dir / 'index.html').exists():
                continue

            slides_md = pres_dir / 'slides.md'
            title, venue = _extract(slides_md) if slides_md.exists() else ('', '')
            if not title:
                title = pres_dir.name.replace('-', ' ').replace('_', ' ').title()

            # meta.yml overrides (title, venue, topic)
            meta_topic = ''
            meta_yml = pres_dir / 'meta.yml'
            if meta_yml.exists():
                for line in meta_yml.read_text(encoding='utf-8').splitlines():
                    k, _, v = line.partition(':')
                    v = v.strip().strip('"\'')
                    if   k.strip() == 'title' and v: title = v
                    elif k.strip() == 'venue' and v: venue = v
                    elif k.strip() == 'topic' and v: meta_topic = v

            presentations.append({
                'year': year_dir.name,
                'folder': pres_dir.name,
                'title': title,
                'venue': venue,
                'path': f'{year_dir.name}/{pres_dir.name}/',
                'meta_topic': meta_topic,
            })
    return presentations


# ── HTML rendering ────────────────────────────────────────────────

def _render(presentations: list[dict], topics: dict[str, str]) -> str:
    all_topics = sorted({t for t in topics.values() if t})
    years = sorted({p['year'] for p in presentations}, reverse=True)
    year_range = f'{years[-1]}–{years[0]}' if len(years) > 1 else years[0]

    pills = '      <button class="filter-pill active" data-topic="all">All</button>\n'
    for t in all_topics:
        pills += f'      <button class="filter-pill" data-topic="{t.lower()}">{html.escape(t)}</button>\n'

    rows = ''
    for year in years:
        rows += f'    <p class="talks-subhead">{year}</p>\n    <div class="talks-list">\n'
        for p in (p for p in presentations if p['year'] == year):
            topic = p['meta_topic'] or topics.get(f"{p['year']}/{p['folder']}", '')
            badge_cls = f' {topic.lower()}' if topic else ''
            venue_html = f'\n          <span class="talk-venue">{html.escape(p["venue"])}</span>' if p['venue'] else ''
            rows += (
                f'      <div class="talk-row" data-topic="{topic.lower()}">\n'
                f'        <span class="talk-badge{badge_cls}">{html.escape(topic)}</span>\n'
                f'        <div class="talk-body">\n'
                f'          <a class="talk-title" href="{p["path"]}">{html.escape(p["title"])}</a>{venue_html}\n'
                f'        </div>\n'
                f'        <span class="talk-year-col">{year}</span>\n'
                f'      </div>\n'
            )
        rows += '    </div>\n'

    return (
        TEMPLATE.read_text(encoding='utf-8')
        .replace('__TOTAL__',      str(len(presentations)))
        .replace('__YEAR_RANGE__', year_range)
        .replace('__PILLS__',      pills)
        .replace('__ROWS__',       rows)
    )


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    topics_path = REPO_ROOT / 'topics.yml'
    presentations = _discover(REPO_ROOT)

    if not topics_path.exists():
        _seed_topics(topics_path, presentations)
        print(f'Seeded {topics_path.name}')

    topics = _load_topics(topics_path)
    index_html = _render(presentations, topics)
    (REPO_ROOT / 'index.html').write_text(index_html, encoding='utf-8')
    print(f'Wrote index.html  ({len(presentations)} presentations)')


if __name__ == '__main__':
    main()
