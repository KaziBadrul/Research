#!/usr/bin/env python3
"""Rebuild the auto-generated sections of README.md.

Scans the repository for Jupyter notebooks and paper PDFs, then rewrites the
blocks between these markers in README.md:

    <!-- AUTO:STATS:START -->      ... <!-- AUTO:STATS:END -->
    <!-- AUTO:NOTEBOOKS:START -->  ... <!-- AUTO:NOTEBOOKS:END -->
    <!-- AUTO:PAPERS:START -->     ... <!-- AUTO:PAPERS:END -->

It also writes shields.io endpoint badges to .github/badges/*.json.

Titles and descriptions come from .github/readme-meta.json when present.
Unknown arXiv PDFs are looked up on the arXiv API and cached back into that
file, so a new paper only has to be dropped into the repo to show up.

Run locally with:  python scripts/update_readme.py
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
META_PATH = ROOT / ".github" / "readme-meta.json"
BADGE_DIR = ROOT / ".github" / "badges"

OWNER_REPO = "KaziBadrul/Research"
BRANCH = "main"

SKIP_DIRS = {".git", ".github", ".ipynb_checkpoints", "node_modules", "__pycache__"}
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


# ---------------------------------------------------------------- helpers
def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {"papers": {}, "notebooks": {}}


def save_meta(meta: dict) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_files(suffix: str) -> list[Path]:
    out = []
    for p in ROOT.rglob(f"*{suffix}"):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        out.append(p)
    return sorted(out, key=lambda p: str(p.relative_to(ROOT)).lower())


def url_path(rel: Path) -> str:
    return "/".join(urllib.parse.quote(part) for part in rel.parts)


def arxiv_lookup(arxiv_id: str) -> dict | None:
    """Fetch title, first author and year from the arXiv API. Returns None on failure."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            root = ET.fromstring(resp.read())
    except Exception as exc:  # network down, rate limit, etc.
        print(f"  ! arXiv lookup failed for {arxiv_id}: {exc}", file=sys.stderr)
        return None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        return None
    title = " ".join(entry.find("a:title", ns).text.split())
    authors = [a.find("a:name", ns).text for a in entry.findall("a:author", ns)]
    first = authors[0].split()[-1] if authors else ""
    return {"title": title, "authors": f"{first} et al." if len(authors) > 1 else first}


def arxiv_date(arxiv_id: str) -> str:
    """2604.05433 -> '2026-04'."""
    yymm = arxiv_id.split(".")[0]
    return f"20{yymm[:2]}-{yymm[2:]}"


# ---------------------------------------------------------------- papers
def collect_papers(meta: dict) -> list[dict]:
    papers, seen_titles = [], set()
    pmeta = meta.setdefault("papers", {})

    for pdf in repo_files(".pdf"):
        rel = pdf.relative_to(ROOT)
        m = ARXIV_RE.search(pdf.stem)
        key = m.group(1) if m else pdf.name
        info = dict(pmeta.get(key, {}))

        if m and "title" not in info:
            fetched = arxiv_lookup(key)
            if fetched:
                info.update(fetched)
                pmeta[key] = {**pmeta.get(key, {}), **fetched}
        if "title" not in info:
            info["title"] = re.sub(r"\s+", " ", pdf.stem.replace("_", " ")).strip()

        norm = info["title"].lower()
        if norm in seen_titles:  # same paper stored twice
            continue
        seen_titles.add(norm)

        arxiv_id = m.group(1) if m else info.get("arxiv")
        papers.append({
            "title": info["title"],
            "authors": info.get("authors", ""),
            "topic": info.get("topic", ""),
            "date": info.get("date") or (arxiv_date(arxiv_id) if arxiv_id else ""),
            "arxiv": arxiv_id,
            "path": rel,
            "folder": info.get("code") or (rel.parts[0] if len(rel.parts) > 1 else None),
        })

    papers.sort(key=lambda p: p["date"] or "0000", reverse=True)
    return papers


def render_papers(papers: list[dict]) -> str:
    lines = [
        "| Date | Paper | Topic | Links |",
        "|:---:|---|---|---|",
    ]
    for p in papers:
        title = f"**{p['title']}**"
        if p["authors"]:
            title += f"<br/><sub>{p['authors']}</sub>"
        links = [f"[📄 PDF]({url_path(p['path'])})"]
        if p["arxiv"]:
            links.append(f"[arXiv](https://arxiv.org/abs/{p['arxiv']})")
        if p["folder"] and any((ROOT / p["folder"]).glob("*.ipynb")):
            links.append(f"[🧪 Code]({url_path(Path(p['folder']))})")
        lines.append(f"| {p['date'] or '–'} | {title} | {p['topic']} | {' · '.join(links)} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- notebooks
def notebook_heading(nb_path: Path) -> tuple[str | None, str | None]:
    """First markdown H1 and the first plain sentence after it."""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", []))
        h1 = re.search(r"^#\s+(.+)$", text, re.M)
        if not h1:
            continue
        rest = [ln.strip() for ln in text[h1.end():].splitlines()
                if ln.strip() and not ln.strip().startswith(("#", "|", "-", "*", "`", "!"))]
        return h1.group(1).strip(), (rest[0] if rest else None)
    return None, None


def collect_notebooks(meta: dict) -> list[dict]:
    nmeta = meta.setdefault("notebooks", {})
    out = []
    for nb in repo_files(".ipynb"):
        rel = nb.relative_to(ROOT)
        key = rel.as_posix()
        info = nmeta.get(key, {})
        h1, blurb = notebook_heading(nb)
        out.append({
            "path": rel,
            "title": info.get("title") or h1 or nb.stem.replace("_", " "),
            "description": info.get("description") or blurb or "",
            "stack": info.get("stack", ""),
            "size_kb": nb.stat().st_size // 1024,
        })
    return out


def render_notebooks(notebooks: list[dict]) -> str:
    lines = [
        "| Notebook | What it does | Stack | Run |",
        "|---|---|---|:---:|",
    ]
    for n in notebooks:
        p = url_path(n["path"])
        colab = f"https://colab.research.google.com/github/{OWNER_REPO}/blob/{BRANCH}/{p}"
        viewer = f"https://nbviewer.org/github/{OWNER_REPO}/blob/{BRANCH}/{p}"
        badge = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        lines.append(
            f"| [**{n['title']}**]({p})<br/><sub>`{n['path'].as_posix()}`</sub> "
            f"| {n['description']} | {n['stack']} "
            f"| {badge}({colab})<br/><sub>[nbviewer]({viewer})</sub> |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- stats + badges
def render_stats(papers: list[dict], notebooks: list[dict]) -> str:
    folders = sorted({p.relative_to(ROOT).parts[0] for p in repo_files(".ipynb")
                      if len(p.relative_to(ROOT).parts) > 1})
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    return (
        f"> 📚 **{len(papers)}** papers &nbsp;·&nbsp; "
        f"🧪 **{len(notebooks)}** notebooks &nbsp;·&nbsp; "
        f"🗂️ **{len(folders)}** paper reproductions &nbsp;·&nbsp; "
        f"🔄 index rebuilt **{today}**"
    )


def write_badges(papers: list[dict], notebooks: list[dict]) -> None:
    BADGE_DIR.mkdir(parents=True, exist_ok=True)
    badges = {
        "papers": {"label": "papers", "message": str(len(papers)), "color": "8B5CF6"},
        "notebooks": {"label": "notebooks", "message": str(len(notebooks)), "color": "F37626"},
    }
    for name, b in badges.items():
        (BADGE_DIR / f"{name}.json").write_text(
            json.dumps({"schemaVersion": 1, **b}, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- main
def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- AUTO:{name}:START -->)(.*?)(<!-- AUTO:{name}:END -->)", re.S)
    if not pattern.search(text):
        print(f"  ! marker AUTO:{name} not found in README.md", file=sys.stderr)
        return text
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def main() -> None:
    meta = load_meta()
    papers = collect_papers(meta)
    notebooks = collect_notebooks(meta)

    text = README.read_text(encoding="utf-8")
    text = replace_block(text, "STATS", render_stats(papers, notebooks))
    text = replace_block(text, "NOTEBOOKS", render_notebooks(notebooks))
    text = replace_block(text, "PAPERS", render_papers(papers))
    README.write_text(text, encoding="utf-8")

    write_badges(papers, notebooks)
    save_meta(meta)
    print(f"README updated: {len(papers)} papers, {len(notebooks)} notebooks.")


if __name__ == "__main__":
    main()
