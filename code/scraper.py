"""
Converts markdown corpus into clean text files for indexing.

- Reads .md files from data/claude/, data/visa/, data/hackerrank/ (recursively)
- Removes YAML frontmatter and markdown formatting
- Outputs plain .txt files for build_index.py
- Subfolders are flattened using "__" in output filenames

Output format:
SOURCE: <url>
TITLE: <title>
DOMAIN: <claude | visa | hackerrank>
---
<clean text>

Usage:
  python scraper.py                  # process all domains
  python scraper.py --domain claude  # process single domain
  python scraper.py --dry-run        # show stats only

Output folders:
  data/claude_txt/
  data/visa_txt/
  data/hackerrank_txt/
"""

import argparse
import re
from pathlib import Path

# Config
BASE_DIR = Path(__file__).parent.parent / "data"
DOMAINS  = ["hackerrank", "claude", "visa"]

def txt_out_dir(domain: str) -> Path:
    return BASE_DIR / f"{domain}_txt"


def parse_frontmatter(raw: str) -> tuple[dict, str]:
    """
    Split a Markdown file into (frontmatter_dict, body_text).
    Handles the --- ... --- block at the top.
    """
    raw  = raw.strip()
    meta = {}
    body = raw

    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            fm_block = raw[3:end].strip()
            body     = raw[end + 4:].strip()

            for line in fm_block.splitlines():
                if ":" not in line:
                    continue
                key, _, val = line.partition(":")
                key = key.strip().lower().replace("-", "_")
                val = val.strip().strip('"').strip("'")
                if val:
                    meta[key] = val

    return meta, body


def extract_metadata(meta: dict, domain: str, file_path: Path) -> dict:
    source = (
        meta.get("source_url")
        or meta.get("final_url")
        or meta.get("url")
        or f"file://{file_path.resolve()}"
    )
    title = (
        meta.get("title")
        or file_path.stem.replace("-", " ").replace("_", " ").title()
    )
    return {"source": source, "title": title, "domain": domain}



# Markdown to plain text conversion
def md_to_text(md: str) -> str:
    """
    Remove Markdown markup and return readable plain text.
    """
    t = md

    # Remove HTML tags
    t = re.sub(r"<[^>]+>", " ", t)

    # Headings: ## Foo  ->  Foo
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)

    # Remove image markdown
    t = re.sub(r"!\[.*?\]\(.*?\)", "", t)

    # Links: [text](url)  ->  text
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)

    # Bold / italic
    t = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", t)
    t = re.sub(r"_{1,3}([^_\n]+)_{1,3}",   r"\1", t)

    # Horizontal rules
    t = re.sub(r"^[-*_]{3,}\s*$", "", t, flags=re.MULTILINE)

    # Blockquote markers
    t = re.sub(r"^>\s*", "", t, flags=re.MULTILINE)

    # Code fences — keep content, drop the fence
    t = re.sub(r"```[^\n]*\n(.*?)```", r"\1", t, flags=re.DOTALL)
    t = re.sub(r"`([^`]+)`", r"\1", t)

    # List bullets
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)

    # Numbered lists: "1. Foo" -> "Foo"
    t = re.sub(r"^\s*\d+\.\s+", "", t, flags=re.MULTILINE)

    # Collapse multiple blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


def convert_md_file(md_path: Path, domain: str, out_dir: Path, dry_run: bool) -> bool:
    """
    Read one .md file, parse frontmatter + body, write .txt.
    Returns True if a non-empty file is produced.
    """
    try:
        raw = md_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        print(f"    [warn] cannot read {md_path.name}: {exc}")
        return False

    meta, body = parse_frontmatter(raw)
    info = extract_metadata(meta, domain, md_path)
    text = md_to_text(body)

    if len(text.strip()) < 50:
        return False   # skip near-empty / index pages

    content = (
        f"SOURCE: {info['source']}\n"
        f"TITLE: {info['title']}\n"
        f"DOMAIN: {info['domain']}\n"
        f"---\n"
        f"{text}\n"
    )

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Flatten subfolder path into filename with __ separator
        try:
            rel = md_path.relative_to(BASE_DIR / domain)
        except ValueError:
            rel = Path(md_path.name)
        slug = str(rel).replace("/", "__").replace("\\", "__")
        slug = re.sub(r"[^a-zA-Z0-9._\-]", "_", slug).replace(".md", ".txt")
        (out_dir / slug).write_text(content, encoding="utf-8")

    return True


def process_domain(domain: str, dry_run: bool) -> int:
    in_dir  = BASE_DIR / domain
    out_dir = txt_out_dir(domain)

    if not in_dir.exists():
        print(f"\n[skip] {in_dir} does not exist — no .md files to convert")
        return 0

    md_files = sorted(in_dir.rglob("*.md"))
    if not md_files:
        print(f"\n[skip] No .md files found in {in_dir}")
        return 0

    print(f"\n{'='*60}")
    print(f"  Domain  : {domain}")
    print(f"  Source  : {in_dir}")
    print(f"  Files   : {len(md_files)} .md files")
    print(f"  Output  : {out_dir}")
    print(f"{'='*60}")

    saved = 0
    for path in md_files:
        ok = convert_md_file(path, domain, out_dir, dry_run)
        if ok:
            saved += 1
            status = "[dry]" if dry_run else "[ok] "
            print(f"  {status} {path.relative_to(BASE_DIR / domain)}")
        else:
            print(f"  [skip] {path.relative_to(BASE_DIR / domain)}  (too short or unreadable)")

    print(f"\n  -> {'Would write' if dry_run else 'Wrote'} {saved} / {len(md_files)} files")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pre-provided .md corpus to .txt for build_index.py"
    )
    parser.add_argument("--domain",  choices=DOMAINS, help="Process only this domain")
    parser.add_argument("--dry-run", action="store_true", help="No file writes, stats only")
    args = parser.parse_args()

    targets = [args.domain] if args.domain else DOMAINS

    total = 0
    for domain in targets:
        total += process_domain(domain, dry_run=args.dry_run)

    print(f"\nTotal converted: {total}")
    if not args.dry_run:
        print("\nOutput directories:")
        for domain in targets:
            out   = txt_out_dir(domain)
            count = len(list(out.glob("*.txt"))) if out.exists() else 0
            print(f"  {out}  ({count} .txt files)")

    print("\nNext step:  python build_index.py")


if __name__ == "__main__":
    main()