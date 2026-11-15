"""Render the repository's Markdown and figures in the documentation site."""

import posixpath
import re
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from mkdocs.structure.files import File


ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "docs/index.md": "index.md",
    "docs/user_guide.md": "user_guide.md",
    "reproduce/README.md": "reproduce/README.md",
    "reproduce/data/README.md": "reproduce/data/README.md",
}
SOURCES = {page: source for source, page in PAGES.items()}
IMAGES = {
    "examples/assets/deceptron_schematic.png",
    "examples/assets/heat3d_preview.png",
    "examples/assets/navier_stokes_withheld_t2.png",
}


def on_files(files, config):
    for source, page in PAGES.items():
        if not source.startswith("docs/"):
            markdown = (ROOT / source).read_text(encoding="utf-8")
            files.append(File.generated(config, page, content=markdown))
    for image in sorted(IMAGES):
        files.append(File.generated(config, image, abs_src_path=str(ROOT / image)))
    return files


def on_page_markdown(markdown, page, config, files):
    uri = page.file.src_uri
    source = SOURCES.get(uri, f"docs/{uri}")

    def resolve(link):
        parts = urlsplit(link)
        if parts.scheme or parts.netloc or not parts.path:
            return link
        path = posixpath.normpath(posixpath.join(posixpath.dirname(source), parts.path))
        if path == "README.md" and parts.fragment == "installation":
            path = "docs/index.md"
        if path in PAGES or path in IMAGES:
            path = posixpath.relpath(PAGES.get(path, path), posixpath.dirname(uri) or ".")
        else:
            kind = "tree" if (ROOT / path).is_dir() else "blob"
            repo = urlsplit(config.repo_url.rstrip("/"))
            path = f"{repo.path}/{kind}/main/{quote(path)}"
            return urlunsplit((repo.scheme, repo.netloc, path, parts.query, parts.fragment))
        return urlunsplit(("", "", path, parts.query, parts.fragment))

    markdown = re.sub(r"\]\(([^\s)]+)\)", lambda m: f"]({resolve(m[1])})", markdown)
    markdown = re.sub(r'\b(src|href)="([^"]+)"', lambda m: f'{m[1]}="{resolve(m[2])}"', markdown)
    markdown = re.sub(r"^```math\s*\n(.*?)\n```[ \t]*$", r"$$\n\1\n$$", markdown, flags=re.M | re.S)

    # The original guide has some LaTeX commands escaped for Markdown.
    def inline_math(match):
        body = re.sub(r"\\+([_{}()^.+-])", r"\1", match[1]).replace("\\\\", "\\")
        return f"${body}$"

    return re.sub(r"(?<!\$)\$([^$\n]+)\$(?!\$)", inline_math, markdown)
