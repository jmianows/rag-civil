"""Shared utilities for ingestion scripts."""

import json
from pathlib import Path


def _load_link(pdf_path: Path, root: Path) -> str:
    """Return the public URL for this PDF from a sidecar or root registry, or ''."""
    # Option A: per-agency sidecar — check _links.json, links.json, and *_links.json glob
    for name in ("_links.json", "links.json"):
        sidecar = pdf_path.parent / name
        if sidecar.exists():
            try:
                links = json.loads(sidecar.read_text(encoding="utf-8"))
                url = links.get(pdf_path.name, "")
                if url:
                    return url
            except Exception:
                pass
    for sidecar in pdf_path.parent.glob("*_links.json"):
        try:
            links = json.loads(sidecar.read_text(encoding="utf-8"))
            url = links.get(pdf_path.name, "")
            if url:
                return url
        except Exception:
            pass
    # Option B: root-level _registry.json keyed by relative path
    registry = root / "_registry.json"
    if registry.exists():
        try:
            reg = json.loads(registry.read_text(encoding="utf-8"))
            rel = pdf_path.relative_to(root).as_posix()
            url = reg.get(rel, "")
            if url:
                return url
        except Exception:
            pass
    return ""
