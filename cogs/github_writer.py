"""Commit Catalogue CSV edits back to GitHub.

Streamlit Cloud has an ephemeral filesystem, so a `df.to_csv(...)` on a
running container is wiped on the next restart/redeploy. Keeping the
repo as the source of truth via the GitHub Contents API means every
edit is durable, auditable in git log, and triggers an auto-redeploy.

Requires three secrets in `.streamlit/secrets.toml`:
- github_token  (fine-grained PAT, Contents: read+write on this repo)
- github_repo   (e.g. "akinsoyleyen/cogs_calculator")
- github_branch (optional, default "main")
"""
import base64
from pathlib import Path

import requests
import streamlit as st

_API = "https://api.github.com"


def github_is_configured() -> bool:
    if not hasattr(st, "secrets"):
        return False
    try:
        return bool(st.secrets.get("github_token")) and bool(st.secrets.get("github_repo"))
    except Exception:
        return False


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {st.secrets['github_token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _branch() -> str:
    return st.secrets.get("github_branch", "main")


def push_file(path: str, message: str) -> str:
    """Commit the current local contents of `path` to GitHub.

    Returns the new commit SHA, or "unchanged" if the remote already
    matches the local bytes. Raises RuntimeError on API failure.
    """
    repo = st.secrets["github_repo"]
    branch = _branch()
    content = Path(path).read_bytes()

    contents_url = f"{_API}/repos/{repo}/contents/{path}"

    # Look up the current file SHA (404 means new file).
    r = requests.get(contents_url, headers=_headers(), params={"ref": branch}, timeout=20)
    file_sha = r.json().get("sha") if r.status_code == 200 else None

    if file_sha:
        try:
            remote_b64 = r.json().get("content", "").replace("\n", "")
            if base64.b64decode(remote_b64) == content:
                return "unchanged"
        except Exception:
            pass

    body = {
        "message": message,
        "content": base64.b64encode(content).decode("ascii"),
        "branch": branch,
    }
    if file_sha:
        body["sha"] = file_sha

    r = requests.put(contents_url, headers=_headers(), json=body, timeout=30)
    if not r.ok:
        raise RuntimeError(
            f"GitHub commit failed ({r.status_code}) for {path}: {r.text[:300]}"
        )
    return r.json()["commit"]["sha"]


def push_paths(paths: list[str], message: str) -> dict[str, str]:
    """Commit several files in one batch (one commit per file).

    Returns {path: commit_sha_or_"unchanged"}. Raises on the first failure
    so the caller can show the user which file didn't make it.
    """
    return {p: push_file(p, message) for p in paths}
