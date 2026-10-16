"""Canonical filesystem locations for the Harvey checkout.

Harvey is repo-centric: config (harvey.yaml, .env), knowledge (prompts/,
skills/), and state (data/) all live next to the package, unless HARVEY_HOME
points the instance files somewhere else (see ``_instance_root``). ``resolve()``
matters — when the package is reached through a symlink (some editable
installs), a bare ``Path(__file__).parent`` would point into
site-packages and Harvey would silently read/write the wrong files.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _instance_root() -> Path:
    """Where this Harvey's *instance* files live.

    By default that is the checkout itself (config, .env and data/ next to the
    package). Set ``HARVEY_HOME`` to run a second, fully separate Harvey from
    the same code: its harvey.yaml / harvey.local.yaml, .env, data/ (database,
    Gmail token, pid, log) and trained product skills all live there, while
    prompts and the shared skills library still come from the checkout.
    """
    home = os.environ.get("HARVEY_HOME", "").strip()
    return Path(home).expanduser().resolve() if home else PROJECT_ROOT


INSTANCE_ROOT = _instance_root()
INSTANCE_MODE = INSTANCE_ROOT != PROJECT_ROOT
DATA_DIR = INSTANCE_ROOT / "data"
ENV_FILE = INSTANCE_ROOT / ".env"
# Instance skills (trained product_knowledge / competitive_intel) win over the
# shared library in the checkout.
SKILLS_DIRS = [INSTANCE_ROOT / "skills"] + ([PROJECT_ROOT / "skills"] if INSTANCE_MODE else [])
