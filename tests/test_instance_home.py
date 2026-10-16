"""HARVEY_HOME: run a second, fully separate Harvey from the same checkout.

Each check runs in a fresh interpreter because paths are resolved at import
time; reloading modules in-process would leak state between tests.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run(code: str, home: Path | None, cwd: Path | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("HARVEY_HOME", "SMTP_HOST")}
    if home is not None:
        env["HARVEY_HOME"] = str(home)
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(cwd or ROOT), env=env, capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "instance"
    (h / "skills").mkdir(parents=True)
    (h / "harvey.yaml").write_text("persona:\n  name: Instance\n")
    (h / ".env").write_text("SMTP_HOST=smtp.instance.test\n")
    (h / "skills" / "product_knowledge.md").write_text("INSTANCE PRODUCT")
    return h


def test_default_mode_is_unchanged():
    r = run("""
        import json
        from harvey import paths
        print(json.dumps({"same": paths.INSTANCE_ROOT == paths.PROJECT_ROOT,
                          "mode": paths.INSTANCE_MODE,
                          "data": str(paths.DATA_DIR), "root": str(paths.PROJECT_ROOT)}))
    """, None)
    assert r["same"] and not r["mode"]
    assert r["data"] == str(Path(r["root"]) / "data")


def test_instance_paths(home):
    r = run("""
        import json
        from harvey import paths, state
        from harvey.integrations import gmail
        print(json.dumps({"root": str(paths.INSTANCE_ROOT), "db": str(state.DB_PATH),
                          "token": str(gmail.TOKEN_FILE),
                          "skills": [str(d) for d in paths.SKILLS_DIRS]}))
    """, home)
    h = str(home.resolve())
    assert r["root"] == h
    assert r["db"] == f"{h}/data/harvey.db"
    assert r["token"] == f"{h}/data/gmail_token.json"
    assert r["skills"] == [f"{h}/skills", str(ROOT / "skills")]


def test_config_never_falls_back_to_checkout(home, tmp_path):
    r = run("""
        import json
        from harvey.config import _find_config_file
        print(json.dumps({"cfg": _find_config_file()}))
    """, home, cwd=ROOT)  # the checkout's own harvey.yaml is in cwd
    assert r["cfg"] == str(home.resolve() / "harvey.yaml")

    (home / "harvey.local.yaml").write_text("persona:\n  name: Local\n")
    r = run("""
        import json
        from harvey.config import _find_config_file
        print(json.dumps({"cfg": _find_config_file()}))
    """, home)
    assert r["cfg"].endswith("harvey.local.yaml")

    empty = tmp_path / "empty"
    empty.mkdir()
    r = run("""
        import json
        from harvey.config import _find_config_file, ConfigFileNotFoundError
        try:
            _find_config_file(); ok = False
        except ConfigFileNotFoundError:
            ok = True
        print(json.dumps({"raised": ok}))
    """, empty)
    assert r["raised"]


def test_env_comes_from_instance(home):
    r = run("""
        import json
        from harvey.config import load_env
        print(json.dumps({"smtp": load_env().smtp_host}))
    """, home)
    assert r["smtp"] == "smtp.instance.test"


def test_db_is_created_in_instance(home):
    r = run("""
        import asyncio, json
        from harvey.state import StateManager
        s = StateManager()
        asyncio.run(s.init_db())
        print(json.dumps({"db": s.db_path}))
    """, home)
    assert Path(r["db"]).exists()
    assert Path(r["db"]).parent == home.resolve() / "data"


def test_instance_skill_wins_shared_skill_still_loads(home):
    r = run("""
        import json
        from harvey.brain import Brain
        b = Brain.__new__(Brain)
        print(json.dumps({"product": Brain.load_skill(b, "product_knowledge"),
                          "shared": len(Brain.load_skill(b, "email_frameworks")) > 0}))
    """, home)
    assert r["product"] == "INSTANCE PRODUCT"
    assert r["shared"]
