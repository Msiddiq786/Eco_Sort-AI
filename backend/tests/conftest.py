"""
Shared pytest fixtures.

Sets DATABASE_URL to a temporary SQLite file so tests never touch
the development database (waste_segregation.db).
The temp file is removed after the test session ends.
"""
from __future__ import annotations

import os
import pathlib
import sys

# Ensure backend root is on sys.path
_BACKEND_DIR = pathlib.Path(__file__).parent.parent.resolve()
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pytest

# ── Override database before any app import resolves settings ────────────────
_TEST_DB = pathlib.Path(__file__).parent / "test_waste_temp.db"
os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{_TEST_DB}"

# ── Disable remote HF downloads during test suite ────────────────────────────
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db():
    """Remove the temporary test database after the whole session finishes."""
    yield
    # On Windows the aiosqlite engine may still hold a file handle at teardown.
    # Suppress the PermissionError rather than failing the test session.
    if _TEST_DB.exists():
        try:
            _TEST_DB.unlink()
        except PermissionError:
            pass  # File still locked by background engine; will be cleaned next run.
