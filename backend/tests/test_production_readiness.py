"""Production deployment readiness tests — config parsing, root /health, CORS & DB handling."""
from __future__ import annotations

import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient


def test_settings_accepts_csv_and_json_cors_and_port_env(monkeypatch):
    from app.core.config import Settings

    monkeypatch.setenv("CORS_ORIGINS", "https://ecosort-ai.vercel.app, http://localhost:5173")
    monkeypatch.setenv("PORT", "9090")
    s = Settings(_env_file=None)
    assert s.CORS_ORIGINS == ["https://ecosort-ai.vercel.app", "http://localhost:5173"]
    assert s.APP_PORT == 9090

    monkeypatch.setenv("CORS_ORIGINS", '["https://example.vercel.app"]')
    s2 = Settings(_env_file=None)
    assert s2.CORS_ORIGINS == ["https://example.vercel.app"]


def test_root_and_health_endpoints():
    from app.main import app

    with TestClient(app) as client:
        root_resp = client.get("/")
        assert root_resp.status_code == 200
        assert root_resp.json()["service"] == "EcoSort AI API"

        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        body = health_resp.json()
        assert body["status"] == "ok"
        assert "vit_loaded" in body
        assert "clip_loaded" in body
        assert "models_ready" in body


def test_image_storage_respects_dynamic_max_size(monkeypatch):
    from fastapi import HTTPException
    from app.core.config import settings
    from app.services.image_storage import validate_upload

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(20, 120, 40)).save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 1)
    validate_upload(img_bytes, "small.jpg")

    oversized = img_bytes + b"\x00" * (2 * 1024 * 1024)
    with pytest.raises(HTTPException) as exc:
        validate_upload(oversized, "oversized.jpg")
    assert exc.value.status_code == 400
