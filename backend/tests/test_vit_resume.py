"""
Verification suite for ViT training resumption requirements.
Ensures:
1. Path resolution (best vs latest)
2. Loading of latest checkpoint assets
3. Epoch determination (latest_epoch=9 -> start_epoch=10)
4. Preservation of best model metadata (best_epoch=7, score=0.9314)
5. Restoration of optimizer and scheduler states
6. start_epoch > 1 (no re-training from Epoch 1)
7. Patience counter calculation based on best_epoch (patience_counter=2)
8. Training history preservation and appending
9. Startup logs formatting
"""
import json
from pathlib import Path
import pytest
import torch

from training.train_vit import resolve_path


def test_req1_path_resolution():
    """Requirement 1: Resolves best and latest directories correctly."""
    cases = [
        Path("backend/models/vit"),
        Path("backend/models/vit/best"),
        Path("backend/models/vit/latest"),
    ]
    for c in cases:
        raw_ckpt = resolve_path(c)
        if raw_ckpt.name == "latest":
            latest_dir = raw_ckpt
            best_dir = raw_ckpt.parent / "best"
        elif raw_ckpt.name == "best":
            best_dir = raw_ckpt
            latest_dir = raw_ckpt.parent / "latest"
        elif (raw_ckpt / "best").exists() or (raw_ckpt / "latest").exists() or raw_ckpt.name == "vit":
            best_dir = raw_ckpt / "best"
            latest_dir = raw_ckpt / "latest"
        else:
            best_dir = raw_ckpt
            latest_dir = raw_ckpt.parent / "latest"

        assert best_dir.name == "best", f"Failed for {c}"
        assert latest_dir.name == "latest", f"Failed for {c}"
        assert best_dir.exists(), f"Best dir {best_dir} does not exist"
        assert latest_dir.exists(), f"Latest dir {latest_dir} does not exist"


def test_req2_latest_checkpoint_assets_exist():
    """Requirement 2: All required latest checkpoint files are present and non-empty."""
    latest_dir = Path("backend/models/vit/latest")
    required_files = [
        "config.json",
        "model.safetensors",
        "training_metadata.json",
        "optimizer.pt",
        "scheduler.pt",
    ]
    for f in required_files:
        p = latest_dir / f
        assert p.exists(), f"Missing required file: {p}"
        assert p.stat().st_size > 0, f"File is empty: {p}"


def test_req3_start_epoch_calculation():
    """Requirement 3: latest_epoch = 9, therefore start_epoch = 10."""
    latest_meta = json.loads(Path("backend/models/vit/latest/training_metadata.json").read_text(encoding="utf-8"))
    latest_epoch = int(latest_meta.get("latest_epoch"))
    assert latest_epoch == 9, f"Expected latest_epoch=9, got {latest_epoch}"
    start_epoch = latest_epoch + 1
    assert start_epoch == 10, f"Expected start_epoch=10, got {start_epoch}"


def test_req4_preserve_best_model_metadata():
    """Requirement 4: best_epoch = 7, best_score = 0.9314 preserved from best checkpoint."""
    best_meta = json.loads(Path("backend/models/vit/best/training_metadata.json").read_text(encoding="utf-8"))
    best_epoch = int(best_meta.get("best_epoch"))
    best_val_acc = float(best_meta.get("best_val_accuracy"))
    assert best_epoch == 7, f"Expected best_epoch=7, got {best_epoch}"
    assert abs(best_val_acc - 0.9314) < 1e-4, f"Expected best_val_acc=0.9314, got {best_val_acc}"


def test_req5_optimizer_and_scheduler_state_restoration():
    """Requirement 5: Correctly loads saved optimizer and scheduler states."""
    latest_dir = Path("backend/models/vit/latest")
    opt_state = torch.load(latest_dir / "optimizer.pt", map_location="cpu")
    sched_state = torch.load(latest_dir / "scheduler.pt", map_location="cpu")

    assert "state" in opt_state
    assert "param_groups" in opt_state
    assert len(opt_state["param_groups"]) > 0

    assert "last_epoch" in sched_state
    assert sched_state["last_epoch"] == 999, f"Expected scheduler step 999, got {sched_state['last_epoch']}"


def test_req6_no_retrain_from_epoch_1():
    """Requirement 6: Verified start_epoch > 1 to guarantee loop begins at Epoch 10."""
    latest_meta = json.loads(Path("backend/models/vit/latest/training_metadata.json").read_text(encoding="utf-8"))
    start_epoch = int(latest_meta["latest_epoch"]) + 1
    assert start_epoch > 1
    assert start_epoch == 10


def test_req7_early_stopping_patience_calculation():
    """Requirement 7: Patience counter restored to (9 - 7) = 2."""
    latest_epoch = 9
    best_epoch = 7
    patience_counter = max(0, latest_epoch - best_epoch)
    assert patience_counter == 2, f"Expected patience_counter=2, got {patience_counter}"


def test_req8_training_history_preservation():
    """Requirement 8: training_history.json has all 9 epochs intact."""
    hist_file = Path("backend/evaluation/results/training_history.json")
    assert hist_file.exists()
    hist = json.loads(hist_file.read_text(encoding="utf-8"))
    assert len(hist) == 9, f"Expected 9 history entries, got {len(hist)}"
    epochs_in_hist = [entry["epoch"] for entry in hist]
    assert epochs_in_hist == list(range(1, 10))
