import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adaptations" / "bts_runtime" / "pytorch"))
sys.path.insert(0, str(ROOT / "tools"))

from checkpoint_retention import (  # noqa: E402
    checkpoint_classes,
    emit_scientific_checkpoint,
    rotate_recovery_checkpoint,
    validate_checkpoint,
)
from kitti_checkpoint_handoff import inspect_checkpoint, stage_checkpoint  # noqa: E402


def checkpoint(step: int) -> dict:
    return {
        "global_step": step,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {0: {"step": torch.tensor(1.0)}}, "param_groups": []},
        "best_eval_measures_higher_better": torch.zeros(3),
        "best_eval_measures_lower_better": torch.ones(6),
        "best_eval_steps": [0] * 9,
    }


class CheckpointRetentionTests(unittest.TestCase):
    def test_milestone_geometry(self):
        cases = [
            (28948, None, ()),
            (28949, 5, ("milestone",)),
            (57899, 10, ("milestone",)),
            (115799, 20, ("milestone",)),
            (173699, 30, ("milestone",)),
            (231599, 40, ("milestone",)),
            (289499, 50, ("milestone", "final")),
        ]
        for step, epoch, classes in cases:
            with self.subTest(step=step):
                self.assertEqual(
                    checkpoint_classes(step, 5790, 50, {5, 10, 20, 30, 40, 50}),
                    (epoch, classes),
                )

    def test_epoch_50_is_one_physical_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = emit_scientific_checkpoint(
                checkpoint(289499), root, 50, ("milestone", "final")
            )
            self.assertEqual(receipt["classes"], ["milestone", "final"])
            self.assertEqual(len(list((root / "checkpoints").rglob("*.pth"))), 1)
            self.assertIn("final", receipt["path"])

    def test_recovery_rotation_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rotate_recovery_checkpoint(checkpoint(499), root)
            manifest = rotate_recovery_checkpoint(checkpoint(999), root)
            self.assertEqual(manifest["latest"]["global_step"], 999)
            self.assertEqual(manifest["previous"]["global_step"], 499)
            written = json.loads(
                (root / "checkpoints" / "recovery" / "recovery_manifest.json").read_text()
            )
            self.assertEqual(written["latest"]["global_step"], 999)

    def test_production_schema_fails_closed(self):
        invalid = checkpoint(2)
        del invalid["best_eval_steps"]
        with self.assertRaises(KeyError):
            validate_checkpoint(invalid)

    def test_handoff_production_and_transport_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.pth"
            torch.save(checkpoint(2), valid)
            staged = stage_checkpoint(valid, root / "stage", "run", "production")
            self.assertEqual(staged["latest"]["global_step"], 2)

            minimal = root / "minimal.pth"
            value = checkpoint(3)
            for key in (
                "best_eval_measures_higher_better",
                "best_eval_measures_lower_better",
                "best_eval_steps",
            ):
                del value[key]
            torch.save(value, minimal)
            with self.assertRaises(KeyError):
                inspect_checkpoint(minimal, "production")
            self.assertEqual(
                inspect_checkpoint(minimal, "transport-preflight")["global_step"], 3
            )


if __name__ == "__main__":
    unittest.main()
