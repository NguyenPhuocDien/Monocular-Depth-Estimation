import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision


OUTPUT = Path("/kaggle/working")
probe = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ],
}

(OUTPUT / "environment_probe.json").write_text(
    json.dumps(probe, indent=2) + "\n", encoding="utf-8"
)
(OUTPUT / "experiment_log.json").write_text(
    json.dumps(
        {
            "run_id": "bts-runtime-environment-probe",
            "status": "completed",
            "created_utc": probe["created_utc"],
            "purpose": "Capture the current Kaggle BTS runtime versions",
            "hardware": probe["gpus"],
            "artifact_paths": ["environment_probe.json"],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
(OUTPUT / "metrics.jsonl").write_text("", encoding="utf-8")
(OUTPUT / "artifacts_manifest.json").write_text(
    json.dumps(
        {
            "artifacts": [
                {
                    "path": "environment_probe.json",
                    "kind": "environment_manifest",
                }
            ]
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

print(json.dumps(probe, indent=2))
