"""Supersede Run 01 Session 2 without performing any training."""

import json
from datetime import datetime, timezone


print(json.dumps({
    "run_id": "bts_kitti_densenet161_full_run_01",
    "session": 2,
    "status": "superseded_by_owner_before_run_02_2xt4",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "optimizer_steps_performed_by_this_version": 0,
    "next_session_authorized": False,
}, indent=2))
