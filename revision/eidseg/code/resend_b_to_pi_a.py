"""Re-send M5_B's raw result for images whose M5_B -> Pi_A upload was lost (2026-10-29).

Run on Pi_B, which received the same M5_B payloads, in the folder its ingest server runs from (not ~/eidseg there):
    ssh pi@192.168.0.93 'cd "$(readlink -f /proc/$(pgrep -of pi_ingest_server)/cwd)" && pwd && python3 -' \
        < revision/eidseg/code/resend_b_to_pi_a.py

The payload is M5_B's original meta from Pi_B's raw log, minus Pi_B's own `_pi` block, plus a `_resent` note,
with the image Pi_B saved. Pi_A then has a,b,c for the image and merges it as usual.
"""
import glob
import json
import os
import subprocess
import tempfile
from datetime import datetime

IDS = ["002721__ed62aa1df3.jpg", "002722__6f2be919cf.jpg"]
PI_A = "http://192.168.0.60:8000/ingest_m5"
SAVE_DIR = os.environ.get("PI_SAVE_DIR", "./tier2_eidseg")


def role_of(m):  # same rule as pi_ingest_server._role_from_device
    d = str(m.get("role", "unknown")).strip().lower()
    d = d[3:] if d.startswith("m5_") else d
    return d if d in ("a", "b", "c") else d[-1:]


raws = sorted(glob.glob(os.path.join(SAVE_DIR, "raw_ingest__*.jsonl")))
print("raw logs:", raws)
found = {}
for p in raws:
    for line in open(p):
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("image_id") in IDS and role_of(m) == "b" and m.get("status") == "ok":
            found[m["image_id"]] = m          # last ok record wins

for i in IDS:
    m = found.get(i)
    img = os.path.join(SAVE_DIR, "images", i)
    if m is None or not os.path.exists(img):
        print(f"SKIP {i}: meta found={m is not None}, image found={os.path.exists(img)}")
        continue
    m.pop("_pi", None)
    m["_resent"] = {"from": "pi_b raw_ingest", "reason": "M5_B->Pi_A upload lost",
                    "at": datetime.now().astimezone().isoformat(timespec="seconds")}
    print(f"{i}: device={m.get('device')} cat={m.get('cat')} sev={m.get('sev')} conf={m.get('conf')}")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(m, f, ensure_ascii=False)
    out = subprocess.run(["curl", "-s", "-m", "30", "-F", f"meta=<{f.name}",
                          "-F", f"image=@{img};type=image/jpeg", PI_A], capture_output=True, text=True)
    os.unlink(f.name)
    print("  Pi_A ->", out.stdout or out.stderr)
