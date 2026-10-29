"""Tamper demo: python demo_tamper.py ..\\Images\\Te-gl_1.jpg"""
import json
import sys

import config
from ledger.factory import get_ledger
from pipeline import run_case
from storage.offchain import resolve
from verification.verify import verify_case

img = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_IMAGE
ledger = get_ledger()
out = run_case(img, ledger=ledger)
cid = out["case_id"]
print(json.dumps(out["result"], indent=2, ensure_ascii=False))

def show(title):
    v = verify_case(cid, ledger)
    print(f"\n== {title}: {v['overall_fr']} ==")
    for a in v["artifacts"]:
        print(f"  {a['artifact_type']:<18} {a['status_fr']}")

show("Before tampering")
target = resolve(next(a["storage_ref"] for a in out["artifacts"]
                      if a["artifact_type"] == "PREDICTION"))
original_bytes = target.read_bytes()
target.write_bytes(original_bytes.replace(b'"glioma"', b'"notumor"', 1) + b" ")
show("After modifying PREDICTION.json")
target.write_bytes(original_bytes)
show("After restoring the file")
