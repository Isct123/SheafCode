                                                                        
                                                                       
                                                                    
                                                                   
                                                               
                                                                  
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sheafcode.experiments import ROOT              
import make_report_v3 as mk              

errors = []
cur = open(os.path.join(ROOT, "REPORT_v3.md")).read()
if cur != mk.text:
    errors.append("REPORT_v3.md differs from a fresh render of FINAL_RESULTS.json (rerun scripts/make_report_v3.py)")
for name, ok in mk.CLAIMS:
    if not ok:
        errors.append(f"claim failed: {name}")
FR = mk.FR
for reg, R in FR["regions"].items():
    for fam in ("P1", "P2", "P3_1_24h"):
        for m, v in R.get(fam, {}).items():
            if np.isfinite(v["p_holm"]) and v["p_holm"] < v["p_wilcoxon_one_sided"] - 1e-12:
                errors.append(f"{reg} {fam} {m}: Holm p below raw p")
            if not (v["diff_ci"][0] - 1e-9 <= v["mean_diff"] <= v["diff_ci"][1] + 1e-9):
                errors.append(f"{reg} {fam} {m}: mean difference outside its CI")
    d = R.get("prospective_deciles")
    if d:
        man = json.load(open(os.path.join(ROOT, "results", f"v3_{reg}", "manifest.json")))
        if d["written_before_test"] > man["finished"]:
            errors.append(f"{reg}: prospective deciles written after the run finished")
        if abs(d["ratio"] - d["rmse_risky"] / d["rmse_safe"]) > 1e-9:
            errors.append(f"{reg}: decile ratio inconsistent")
    b = R.get("bound_check")
    if b and b["violations"] != 0:
        errors.append(f"{reg}: bound violations recorded ({b['violations']})")
    if not (0 <= R["identifiable_fraction"] <= 1):
        errors.append(f"{reg}: identifiable fraction out of range")
if errors:
    print("CHECK FAILED"); [print(" -", e) for e in errors]; sys.exit(1)
print(f"REPORT_v3.md verified: render identical, {len(mk.CLAIMS)} claims hold, {len(FR['regions'])} regions consistent")
