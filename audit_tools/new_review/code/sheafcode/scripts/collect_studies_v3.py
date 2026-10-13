                                                                   
                                                                     
                                                                        
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheafcode.experiments import ROOT              

ap = argparse.ArgumentParser()
ap.add_argument("--regions", default="delhi,ncr,mumbai,kolkata,hyderabad,bengaluru,india,delhi2025,bengaluru_raw")
args = ap.parse_args()
fr = os.path.join(ROOT, "results", "FINAL_RESULTS.json")
final = json.load(open(fr)) if os.path.exists(fr) else {}
studies = final.get("studies", {})
for reg in args.regions.split(","):
    out = os.path.join(ROOT, "results", f"v3_{reg}")
    S = studies.get(reg, {})
    for name in ("ablation", "faults", "structure", "extras", "intervals"):
        f = os.path.join(out, f"{name}.json")
        if os.path.exists(f):
            S[name] = json.load(open(f))
    if S:
        studies[reg] = S
final["studies"] = studies
json.dump(final, open(fr, "w"), indent=1, default=float)
print("studies merged for", sorted(studies), "->", fr)
