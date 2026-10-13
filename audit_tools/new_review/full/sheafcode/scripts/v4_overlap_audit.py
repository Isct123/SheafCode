                                                                   
                                                                   
                                                                                    
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sheafcode.experiments import ROOT              
from sheafcode.data import haversine_km              
import pandas as pd              

REG = ["delhi", "ncr", "mumbai", "kolkata", "hyderabad", "bengaluru", "india", "delhi2025", "bengaluru_raw"]
info = {}
for r in REG:
    f = os.path.join(ROOT, "results", f"v3_{r}", "diagnostics.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    co = pd.DataFrame(d["coords"]).set_index("id") if "id" in pd.DataFrame(d["coords"]).columns else pd.DataFrame(d["coords"], index=d["stations"])
    info[r] = dict(stations=d["stations"], names=d["names"], coords=co, span=d["span"], test_span=d["test_span"], source="raw" if r in ("delhi2025", "bengaluru_raw") else "cpcb_subindex_2017_2020")
pairs = []
for i, a in enumerate(info):
    for b in list(info)[i + 1:]:
        A, B = info[a], info[b]
        shared_ids = sorted(set(A["stations"]) & set(B["stations"]))
                                         
        ca = A["coords"][["latitude", "longitude"]].to_numpy(); cb = B["coords"][["latitude", "longitude"]].to_numpy()
        allc = pd.DataFrame(np.vstack([ca, cb]), columns=["latitude", "longitude"])
        D = haversine_km(allc)[:len(ca), len(ca):]
        near = int((D.min(axis=1) < 0.2).sum()) if D.size else 0
        ta = (pd.Timestamp(A["span"][0]), pd.Timestamp(A["span"][1])); tb = (pd.Timestamp(B["span"][0]), pd.Timestamp(B["span"][1]))
        overlap_h = max(0, (min(ta[1], tb[1]) - max(ta[0], tb[0])).total_seconds() / 3600)
        ta2 = (pd.Timestamp(A["test_span"][0]), pd.Timestamp(A["test_span"][1])); tb2 = (pd.Timestamp(B["test_span"][0]), pd.Timestamp(B["test_span"][1]))
        test_overlap_h = max(0, (min(ta2[1], tb2[1]) - max(ta2[0], tb2[0])).total_seconds() / 3600)
        pairs.append(dict(a=a, b=b, shared_station_ids=len(shared_ids), stations_within_200m=near, period_overlap_hours=int(overlap_h), test_period_overlap_hours=int(test_overlap_h),
                          same_source=A["source"] == B["source"], independent=(len(shared_ids) == 0 and near == 0 and overlap_h == 0)))
P = pd.DataFrame(pairs)
summary = dict(pairs=P.to_dict(orient="records"),
               independent_groups="Every 2017-2020 regional cohort is a subset of the national cohort (same archive, overlapping test periods): one source. delhi2025 occupies 7 of the same physical sites as the Delhi archive stations but shares no hour with any other set: an independent period, not an independent place. bengaluru_raw shares 3 sites with the Bengaluru archive stations and its 2019-2020 hours overlap the archive's calendar (its test block, Oct 2024 to Dec 2025, does not): an independent measurement path, partly overlapping period.",
               effective_independent_sources="one archive plus two partially independent replication sets; no fully independent place-and-period set")
json.dump(summary, open(os.path.join(ROOT, "results", "overlap_audit_v4.json"), "w"), indent=1)
print(P.to_string(index=False))
