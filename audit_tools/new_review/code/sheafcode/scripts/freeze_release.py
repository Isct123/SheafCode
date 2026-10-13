                                                                          
                                                                       
                                                                         
                                                             
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "sheafcode_iris_2026_final"
rc = subprocess.call([sys.executable, os.path.join(ROOT, "scripts", "check_report_v3.py")], cwd=ROOT)
if rc != 0:
    print("report check failed; not freezing"); sys.exit(rc)
rc = subprocess.call([sys.executable, os.path.join(ROOT, "tests", "test_leakage.py")], cwd=ROOT, stdout=subprocess.DEVNULL)
if rc != 0:
    print("leakage tests failed; not freezing"); sys.exit(rc)
include_dirs = ["sheafcode", "scripts", "tests", "docs", "figures"]
include_files = ["README.md", "REPORT_v3.md", "FINAL_PROTOCOL.md", "CHANGELOG.md", "AI_USE_LOG.md", "NOVELTY_LOG.md", "reproduce.py", "data/README.md",
                 "results/FINAL_RESULTS.json", "REPORT_v3.pdf"]
files = []
for d in include_dirs:
    for dp, _, fn in os.walk(os.path.join(ROOT, d)):
        if "__pycache__" in dp:
            continue
        for f in fn:
            if f.endswith((".py", ".md", ".png", ".sh", ".pdf", ".svg")):
                files.append(os.path.relpath(os.path.join(dp, f), ROOT))
for f in include_files:
    if os.path.exists(os.path.join(ROOT, f)):
        files.append(f)
for d in sorted(os.listdir(os.path.join(ROOT, "results"))):
    if d.startswith("v3_"):
        for f in os.listdir(os.path.join(ROOT, "results", d)):
            if f.endswith((".json", ".csv")):
                files.append(os.path.join("results", d, f))
files = sorted(set(files))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


hashes = {f: sha(os.path.join(ROOT, f)) for f in files}
rel = [f"# {NAME}", "", f"Frozen {time.strftime('%Y-%m-%d %H:%M:%S')}. Protocol: FINAL_PROTOCOL.md. Report: REPORT_v3.md (verified by scripts/check_report_v3.py at freeze time).",
       "", "Raw data: see docs/PROVENANCE.md (mirrors, commits and hashes); the raw archives are shipped separately.", "", "## Files (SHA-256)", ""]
rel += [f"{h}  {f}" for f, h in hashes.items()]
open(os.path.join(ROOT, "RELEASE.md"), "w").write("\n".join(rel) + "\n")
main = [f for f in files if not (f.startswith("results/v3_") and f.endswith(".csv"))] + ["RELEASE.md"]
tables = [f for f in files if f.startswith("results/v3_") and f.endswith(".csv")]
for suffix, lst in (("", main), ("_result_tables", tables)):
    out = os.path.join(os.path.dirname(ROOT), f"{NAME}{suffix}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in lst:
            z.write(os.path.join(ROOT, f), os.path.join(NAME, f))
    print(f"{out}: {len(lst)} files, {os.path.getsize(out) / 1e6:.1f} MB")
