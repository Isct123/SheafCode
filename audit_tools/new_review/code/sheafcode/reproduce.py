                                                                  

                                                                                  
                                                                                     
                                               
                                                                                         

                                                                    
                                                                       
                                                                     
                                                               
                                                            
                                                                   
                                                                     
                                                                  
                                                      
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
ALL = "bengaluru,bengaluru_raw,delhi2025,hyderabad,kolkata,mumbai,delhi,ncr,india"
ap = argparse.ArgumentParser()
ap.add_argument("--regions", default=ALL)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--check-only", action="store_true")
ap.add_argument("--skip-tests", action="store_true")
ap.add_argument("--skip-studies", action="store_true")
args = ap.parse_args()
if args.quick:
    args.regions = "bengaluru"
t0 = time.time()


def run(cmd, log=None):
    print(f"$ {' '.join(cmd)}", flush=True)
    if log:
        with open(log, "w") as f:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    else:
        rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        print(f"FAILED (exit {rc}): {' '.join(cmd)}", flush=True); sys.exit(rc)


os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
if not args.check_only:
    if not args.skip_tests:
        for t in ("tests/test_sheaf_synthetic.py", "tests/test_cpcb_index.py", "tests/test_theory.py", "tests/test_leakage.py"):
            if os.path.exists(os.path.join(ROOT, t)):
                run([sys.executable, t])
    for reg in args.regions.split(","):
        extra = ["--max_days", "20", "--n_random", "10"] if args.quick else []
        if not os.path.exists(os.path.join(ROOT, "results", f"v3_{reg}", "prolonged_results.csv")):
            run([sys.executable, "scripts/run_v3.py", "--region", reg] + extra, log=os.path.join(ROOT, "logs", f"v3_{reg}.log"))
        else:
            print(f"results/v3_{reg} exists, not rerun (delete it or pass --tag to run_v3.py to rerun)")
        if not args.skip_studies:
            sd, ns = {"india": ("12", "20"), "ncr": ("20", "30")}.get(reg, ("30", "40"))
            if args.quick:
                sd, ns = "8", "40"
            run([sys.executable, "scripts/ablation_v3.py", "--region", reg, "--max_days", sd, "--n_stations", ns], log=os.path.join(ROOT, "logs", f"v3_{reg}_ablation.log"))
            run([sys.executable, "scripts/faults_v3.py", "--region", reg, "--max_days", "4" if args.quick else "20", "--n_stations", "12"], log=os.path.join(ROOT, "logs", f"v3_{reg}_faults.log"))
            run([sys.executable, "scripts/extras_v3.py", "--region", reg], log=os.path.join(ROOT, "logs", f"v3_{reg}_extras.log"))
            run([sys.executable, "scripts/intervals_v3.py", "--region", reg], log=os.path.join(ROOT, "logs", f"v3_{reg}_intervals.log"))
            run([sys.executable, "scripts/structure_v3.py", "--region", reg, "--max_days", sd, "--n_stations", ns], log=os.path.join(ROOT, "logs", f"v3_{reg}_structure.log"))
    run([sys.executable, "scripts/analyze_v3.py", "--regions", args.regions])
    run([sys.executable, "scripts/collect_studies_v3.py", "--regions", args.regions])
if os.path.exists(os.path.join(ROOT, "scripts", "check_report_v3.py")) and os.path.exists(os.path.join(ROOT, "REPORT_v3.md")):
    run([sys.executable, "scripts/check_report_v3.py"])
print(f"done in {time.time() - t0:.0f}s")
