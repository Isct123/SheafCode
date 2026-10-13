import argparse
import importlib
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common

def main():
    parser=argparse.ArgumentParser(description='Run the main empirical experiments. Data: DATA/COHORT/dataset.npz; arrays X,Z,mu,sd,split,times,stations,channels. Split labels: training 0, selection 1, calibration 3, test 2. Channels start with PM2.5. Use the fixed paper station groups and chronological grids to reproduce its numbers. New grids are new experiments.')
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--cohorts',nargs='+')
    parser.add_argument('--stage',choices=['fit','reconstruction','random_forest','grin','ablations','intervals','alerts','thresholds','all'],default='reconstruction')
    args=parser.parse_args()
    common.OUT=args.output.resolve();common.OUT.mkdir(parents=True,exist_ok=True)
    available=sorted(p.parent.name for p in args.data.glob('*/dataset.npz'))
    chosen=args.cohorts or available
    if not chosen or not set(chosen).issubset(available):parser.error('No matching cohort datasets')
    for name in chosen:
        folder=common.OUT/name;folder.mkdir(exist_ok=True)
        src=args.data/name/'dataset.npz';target=folder/'dataset.npz'
        if src.resolve()!=target.resolve():
            if target.exists() and common.sha(target)!=common.sha(src):raise RuntimeError('Use a new output folder when changing data')
            shutil.copy2(src,target)
    common.regions=lambda:chosen
    stages=['fit','reconstruction','random_forest','ablations','intervals','alerts','thresholds'] if args.stage=='all' else [args.stage]
    if args.stage=='grin' and not args.cohorts:parser.error('Select the paper GRIN cohorts explicitly with --cohorts; GRIN is retrospective and provider selection is fixed in the paper')
    for stage in stages:
        print('Running',stage,'for',', '.join(chosen),flush=True)
        importlib.import_module(stage).main()

if __name__=='__main__':main()
