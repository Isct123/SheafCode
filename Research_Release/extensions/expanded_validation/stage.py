                                                                                   
from pathlib import Path
import sys, importlib, types
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
RECENT=ROOT/'Research_Release/extensions/recent_data'
sys.path.insert(0,str(RECENT))
import common
common.OUT=ROOT/'Research_Release/expanded_validation_results'
region=sys.argv[1]; common.regions=lambda:[region]
name=sys.argv[2]
if name in ['final_analysis','fault_partial','mathematics','faults']:
    source=(RECENT/(name+'.py')).read_text().replace("regions()+['epa_la']",'regions()')
    source=source.replace('Availability extension after strict 24h complete-network criterion yielded zero episodes;', 'Available-candidate evaluation, separate from strict whole-network evaluation;')
    if name=='mathematics':
        source=source.replace('cal.reshape(len(cal),-1)','cal.reshape(len(cal),n*p)')
        source=source.replace("test=np.flatnonzero((ds['split']==2)&full)","test=np.flatnonzero((ds['split']==2)&full) if np.isfinite(eps2) and np.isfinite(epsinf) else np.array([],dtype=int)")
        columns=['timestamp','station','functional_status','functional_lower','functional_upper','functional_covered','functional_assumptions_met','bounded_status','bounded_lower','bounded_upper','bounded_covered','bounded_assumptions_met','dual_upper','dual_lower','joint_status','joint_lower','joint_upper','joint_extra_lower','joint_decision','physical_scope']
        source=source.replace('pd.DataFrame(bounds)', 'pd.DataFrame(bounds,columns='+repr(columns)+')')
        source=source.replace("'actual_functional_cases':len(bounds)","'actual_functional_cases':len(bounds),'real_data_certificate_status':'evaluated' if len(bounds) else 'unavailable: no finite complete-vector calibration budget or no complete test vectors'")
    if name=='faults':
        source=source.replace("X=np.asarray(X);Y=np.asarray(Y);cut=", "\n        if len(valid)<2:\n            save(region+'/fault_status.json',{'status':'insufficient_full_selection_vectors','count':len(valid)});continue\n        X=np.asarray(X);Y=np.asarray(Y,dtype=int);cut=")
        source=source.replace("decoder.cutoff=quantile([decoder.features(Z[t])[2] for t in cal],.05)","\n        if len(cal)<19:\n            save(region+'/fault_status.json',{'status':'insufficient_full_calibration_vectors','count':len(cal),'required_for_finite_95_percent_quantile':19});continue\n        decoder.cutoff=quantile([decoder.features(Z[t])[2] for t in cal],.05)")
    module=types.ModuleType('expanded_'+name);module.__file__=str(RECENT/(name+'.py'))
    exec(compile(source,module.__file__,'exec'),module.__dict__)
else:module=importlib.import_module(name)
module.regions=lambda:[region]
if name=='fit':module.prepare=lambda:None
module.main()
