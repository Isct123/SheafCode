                                                                            

                                                                            
                                                                           
   
import sys,types,importlib,time,copy,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from common import *
sys.path.insert(0,str(ROOT.parent/'audit_tools/torch_packages'))
import torch
from sheafcode.data import haversine_km
torch.set_num_threads(1)
def official():
    base=HERE/'vendor/grin/lib';source=json.loads((base.parent/'SOURCE.json').read_text())
    for name,sha in source['files'].items():
        if digest(base.parent/name)!=sha:raise RuntimeError('Changed vendor source '+name)
                                                                            
                                                                             
    for name,path in [('lib',base),('lib.nn',base/'nn'),('lib.nn.layers',base/'nn/layers'),('lib.nn.models',base/'nn/models'),('lib.nn.utils',base/'nn/utils'),('lib.utils',base/'utils')]:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    sys.modules['lib'].epsilon=1e-8
    layer=importlib.import_module('lib.nn.layers.gril');sys.modules['lib.nn.layers'].BiGRIL=layer.BiGRIL
    return importlib.import_module('lib.nn.models.grin').GRINet
def tensor_windows(Z,ds,part,count):
    ans=[]
    for day,ix in dayblocks(ds,part,count):
        for v in range(ds.n):
            good=np.isfinite(Z[ix[24:],v])
            if good.sum()<12*ds.p:continue
            x=Z[ix].copy();mask=np.isfinite(x);target=mask.copy();target[:]=False;target[24:,v]=good
            mask[24:,v]=False;x[~mask]=0
            ans.append((day,v,ix,x,mask,target,np.nan_to_num(Z[ix])))
    return ans
def arrays(cases):
    return [torch.as_tensor(np.stack([c[j] for c in cases]),dtype=torch.bool if j in [4,5] else torch.float32) for j in [3,4,5,6]]
def evaluate(model,cases):
    model.eval();loss=[]
    with torch.no_grad():
        for start in range(0,len(cases),16):
            x,mask,target,y=arrays(cases[start:start+16]);pred=model(x,mask=mask)
            loss.extend([float(((a-b)[t]**2).mean()) for a,b,t in zip(pred,y,target)])
    return float(np.mean(loss))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--seeds',type=int,default=3);args=ap.parse_args()
    GRIN=official();records=[];training=[];settings={};start=time.time()
    progress=OUT/'graph_progress.json'
    save('graph_progress.json',{'status':'running','expected_seeds':args.seeds,'max_epochs':args.epochs,'pid':__import__('os').getpid()})
    for region in REGIONS:
        ds,m,folder=cohort(region);ds.set_transform('none');Z=ds.Z()
        dist=haversine_km(ds.coords);positive=dist[dist>0];scale=float(np.median(positive))
        adj=np.exp(-(dist/scale)**2);np.fill_diagonal(adj,0)
        train=tensor_windows(Z,ds,0,80);val=tensor_windows(Z,ds,1,8);test=tensor_windows(Z,ds,2,None)
        allowed=set(map(tuple,pd.read_csv(folder/'prolonged.csv')[['day','station']].drop_duplicates().astype(str).values))
        test=[c for c in test if (c[0],ds.stations[c[1]]) in allowed]
        for seed in range(args.seeds):
            candidates=[]
            for width in [32,64]:
                torch.manual_seed(1700+seed);rng=np.random.default_rng(1700+seed)
                model=GRIN(adj,ds.p,width,width,0.,d_emb=8);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
                bestloss=np.inf;best=None;bestepoch=0
                checkpoint=OUT/f'{region}_grin_seed{seed}_width{width}_training.pt'
                identity={'vendor_commit':json.loads((HERE/'vendor/grin/SOURCE.json').read_text())['commit'],'model_sha256':digest(folder/'model.npz'),'data_sha256':digest(folder/'dataset.npz'),'train_days':80,'validation_days':8,'width':width,'seed':seed,'max_epochs':args.epochs,'source_sha256':digest(__file__)}
                firstepoch=0;finished=False;localtraining=[]
                if checkpoint.exists():
                    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    if state['identity']!=identity:raise RuntimeError('Graph checkpoint protocol/source mismatch; preserve old checkpoint and use a new output directory')
                    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);bestloss=state['bestloss'];best=state['best'];bestepoch=state['bestepoch'];firstepoch=state['epoch'];finished=state['finished'];localtraining=state['training']
                    torch.set_rng_state(state['torch_rng']);rng.bit_generator.state=state['numpy_rng']
                for epoch in ([] if finished else range(firstepoch,args.epochs)):
                    model.train();order=rng.permutation(len(train))
                                                                                    
                    for chunk in range(0,len(order),16):
                        batch=[train[i] for i in order[chunk:chunk+16]];x,mask,target,y=arrays(batch)
                        optimizer.zero_grad();pred,aux=model(x,mask=mask)
                        loss=((pred-y)[target]**2).mean()+sum(((a-y)[target]**2).mean() for a in aux)/len(aux)
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    validation=evaluate(model,val);localtraining.append({'region':region,'seed':seed,'width':width,'epoch':epoch+1,'validation_mse':validation})
                    if validation<bestloss:bestloss=validation;best=copy.deepcopy(model.state_dict());bestepoch=epoch+1
                    finished=epoch+1-bestepoch>=7 or epoch+1==args.epochs
                    temporary=checkpoint.with_suffix('.tmp');torch.save({'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'bestloss':bestloss,'best':best,'bestepoch':bestepoch,'epoch':epoch+1,'finished':finished,'training':localtraining,'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state},temporary);temporary.replace(checkpoint)
                    save('graph_progress.json',{'status':'running','region':region,'seed':seed,'width':width,'epoch':epoch+1,'max_epochs':args.epochs,'expected_seeds':args.seeds,'elapsed_seconds':time.time()-start,'pid':__import__('os').getpid()})
                    print(region,'GRIN',seed,width,'epoch',epoch+1,'validation',round(validation,4),'elapsed',round(time.time()-start),flush=True)
                    if finished:break
                training.extend(localtraining)
                model.load_state_dict(best);candidates.append((bestloss,model,width,bestepoch))
            loss,model,width,epoch=min(candidates,key=lambda c:c[0]);model.eval()
            checkpoint=OUT/f'{region}_grin_seed{seed}.pt';torch.save(model.state_dict(),checkpoint)
            settings[region+'_'+str(seed)]={'width':width,'epoch':epoch,'validation_mse':loss,'checkpoint_sha256':digest(checkpoint),'training_cases':len(train),'validation_cases':len(val),'torch_version':torch.__version__}
            save('graph_choices.json',settings)
            with torch.no_grad():
                for chunk in range(0,len(test),16):
                    batch=test[chunk:chunk+16];x,mask,target,y=arrays(batch);pred=model(x,mask=mask).numpy()
                    for c,pr in zip(batch,pred):
                        day,v,ix=c[:3];truth=ds.X[ix[24:],v];ok=np.isfinite(truth);raw=ds.raw_from_Z(pr[24:,v][:,None],ix[24:],[v])[:,0]
                        records.append({'region':region,'seed':seed,'day':day,'station':ds.stations[v],'method':'official_grin','mse':float(np.mean(((raw-truth)/ds.std[v])[ok]**2))})
            pd.DataFrame(records).to_csv(OUT/'graph_cases.csv',index=False);pd.DataFrame(training).to_csv(OUT/'graph_training.csv',index=False)
    summaries={}
    for region in REGIONS:
        f=pd.DataFrame(records);ref=pd.read_csv(ROOT/'extension_results/strong_baselines'/f'{region}_cases.csv');ref=ref[ref['mode']=='retrospective'].copy();ref['method']=ref.method.replace({'seasonal_persistence':'seasonal'})
        summaries[region]={str(seed):{method:paired_interval(pd.concat([f[(f.region==region)&(f.seed==seed)],ref]),'official_grin',method) for method in ['temporal_sheaf','full_covariance_var','seasonal']} for seed in range(args.seeds)}
    save('graph_summary.json',{'comparisons':summaries,'settings':settings,'source_sha256':digest(__file__),'vendor_commit':json.loads((HERE/'vendor/grin/SOURCE.json').read_text())['commit'],'scope':'Official pinned bidirectional architecture, adapted outage training on raw training data, validation-selected widths/checkpoints, multiple seeds; retrospective comparisons only; no reproduction of original published dataset scores; internally examined archives.'})
    save('graph_progress.json',{'status':'complete','seeds':args.seeds,'elapsed_seconds':time.time()-start})
if __name__=='__main__':main()
