                                                                                       
from pathlib import Path
import hashlib,json,sys,subprocess,math
import numpy as np
import pandas as pd
from scipy.linalg import solve
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
import build_video_kit as kit

ROOT=kit.ROOT;OUT=kit.OUT
sys.path.insert(0,str(ROOT/'Research_Release/extensions/recent_data'))
import common
POSITIONS={'17':(515,377),'50':(245,572),'10485':(923,545),'8239':(804,755),'8235':(336,783)}
CMAP=LinearSegmentedColormap.from_list('current_influence',['#CCDEF7','#F4B263','#C46520'])

def compute_influence():
    ds=common.load('delhi');models=common.models('delhi');model=models['sheaf']
    ids=list(ds['stations'].astype(str));v=ids.index('8235');p=model.p;n=model.n
    cases=pd.read_csv(common.OUT/'delhi/outage_cases.csv',dtype={'station':str})
    days=cases[(cases.method=='sheaf_causal')&(cases.station=='8235')&cases.primary].day
    blocks={day:ix for day,ix in common.blocks(ds,2)}
    day=next(day for day in days if np.isfinite(ds['Z'][blocks[day][24],np.arange(n)!=v]).all())
    ix=blocks[day];prefix=ds['Z'][ix[:25]].copy();prefix[-1,v]=np.nan
    Q=model.precision(len(prefix)).toarray();flat=prefix.ravel()
    U=np.flatnonzero(~np.isfinite(flat));O=np.flatnonzero(np.isfinite(flat))
    target=24*n*p+np.arange(v*p,(v+1)*p)
    H=-solve(Q[np.ix_(U,U)],Q[np.ix_(U,O)],assume_a='pos')
    B=H[np.searchsorted(U,target)]
    baseline=model.solve(prefix)[-1,v]
    np.testing.assert_allclose(B@flat[O],baseline,rtol=1e-9,atol=1e-9)
    scores=[];blocks_export={}
    for u,station in enumerate(ids):
        mask=(O//(n*p)==24)&((O//p)%n==u)
        block=B[:,mask];value=float(np.linalg.norm(block,'fro')) if u!=v else 0.
        scores.append(value);blocks_export[station]=block.tolist()
        if u!=v:
            for j in range(p):
                altered=prefix.copy();altered[-1,u,j]+=1e-5
                derivative=(model.solve(altered)[-1,v]-baseline)/1e-5
                np.testing.assert_allclose(derivative,block[:,j],rtol=1e-6,atol=1e-8)
    percent=np.array(scores)/sum(scores)*100
    result={'target_station':'8235','day':day,'hour':0,'prediction':'First outage hour; all four target channels hidden',
            'scope':'Available current readings at surviving stations; history and other inputs are held fixed',
            'normalization':'Training-normalized pollutant coordinates',
            'definition':'Frobenius norm of each station block of the exact decoder Jacobian, divided by the sum of surviving-station block norms',
            'interpretation':'Relative reconstruction sensitivity in this context; unsigned magnitudes, not coefficients of a convex weighted average',
            'stations':ids,'edges':[[ids[u],ids[w]] for u,w in models['edges']],
            'relative_current_influence_percent':dict(zip(ids,percent.tolist())),
            'raw_block_norms':dict(zip(ids,scores)),'jacobian_blocks':blocks_export,
            'prediction_normalized':baseline.tolist(),'jacobian_prediction_replay_passed':True,
            'finite_difference_checks':(n-1)*p,
            'model_sha256':common.sha(common.OUT/'delhi/models.pkl'),
            'dataset_sha256':common.sha(common.OUT/'delhi/dataset.npz'),
            'positions':'Schematic layout; fitted edges are exact'}
    (OUT/'GRAPH_INFLUENCE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame([{'station':s,'relative_current_influence_percent':float(percent[i]),
                  'current_status':'offline' if s=='8235' else 'reporting','jacobian_block_norm':scores[i]} for i,s in enumerate(ids)]).to_csv(OUT/'GRAPH_INFLUENCE.csv',index=False)
    explanation=f'''# What the graph colours mean

The graph uses the exact nine station pairs stored in the fitted Indian benchmark model. Its positions are schematic. Station 8235 is the simulated target outage; no physical incident is claimed.

The colours describe the first outage-hour solve on {day}, with all four target channels hidden and actual pre-outage history retained. For each surviving station, we take the Frobenius norm of the linear decoder's Jacobian block connecting its currently available, training-normalized pollutant readings to the four normalized target predictions. We divide these nonnegative magnitudes by their sum to obtain the displayed relative percentages. This is a sensitivity visualization, not a new reconstruction experiment.

History and naturally missing coordinates affect the solve but are not part of the displayed percentage denominator. Colour values apply to this selected context; they are not universal station weights, pollution-source contributions or probabilities of a correct estimate. The decoder has signed matrix coefficients, so these percentages do not make it a convex weighted average.

The Jacobian was checked against the actual saved model prediction and {result['finite_difference_checks']} finite-difference perturbations. Exact station pairs, matrices, percentages, prediction context and source hashes are saved in GRAPH_INFLUENCE.json. GRAPH_INFLUENCE.csv provides the values for editing.

The black node denotes a station that has stopped reporting current readings. Its earlier observations remain available. Moving dots in the animation illustrate information used in reconstruction; they do not depict pollution transport. The fitted relationships continue to constrain the unknown values during the outage.
'''
    (OUT/'GRAPH_COLOUR_GUIDE.md').write_text(explanation,encoding='utf-8')
    return result

def draw_graph(info,scene=3,phase='weighted',name=False):
    if scene==1:
        title='When a station fails,\nlocal pollution information is lost.'
        subtitle=''
    else:
        title='The fitted graph. Influence shown in colour.'
        subtitle='Colour measures sensitivity to current surviving readings for this estimate.'
    fig,ax=kit.base(scene,title,subtitle)
    weights=info['relative_current_influence_percent'];maxweight=max(weights.values())
    offline=phase!='live'
    for a,b in info['edges']:
        x,y=POSITIONS[a];xx,yy=POSITIONS[b]
        target='8235' in (a,b)
        ax.plot([x,xx],[y,yy],color=kit.ORANGE if offline and target else '#B5C3D7',
                lw=4 if offline and target else 2.7,zorder=0,alpha=.8)
    for station,(x,y) in POSITIONS.items():
        missing=station=='8235' and offline
        fill='#080A0D' if missing else CMAP(weights[station]/maxweight) if phase=='weighted' else kit.BLUE
        ax.add_patch(Circle((x,y),58,facecolor=fill,edgecolor=kit.WHITE,lw=4,zorder=4))
        colour=kit.WHITE if missing or phase!='weighted' or weights[station]>20 else kit.INK
        kit.text(ax,x,y,station,size=24,bold=True,color=colour,ha='center',va='center',zorder=5)
        if missing:
            kit.text(ax,x,y+76,'OFFLINE',size=20,bold=True,color=kit.INK,ha='center')
        elif phase=='weighted':
            kit.text(ax,x,y+76,f'{weights[station]:.1f}%',size=21,bold=True,color=kit.INK,ha='center')
    kit.box(ax,1100,360,724,501)
    if phase=='live':
        kit.text(ax,1140,401,'01  Monitoring local pollution',size=26,bold=True,color=kit.BLUE)
        kit.text(ax,1140,483,'Local readings help us see\nvariation between locations.',size=27,bold=True)
        kit.text(ax,1140,639,'Information that can support\npollution warnings and responses.',size=24,color=kit.MUTED)
    elif phase=='down':
        kit.text(ax,1140,401,'02  Station 8235 goes offline',size=25,bold=True)
        kit.text(ax,1140,492,'A gap in local information.',size=28,bold=True)
        kit.text(ax,1140,613,'Is pollution below or\nabove the reference?',size=27,color=kit.MUTED)
    else:
        kit.text(ax,1140,401,'03  Estimate using survivors',size=26,bold=True)
        kit.text(ax,1140,475,'Relative current-reading influence',size=22,bold=True)
        grad=np.linspace(0,1,256)[None,:]
        ax.imshow(grad,extent=(1140,1740,588,551),cmap=CMAP,aspect='auto',zorder=6)
        kit.text(ax,1140,609,'0%',size=20,color=kit.MUTED)
        kit.text(ax,1740,609,f'{maxweight:.1f}%',size=20,color=kit.MUTED,ha='right')
        kit.text(ax,1140,682,'Warmer colour → more influence\non this example’s reconstruction.',size=23,color=kit.MUTED)
        kit.text(ax,1140,785,'History included • Future data excluded.',size=20,bold=True,color=kit.TEAL)
    if not name:
        kit.text(ax,96,927,'Fitted connections • Schematic layout • First outage-hour example: 13 August 2026',size=19,color=kit.MUTED)
    if name:
        kit.box(ax,96,892,1728,87,fill=kit.INK)
        kit.text(ax,133,906,'[FULL NAME]  •  Grade [GRADE]',size=27,color=kit.WHITE,bold=True)
    return fig

def animation(info):
    folder=OUT/'animation';folder.mkdir(exist_ok=True)
    pictures={}
    for phase in ['live','down','weighted']:
        fig=draw_graph(info,scene=1,phase=phase)
        path=folder/f'graph_{phase}.png';fig.savefig(path,dpi=100,facecolor=kit.BG)
        fig.savefig(path.with_suffix('.svg'),facecolor=kit.BG);plt.close(fig)
        pictures[phase]=Image.open(path).convert('RGB')
    sys.path.insert(0,str(ROOT/'audit_tools/video_packages'))
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    fps=24;seconds=10;target=folder/'Station_Outage_Animation.mp4'
    command=[ffmpeg,'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),
             '-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    thumbs=[]
    with (folder/'encoder.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for frame in range(fps*seconds):
                t=frame/fps;phase='live' if t<4 else 'down' if t<6 else 'weighted'
                image=pictures[phase].copy();draw=ImageDraw.Draw(image)
                if phase=='live':
                    for x,y in POSITIONS.values():
                        r=63+6*math.sin(t*4)
                        draw.ellipse((x-r,y-r,x+r,y+r),outline=kit.BLUE,width=2)
                elif phase=='down':
                    x,y=POSITIONS['8235'];r=66+8*math.sin((t-4)*5)
                    draw.ellipse((x-r,y-r,x+r,y+r),outline=kit.ORANGE,width=3)
                else:
                    for index,station in enumerate(['50','8239','10485']):
                        a=np.array(POSITIONS[station]);b=np.array(POSITIONS['8235'])
                        fraction=((t-6)*.45+index*.31)%1
                        x,y=a+(b-a)*(.14+.72*fraction)
                        draw.ellipse((x-8,y-8,x+8,y+8),fill=kit.ORANGE,outline=kit.WHITE,width=2)
                process.stdin.write(image.tobytes())
                if frame%8==0:thumbs.append(image.resize((960,540)))
            process.stdin.close();returncode=process.wait()
        except Exception:
            process.kill();process.wait();raise
    assert returncode==0,(folder/'encoder.log').read_text()
    thumbs[0].save(folder/'Station_Outage_Animation.gif',save_all=True,append_images=thumbs[1:],duration=333,loop=0)
                                                                                       
    reader=imageio_ffmpeg.read_frames(str(target),pix_fmt='rgb24');metadata=next(reader)
    frames=sum(1 for _ in reader)
    assert metadata['size']==(1920,1080) and frames==fps*seconds
    assert abs(metadata['duration']-seconds)<.1
    result={'status':'passed','video':target.name,'seconds':seconds,'frames':frames,'fps':fps,'resolution':[1920,1080],
            'audio':'None; record your own narration','ffmpeg_version':imageio_ffmpeg.get_ffmpeg_version(),
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Verified 10-second station-outage animation, 240 decoded frames.',flush=True)
    return target

if __name__=='__main__':animation(compute_influence())
