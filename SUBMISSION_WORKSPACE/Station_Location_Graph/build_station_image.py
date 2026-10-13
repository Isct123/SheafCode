                                                                                       
from pathlib import Path
import sys, json, hashlib, csv, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'Research_Release/extensions/recent_data'))
import common

BG='#F5F7FC'; INK='#192B42'; MUTED='#586B80'; BLUE='#245CED'
TEAL='#007E7B'; ORANGE='#C36524'; GRID='#DFE6F0'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK,
    'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
    'svg.fonttype':'none','pdf.fonttype':42,'font.size':13})

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def load_data():
    source=ROOT/'Research_Release/recent_data_results/availability_expansion/station_mapping.json'
    ds=common.load('delhi'); models=common.models('delhi')
    ids=list(ds['stations'].astype(str)); metadata=json.loads(source.read_text())['delhi']
    stations={str(x['location_id']):x for x in metadata if str(x['location_id']) in ids}
    assert set(stations)==set(ids) and len(ids)==5
    edges=[(ids[u],ids[v]) for u,v in models['edges']]
    assert len(edges)==9
    row=int(ds['times'].searchsorted('2026-08-13 00:00:00'))
    u,v=ids.index('17'),ids.index('50'); p=models['sheaf'].p
    k=models['edges'].index((u,v)); D=models['sheaf'].D
    M=D[k*p:(k+1)*p,u*p:(u+1)*p]
    N=-D[k*p:(k+1)*p,v*p:(v+1)*p]
    xu,xv=ds['Z'][row,u],ds['Z'][row,v]
    assert np.isfinite(np.r_[xu,xv]).all()
    a,b=M@xu,N@xv
    delta=a-b
    pair=np.zeros(len(ids)*p);pair[u*p:(u+1)*p]=xu;pair[v*p:(v+1)*p]=xv
    np.testing.assert_allclose(D[k*p:(k+1)*p]@pair,delta)
    payload={'scope':'All five stations included in the fixed Indian benchmark; not every station in the city or all international cohorts',
        'coordinate_source':str(source.relative_to(ROOT)), 'coordinate_source_sha256':sha(source),
        'stations':[stations[s] for s in ids], 'edges':edges,
        'excluded_candidate':'8118 is excluded by the saved dataset manifest',
        'projection':'Local equirectangular projection, spherical radius 6371.0088 km; origin at station 17; longitude scaled by cosine of mean station latitude',
        'edge_example':{'u':'17','v':'50','time_local':str(ds['times'][row]),
          'channels':ds['channels'].tolist(),'x_u_normalized':xu.tolist(),'x_v_normalized':xv.tolist(),
          'M':M.tolist(),'N':N.tolist(),'M_x_u':a.tolist(),'N_x_v':b.tolist(),
          'residual':delta.tolist(),'residual_norm':float(np.linalg.norm(delta)),
          'vector_plot_components':[2,3],'physical_u':ds['X'][row,u].tolist(),'physical_v':ds['X'][row,v].tolist()},
        'model_sha256':sha(common.OUT/'delhi/models.pkl'),'dataset_sha256':sha(common.OUT/'delhi/dataset.npz'),
        'verified':{'station_ids_match_dataset':True,'edge_count':9,'vector_replay':True}}
    (OUT/'SOURCE_DATA.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    with (OUT/'STATION_COORDINATES.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['location_id','source_name','lat','lon']);writer.writeheader()
        writer.writerows(payload['stations'])
    return stations,edges,payload

def geographic_graph(ax,stations,edges):
    lat0=np.mean([s['lat'] for s in stations.values()]); origin=stations['17']
    xy={k:(6371.0088*np.deg2rad(s['lon']-origin['lon'])*np.cos(np.deg2rad(lat0)),
           6371.0088*np.deg2rad(s['lat']-origin['lat'])) for k,s in stations.items()}
    ax.set_facecolor('white');ax.set_xlim(-14,17);ax.set_ylim(-9,34);ax.set_aspect('equal')
    ax.set_xticks(np.arange(-10,16,5));ax.set_yticks(np.arange(-5,35,5))
    ax.grid(color=GRID,linewidth=.8);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_edgecolor(GRID)
    for u,v in edges:
        highlight={u,v}=={'17','50'}
        ax.plot([xy[u][0],xy[v][0]],[xy[u][1],xy[v][1]],
            color=ORANGE if highlight else '#AFBDCF',lw=3.4 if highlight else 1.45,zorder=2,
            alpha=1 if highlight else .8)
    names={'17':'R K Puram','50':'Punjabi Bagh','8235':'Jahangirpuri','8239':'Okhla Phase-2','10485':'Narela'}
    offsets={'17':(-1.4,-2.2,'right'),'50':(-1.5,1.5,'right'),
             '8235':(2.3,1.7,'left'),'8239':(2.3,.2,'left'),'10485':(2.3,1.7,'left')}
    for k,s in stations.items():
        x,y=xy[k];col=TEAL if k=='17' else BLUE
        ax.scatter(x,y,s=1050,color=col,edgecolor='white',linewidth=2.6,zorder=5)
        ax.text(x,y,k,ha='center',va='center',color='white',weight='bold',
            fontsize=9 if len(k)>=4 else 12,zorder=6)
        dx,dy,ha=offsets[k]
        ax.annotate(names[k],(x,y),xytext=(x+dx,y+dy),ha=ha,va='bottom',
            fontsize=14,fontweight='bold',zorder=7,
            bbox={'facecolor':'white','edgecolor':'none','pad':2.3})
        ax.text(x+dx,y+dy-.55,f"{s['lat']:.6f}° N\n{s['lon']:.6f}° E",
            ha=ha,va='top',fontsize=9,color=MUTED,zorder=7,
            bbox={'facecolor':'white','edgecolor':'none','pad':2.2})
    ax.set_xlabel('East / west from R K Puram (km)',labelpad=10)
    ax.set_ylabel('North / south from R K Puram (km)',labelpad=10)
    ax.annotate('N',xy=(14,31.3),xytext=(14,27.5),ha='center',weight='bold',
        arrowprops={'arrowstyle':'-|>','color':INK,'lw':1.4})
    ax.plot([-12,-7],[-7,-7],color=INK,lw=2)
    ax.text(-9.5,-6.5,'5 km',ha='center',fontsize=10)
    return xy

def flow(fig,rect,example):
    ax=fig.add_axes(rect);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    def box(x,y,w,h,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.014',
            facecolor='white',edgecolor=color,linewidth=1.7))
    def txt(x,y,s,**kw):ax.text(x,y,s,ha='center',va='center',**kw)
    box(.03,.55,.38,.34,TEAL);box(.59,.55,.38,.34,BLUE)
    txt(.22,.84,'R K Puram · 17',fontsize=15,weight='bold',color=TEAL)
    txt(.78,.84,'Punjabi Bagh · 50',fontsize=15,weight='bold',color=BLUE)
    txt(.22,.765,r'$x_{17}\in\mathbb{R}^{4}$',fontsize=22)
    txt(.78,.765,r'$x_{50}\in\mathbb{R}^{4}$',fontsize=22)
    fmt=lambda vals:'['+', '.join(f'{v:.2f}' for v in vals)+']'
    for x,key in [(.22,'x_u_normalized'),(.78,'x_v_normalized')]:
        txt(x,.67,fmt(example[key]),fontsize=12)
        txt(x,.59,'Training-normalized pollutant vector',fontsize=10,color=MUTED)
    box(.32,.06,.36,.25,ORANGE)
    txt(.5,.22,r'$C_e\cong\mathbb{R}^{4}$',fontsize=24,color=ORANGE)
    txt(.5,.12,'Shared comparison space',fontsize=12)
    for start,col,label in [(.22,TEAL,r'$M_e$'),(.78,BLUE,r'$N_e$')]:
        ax.annotate('',xy=(.45 if start<.5 else .55,.325),xytext=(start,.535),
            arrowprops={'arrowstyle':'-|>','color':col,'lw':2.5})
        txt(.28 if start<.5 else .72,.40,label,fontsize=22,color=col)

def vector_plot(ax,example):
    a=np.array(example['M_x_u'])[[1,2]];b=np.array(example['N_x_v'])[[1,2]]
    ax.set_facecolor('white');ax.set_xlim(-2.45,.35);ax.set_ylim(-2.18,.38)
    ax.set_aspect('equal');ax.grid(color=GRID,lw=.8);ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_edgecolor(GRID)
    ax.axhline(0,color='#AAB8C8',lw=.8);ax.axvline(0,color='#AAB8C8',lw=.8)
    for z,col in [(a,TEAL),(b,BLUE)]:
        ax.annotate('',xy=z,xytext=(0,0),arrowprops={'arrowstyle':'-|>','color':col,'lw':2.8})
        ax.scatter(*z,s=65,color=col,zorder=5)
    ax.annotate('',xy=a,xytext=b,arrowprops={'arrowstyle':'-|>','color':ORANGE,'lw':2.3})
    ax.text(a[0]-.1,a[1]+.12,r'$M_e x_{17}$',color=TEAL,ha='center',fontsize=15,
        bbox={'facecolor':'white','edgecolor':'none','pad':2})
    ax.text(b[0]+.10,b[1]-.20,r'$N_e x_{50}$',color=BLUE,fontsize=15)
    ax.text(-1.7,-1.73,r'$\Delta_e$',fontsize=17,color=ORANGE)
    ax.set_xlabel('Comparison component 2',fontsize=11)
    ax.set_ylabel('Comparison component 3',fontsize=11)
    ax.tick_params(labelsize=10)

def save(fig,name):
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=160,facecolor=BG)
    plt.close(fig)

def main():
    stations,edges,data=load_data();example=data['edge_example']
    fig=plt.figure(figsize=(24,13.5),facecolor=BG)
    fig.text(.045,.94,'Actual station locations. The fitted sheaf graph.',fontsize=30,weight='bold')
    fig.text(.045,.895,'Five stations in the Indian benchmark • Archived coordinates • All nine fitted connections',fontsize=17,color=MUTED)
    ax=fig.add_axes([.055,.16,.435,.675]);geographic_graph(ax,stations,edges)
    fig.text(.055,.09,'Orange edge: the R K Puram–Punjabi Bagh comparison expanded at right.',fontsize=13,color=ORANGE)
    fig.text(.54,.83,'From pollutant vectors to an edge vector',fontsize=21,weight='bold')
    flow(fig,[.525,.52,.43,.29],example)
    fig.text(.54,.495,'Actual learned-map outputs',fontsize=18,weight='bold')
    ax=fig.add_axes([.545,.215,.225,.25]);vector_plot(ax,example)
    fig.text(.795,.42,r'$\Delta_e=M_e x_{17}-N_e x_{50}$',fontsize=20,color=ORANGE)
    fig.text(.795,.365,'Orange arrow: disagreement\nin the shared edge space.',fontsize=14,linespacing=1.6)
    fig.text(.795,.29,f"Full 4D residual norm: {example['residual_norm']:.3f}",fontsize=14,weight='bold')
    fig.text(.795,.235,'Plot shows components 2 and 3.\nAll four components enter the penalty.',fontsize=12,color=MUTED,linespacing=1.6)
    fig.text(.54,.145,'Snapshot: 13 August 2026, 00:00 local time.\nVector order: PM2.5, PM10, ozone, NO₂.',fontsize=13,color=MUTED,linespacing=1.6)
    fig.text(.045,.046,'Locations come from the archived OpenAQ station mapping. Connections compare readings; they do not depict pollution transport.',fontsize=13,color=MUTED)
    save(fig,'Actual_Stations_and_Sheaf_Vector')
    fig=plt.figure(figsize=(12,12),facecolor=BG)
    fig.text(.07,.94,'The actual station network',fontsize=27,weight='bold')
    fig.text(.07,.9,'Indian benchmark • Five stations • Nine fitted edges',fontsize=15,color=MUTED)
    geographic_graph(fig.add_axes([.13,.145,.77,.7]),stations,edges)
    fig.text(.07,.065,'Geographic positions from archived metadata.\nOrange: one learned comparison. Edges are not pollution routes.',fontsize=12,color=MUTED)
    save(fig,'Actual_Station_Locations')
    fig=plt.figure(figsize=(16,10),facecolor=BG)
    fig.text(.055,.92,'One sheaf edge: two stations, a shared comparison',fontsize=26,weight='bold')
    flow(fig,[.055,.45,.53,.39],example)
    vector_plot(fig.add_axes([.66,.47,.27,.36]),example)
    fig.text(.66,.37,r'$\Delta_e=M_e x_{17}-N_e x_{50}$',fontsize=24,color=ORANGE)
    fig.text(.055,.27,'Each station carries four normalized pollutant readings.\nThe learned matrices map them into the same four-dimensional edge space.',fontsize=17,linespacing=1.6)
    fig.text(.055,.14,f"Actual snapshot: {example['time_local']} local time.  Full residual norm = {example['residual_norm']:.3f}.\nThe vector plot shows only comparison components 2 and 3; all four enter the squared-disagreement penalty.",fontsize=13,color=MUTED,linespacing=1.7)
    save(fig,'Sheaf_Vector_Detail')
    (OUT/'README.md').write_text('''# Station locations and sheaf vectors

The combined image shows every station included in the saved fixed Indian benchmark, at its archived geographic coordinates, and every edge in the fitted model. The six-station acquisition list includes candidate 8118, which the benchmark manifest excludes; it is therefore not shown as a fitted node. This image does not combine unrelated international networks.

Coordinates are converted to kilometres with a local equirectangular projection, preserving their relative geographic arrangement. No street map or invented city outline is used. Names and coordinates come from `Research_Release/recent_data_results/availability_expansion/station_mapping.json`.

The highlighted edge joins station 17 (R K Puram) to station 50 (Punjabi Bagh). Both vectors are actual complete readings at 00:00 local time on 13 August 2026, normalized with the saved training means and standard deviations. The four rows of the saved spatial comparison operator supply the learned matrices M and N, including their comparison weights. Their vector difference is checked against the corresponding operator block. The arrow plot displays components 2 and 3 of a four-dimensional comparison; it is a projection, not a claim that pollutant vectors are two-dimensional. A sheaf has vector spaces at stations and edges and maps between them; the orange arrow is specifically the comparison residual in the edge space.

PNG is supplied for slides, SVG for editing, and PDF for printing. SOURCE_DATA.json contains coordinates, edges, full vectors, full matrices and source hashes; STATION_COORDINATES.csv provides the five geographic points. Regenerate using `audit_tools/python/python.exe SUBMISSION_WORKSPACE/Station_Location_Graph/build_station_image.py` from the project root. Existing manuscripts and the video kit are unchanged.
''',encoding='utf-8')
    with zipfile.ZipFile(OUT/'Station_Location_and_Sheaf_Images.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
    print(json.dumps({'stations':5,'fitted_edges':9,'vector_replay':'passed','output':str(OUT)},indent=2))

if __name__=='__main__':main()
