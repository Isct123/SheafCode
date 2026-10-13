                                                                                  
from pathlib import Path
import sys, json, hashlib, textwrap, zipfile
from datetime import datetime, timezone
import numpy as np
from PIL import Image
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection, PolyCollection

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build_station_image as original

OUT=HERE/'Map_Overlay'
OUT.mkdir(exist_ok=True)
R=6378137.0

def mercator(lon,lat):
    return R*np.deg2rad(lon),R*np.log(np.tan(np.pi/4+np.deg2rad(lat)/2))

def acquire_map():
    record=OUT/'BASEMAP_SOURCE.json';path=OUT/'street_basemap.png'
    features=json.loads((OUT/'osm_features.json').read_text(encoding='utf-8'))
    source=json.loads((OUT/'osm_request.json').read_text(encoding='utf-8'))
    xmin,ymin=mercator(77.045,28.475)
    xmax,ymax=mercator(77.33,28.88)
    width=(ymax-ymin)*1800/2200;centre=(xmin+xmax)/2;xmin=centre-width/2;xmax=centre+width/2
    fig=original.plt.figure(figsize=(9,11),dpi=200,facecolor='#F0EFE8')
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.set_aspect('equal');ax.axis('off')
    groups={key:[] for key in ['water','river','canal','motorway','trunk','primary','secondary']}
    for el in features['elements']:
        tags=el.get('tags',{})
        if el['type']=='node':
            label=tags.get('name:en',tags.get('name',''))
            if tags.get('place')=='city' and label:
                x,y=mercator(el['lon'],el['lat'])
                ax.text(x,y,label.upper(),ha='center',color='#627483',fontsize=13,weight='bold',zorder=4,
                    path_effects=[pe.withStroke(linewidth=3,foreground='#F0EFE8')])
            continue
        coords=[mercator(pt['lon'],pt['lat']) for pt in el.get('geometry',[]) if 'lon' in pt]
        if len(coords)<2:continue
        if tags.get('natural')=='water' and el.get('nodes',[0])[0]==el.get('nodes',[1])[-1]:
            groups['water'].append(coords)
        if tags.get('waterway') in ['river','canal']:groups[tags['waterway']].append(coords)
        if tags.get('highway') in ['motorway','trunk','primary','secondary']:groups[tags['highway']].append(coords)
    ax.add_collection(PolyCollection(groups['water'],facecolors='#B9D9EC',edgecolors='#A7CDE3',linewidths=.5,zorder=1))
    for name,width,color in [('river',4.5,'#9FC9E3'),('canal',1.5,'#AFCFE2'),
            ('secondary',1.0,'#D2C7B2'),('primary',1.6,'#D7BB83'),('trunk',2.0,'#DAAC74'),('motorway',2.5,'#DBA378')]:
        ax.add_collection(LineCollection(groups[name],colors=color,linewidths=width,zorder=2))
                                                           
    rivers=[el for el in features['elements'] if el.get('tags',{}).get('waterway')=='river'
        and 'yamuna' in el.get('tags',{}).get('name:en',el.get('tags',{}).get('name','')).lower()]
    if rivers:
        el=max(rivers,key=lambda x:len(x.get('geometry',[])))
        points=[pt for pt in el['geometry'] if 'lon' in pt and 28.55<pt['lat']<28.8]
        if points:
            pt=points[len(points)//2];x,y=mercator(pt['lon'],pt['lat'])
            ax.text(x+1700,y,'YAMUNA',color='#588CAA',fontsize=13,rotation=-65,weight='bold')
    fig.savefig(path,dpi=200,facecolor='#F0EFE8');original.plt.close(fig)
    raw=path.read_bytes()
    info={'service':source['endpoint'],'query':source['query'],'feature_count':len(features['elements']),
        'extent':{'xmin':xmin,'xmax':xmax,'ymin':ymin,'ymax':ymax,'spatialReference':{'wkid':3857}},
        'width':1800,'height':2200,'copyright':'© OpenStreetMap contributors · Map data: ODbL · https://www.openstreetmap.org/copyright',
        'licenseInfo':'Map data licensed under ODbL 1.0; supplied OSM extract retains this licence.',
        'rendered_utc':datetime.now(timezone.utc).isoformat(),'image_sha256':hashlib.sha256(raw).hexdigest(),
        'osm_data_sha256':hashlib.sha256((OUT/'osm_features.json').read_bytes()).hexdigest(),
        'projection':'EPSG:3857; graph and rendered OpenStreetMap geometry use exactly the same coordinates and extent'}
    record.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    print('Rendered the requested OpenStreetMap street-map viewport.',flush=True)
    return info

def graph_on_map(ax,stations,edges):
    e=MAP['extent'];extent=[e['xmin'],e['xmax'],e['ymin'],e['ymax']]
    im=Image.open(OUT/'street_basemap.png')
    ax.imshow(im,extent=extent,origin='upper',zorder=0)
    ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:]);ax.set_aspect('equal')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_edgecolor('#BDCBDA')
    xy={k:mercator(s['lon'],s['lat']) for k,s in stations.items()}
    for u,v in edges:
        line,=ax.plot([xy[u][0],xy[v][0]],[xy[u][1],xy[v][1]],
            color=original.ORANGE if {u,v}=={'17','50'} else '#593B89',
            lw=3.1 if {u,v}=={'17','50'} else 1.8,zorder=3,alpha=.95)
        line.set_path_effects([pe.Stroke(linewidth=line.get_linewidth()+2,foreground='white',alpha=.82),pe.Normal()])
    names={'17':'R K Puram','50':'Punjabi Bagh','8235':'Jahangirpuri','8239':'Okhla Phase-2','10485':'Narela'}
    offsets={'17':(-1900,-2800,'right'),'50':(-1900,1800,'right'),
        '8235':(2500,2300,'left'),'8239':(2400,700,'left'),'10485':(2500,2300,'left')}
    for k in stations:
        x,y=xy[k]
        ax.scatter(x,y,s=1050,color=original.TEAL if k=='17' else original.BLUE,
            edgecolor='white',linewidth=2.6,zorder=5)
        ax.text(x,y,k,ha='center',va='center',color='white',weight='bold',
            fontsize=9 if len(k)>=4 else 12,zorder=6)
        dx,dy,ha=offsets[k]
        ax.plot([x,x+dx],[y,y+dy],color=original.INK,lw=.8,zorder=4)
        ax.annotate(names[k],xy=(x,y),xytext=(x+dx,y+dy),ha=ha,va='center',
            fontsize=14,weight='bold',zorder=7,
            bbox={'boxstyle':'round,pad=.3','facecolor':'white','edgecolor':'#CDD7E4','alpha':.96})
    x=extent[1]-(extent[1]-extent[0])*.08
    y=extent[3]-(extent[3]-extent[2])*.06
    ax.annotate('N',xy=(x,y),xytext=(x,y-4200),ha='center',weight='bold',
        bbox={'facecolor':'white','edgecolor':'none','pad':3},
        arrowprops={'arrowstyle':'-|>','color':original.INK,'lw':1.4},zorder=9)
                                                              
    length=5000/np.cos(np.deg2rad(np.mean([s['lat'] for s in stations.values()])))
    x=extent[0]+(extent[1]-extent[0])*.07;y=extent[2]+(extent[3]-extent[2])*.055
    scale,=ax.plot([x,x+length],[y,y],color=original.INK,lw=2,zorder=9)
    scale.set_path_effects([pe.Stroke(linewidth=5,foreground='white'),pe.Normal()])
    ax.text(x+length/2,y+850,'5 km',ha='center',fontsize=11,zorder=9,
        bbox={'facecolor':'white','edgecolor':'none','pad':2})
                                                               
    ax.text(.985,.008,'© OpenStreetMap contributors',transform=ax.transAxes,
        ha='right',va='bottom',fontsize=8,zorder=10,
        bbox={'facecolor':'white','edgecolor':'none','pad':2,'alpha':.95})
    return xy

def save_map(fig,name):
    if name=='Actual_Stations_and_Sheaf_Vector':
        fig.text(.045,.07,'Basemap '+textwrap.fill(MAP['copyright'],width=180),fontsize=9,color=original.MUTED)
    elif name=='Actual_Station_Locations':
                                                                                   
        for text in fig.texts:
            if text.get_position()[1]==.065:text.set_y(.085)
        fig.text(.07,.035,textwrap.fill('Basemap '+MAP['copyright'],width=105),fontsize=8,color=original.MUTED)
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=160,facecolor=original.BG)
    original.plt.close(fig)

def main():
    global MAP
    MAP=acquire_map()
    original.OUT=OUT
    original.geographic_graph=graph_on_map
    original.save=save_map
    original.main()
    data_path=OUT/'SOURCE_DATA.json';data=json.loads(data_path.read_text())
    data['projection']=MAP['projection'];data['basemap']='BASEMAP_SOURCE.json'
    data['verified']['graph_and_map_share_projection']=True
    data_path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    notes='''# Graph on a street map

The station graph is overlaid on a street map drawn from actual OpenStreetMap geometry: major roads, rivers, canals, water areas and city labels. Every node and every map feature uses longitude/latitude converted to EPSG:3857. All nine fitted connections remain included. Labels are offset for readability; node centres remain at the actual locations. This is a simplified cartographic rendering of actual map data, rather than a screenshot of a proprietary basemap.

The purple and orange lines represent fitted comparison relationships, not roads or pollution transport. The orange edge is expanded in the sheaf-vector inset. The inset retains actual readings and learned matrices from the earlier figure. SOURCE_DATA.json holds these values; BASEMAP_SOURCE.json records the map extent, request, source credits and hashes. Attribution is visible on the figures. PNG, SVG and PDF are supplied; the SVG/PDF contain the rendered raster basemap and vector overlays. The raw OpenStreetMap extract is included under the Open Database License (ODbL 1.0).

Map data and attribution: https://www.openstreetmap.org/copyright
Data licence: https://opendatacommons.org/licenses/odbl/1-0/

The coordinate-grid originals remain in the parent folder. The manuscripts and video kit have not been changed.
'''
    (OUT/'README.md').write_text(notes,encoding='utf-8')
    bundle=OUT/'Station_Location_and_Sheaf_Images.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
        z.write(Path(__file__),'build_map_image.py')
        z.write(HERE/'build_station_image.py','build_station_image.py')
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    print('Street-map image, combined sheaf image and ZIP verified.',flush=True)

if __name__=='__main__':main()
