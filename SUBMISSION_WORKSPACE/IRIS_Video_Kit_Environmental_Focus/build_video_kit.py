                                                                                           
from pathlib import Path
import hashlib
import html
import json
import os
import sys
import re
import zipfile
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'Research_Release/matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps, ImageDraw

BG='#F6F8FC'; INK='#14263E'; MUTED='#586A80'; BLUE='#245DEB'; TEAL='#007B77'
ORANGE='#C66022'; LIGHT='#E7EDF7'; WHITE='#FFFFFF'; RED='#B74442'
FONTS=Path('C:/Windows/Fonts')
REG=FontProperties(fname=str(FONTS/'segoeui.ttf'))
BOLD=FontProperties(fname=str(FONTS/'segoeuib.ttf'))
plt.rcParams['svg.fonttype']='none'
plt.rcParams['pdf.fonttype']=42

SCENES=[
 {'id':'01_station_outage','title':'The environmental problem','seconds':8,
  'speech':'Hi, I’m [FULL NAME], in grade [GRADE]. When an air-quality station fails, pollution continues—but local information disappears.'},
 {'id':'02_three_questions','title':'The research question','seconds':9,
  'speech':'My question was: can we recover enough information to decide whether pollution crosses an alert threshold, without measuring everything?'},
 {'id':'03_learned_comparisons','title':'CCA and causal sheaf reconstruction','seconds':11,
  'speech':'I built SheafCode using sheaf theory. Canonical correlation analysis learns how stations’ pollutant vectors relate, and past readings help reconstruct missing values.'},
 {'id':'04_outage_experiment','title':'Recoverability, joint certification and verified bounds','seconds':15,
  'speech':'I derived conditions for unique recovery and joint threshold certification under stated error limits, with verified bounds that remain valid even if fault searches stop early. I checked the algorithms against independent calculations.'},
 {'id':'05_reconstruction_result','title':'Research design and reconstruction evidence','seconds':15,
  'speech':'I tested twenty-four-hour outages on later data kept separate from training and tuning. Across seventy-seven matched Indian station-days, normalized error was sixteen point five percent lower than tuned causal VAR. External results were mixed.'},
 {'id':'06_decisions_and_queries','title':'Robust repair, stability theorem and adaptive queries','seconds':23,
  'speech':'I also developed minimum-cost measurement plans that tolerate noise and corruption. My stability theorem shows exactly how much archived patterns can change before a fixed plan loses its guarantee. Adaptive plans choose the next station from earlier answers. In ten archive settings, deciding the alert needed twenty-five percent fewer queries than identifying the full pattern.'},
 {'id':'07_takeaway','title':'Intended environmental benefit','seconds':7,
  'speech':'This could support pollution warnings during outages while focusing measurement effort. Field validation is still needed.'},
]

def text(ax,x,y,s,size=24,color=INK,bold=False,ha='left',va='top',**kwargs):
    return ax.text(x,y,s,fontsize=size,fontproperties=BOLD if bold else REG,
                   color=color,ha=ha,va=va,linespacing=1.25,**kwargs)

def box(ax,x,y,w,h,fill=WHITE,edge=None,r=22,lw=1.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                              facecolor=fill,edgecolor=edge or fill,linewidth=lw))

def arrow(ax,start,end,color=BLUE,lw=3,style='-|>',dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=22,
                                linewidth=lw,color=color,linestyle='--' if dashed else '-'))

def base(i,title,subtitle=''):
    fig=plt.figure(figsize=(19.2,10.8),dpi=100,facecolor=BG)
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,1920);ax.set_ylim(1080,0);ax.axis('off')
    text(ax,96,42,'SheafCode',size=21,bold=True)
    text(ax,96,83,'Pollution information during sensor outages',size=14,color=MUTED)
    text(ax,1820,47,f'{i:02} / 07',size=16,color=MUTED,ha='right')
    text(ax,96,151,title,size=38,bold=True)
    if subtitle:text(ax,96,267,subtitle,size=23,color=MUTED)
    ax.plot([96,1824],[1011,1011],color=LIGHT,lw=1.2)
    text(ax,96,1031,'IRIS project video  •  Original research diagrams',size=13,color=MUTED)
    return fig,ax

def station(ax,x,y,label,missing=False,small=False):
    radius=50 if small else 65
    ax.add_patch(Circle((x,y),radius,facecolor=WHITE,edgecolor=ORANGE if missing else BLUE,lw=3))
    ax.plot([x-18,x+18],[y+10,y+10],color=INK,lw=4)
    ax.plot([x,x],[y-22,y+10],color=INK,lw=4)
    for offset in [0,11]:
        ax.add_patch(matplotlib.patches.Arc((x,y-24),26+offset*2,22+offset*2,
                        theta1=200,theta2=340,color=ORANGE if missing else BLUE,lw=2))
    if missing:
        ax.plot([x-23,x+23],[y-28,y+24],color=ORANGE,lw=5)
        ax.plot([x+23,x-23],[y-28,y+24],color=ORANGE,lw=5)
    text(ax,x,y+radius+18,label,size=20,bold=True,ha='center')

def draw1():
    fig,ax=base(1,'When a station goes silent,\nlocal information is lost.')
    points=[(335,454),(755,439),(520,739),(975,710)]
    for a,b in [(0,1),(0,2),(1,2),(1,3),(2,3)]:
        ax.plot([points[a][0],points[b][0]],[points[a][1],points[b][1]],color=LIGHT,lw=8,zorder=0)
    for j,(x,y) in enumerate(points):station(ax,x,y,f'Station {chr(65+j)}',missing=j==3)
    box(ax,1180,414,585,380)
    text(ax,1224,460,'Surviving stations',size=24,bold=True,color=BLUE)
    text(ax,1224,510,'still have useful readings.',size=24)
    text(ax,1224,605,'Missing station',size=24,bold=True,color=ORANGE)
    text(ax,1224,654,'What can we still recover?',size=24)
    box(ax,96,886,1728,87,fill=INK)
    text(ax,133,906,'[FULL NAME]  •  Grade [GRADE]',size=27,color=WHITE,bold=True)
    return fig

def draw2():
    fig,ax=base(2,'Pollution information supports action.',
                'Local conditions matter when deciding whether a warning is needed.')
    cards=[(96,'01','MONITOR','Know local conditions.','A city average can miss\nvariation between stations.',BLUE),
           (690,'02','CHECK','Does pollution cross\nthe reference?','A threshold connects readings\nto a decision.',TEAL),
           (1284,'03','RESPOND','Inform warnings and responses.','Useful information can support\nprotective action.',ORANGE)]
    for x,num,label,heading,body,color in cards:
        box(ax,x,366,540,487)
        text(ax,x+38,408,num,size=43,color=color,bold=True)
        text(ax,x+38,488,label,size=24,color=color,bold=True)
        text(ax,x+38,560,heading,size=23,bold=True)
        text(ax,x+38,691,body,size=21,color=MUTED)
    text(ax,96,900,'Project aim: inform pollution responses when monitoring stations stop reporting.',size=22,color=MUTED)
    text(ax,96,945,'Our reference thresholds are illustrative. Official alert recovery was not tested.',size=19,color=MUTED)
    return fig

def draw3():
    fig,ax=base(3,'Learn relationships between stations.',
                'Related stations need not report identical concentrations.')
    for x,label,values in [(96,'Station A','Pollutant vector A'),(96,'Station B','Pollutant vector B')]:
        y=383 if label.endswith('A') else 641
        box(ax,x,y,430,174)
        text(ax,x+31,y+26,label,size=27,bold=True)
        text(ax,x+31,y+89,values,size=23,color=MUTED)
    box(ax,735,454,487,264,fill=BLUE)
    text(ax,978,493,'SHARED COMPARISON',size=22,color=WHITE,bold=True,ha='center')
    text(ax,978,562,'Transform → compare',size=27,color=WHITE,bold=True,ha='center')
    text(ax,978,625,'Maps learned on training data',size=19,color=WHITE,ha='center')
    arrow(ax,(530,470),(730,528));arrow(ax,(530,730),(730,650))
    text(ax,586,412,'Map A',size=21,color=BLUE)
    text(ax,586,761,'Map B',size=21,color=BLUE)
    arrow(ax,(1224,584),(1378,584))
    box(ax,1384,434,440,313)
    text(ax,1425,478,'Spatial relationships',size=23,bold=True)
    text(ax,1425,550,'+ earlier readings',size=24,bold=True)
    text(ax,1425,640,'→ missing-value estimate',size=21,color=BLUE)
    box(ax,96,886,1728,87,fill='#E5F1EC')
    text(ax,132,909,'Prediction time: use available history and current readings. Exclude future observations.',size=22,color=TEAL,bold=True)
    return fig

def draw4():
    fig,ax=base(4,'Mathematical results behind the decision.',
                'Check every reading allowed by the observations and declared error limits.')
    box(ax,96,339,1728,95,fill='#E5F1EC')
    text(ax,132,364,'RECOVERABILITY  •  Anchors and remaining constraints determine when readings are unique.',size=23,bold=True,color=TEAL)
    for x,label,values,colours,body in [
        (96,'ALL ALLOWED READINGS AGREE',[20,25,30],[TEAL]*3,'The declared model settles\nthe threshold decision.'),
        (990,'OPPOSITE DECISIONS REMAIN',[30,40],[TEAL,ORANGE],'Choose new readings that\ndistinguish the opposite decisions.')]:
        box(ax,x,468,834,355)
        text(ax,x+39,496,label,size=24,bold=True,color=TEAL if x==96 else ORANGE)
        left=x+99;right=x+744;y=632;threshold=left+645*(35-15)/30
        ax.plot([left,right],[y,y],color=LIGHT,lw=14,solid_capstyle='round')
        ax.plot([threshold,threshold],[y-66,y+73],color=MUTED,lw=2,linestyle='--')
        text(ax,threshold,y-84,'35',size=23,bold=True,ha='center')
        for value,colour in zip(values,colours):
            pos=left+645*(value-15)/30
            ax.add_patch(Circle((pos,y),12,facecolor=colour,zorder=5))
            text(ax,pos,y+31,str(value),size=24,bold=True,ha='center',color=colour)
        text(ax,x+39,735,body,size=23,bold=True)
    box(ax,96,868,1728,86,fill='#E5F1EC')
    text(ax,132,891,'VERIFIED BOUNDS  •  Certificates can remain valid when the fault search stops early.',size=24,bold=True,color=TEAL)
    text(ax,96,969,'Illustrative PM2.5 values and reference, in µg/m³. Certificates depend on the stated assumptions.',size=17,color=MUTED)
    return fig

def draw5(gain,pm):
    fig,ax=base(5,'Lower error on the fixed Indian benchmark.',
                '24-hour outages • Later tests kept separate from training / tuning • 77 station-days, 16 dates')
    text(ax,96,365,f'{gain:.2f}%',size=88,bold=True,color=BLUE)
    text(ax,103,516,'lower normalized RMSE',size=30,bold=True)
    text(ax,103,575,'than the tuned causal VAR model',size=24,color=MUTED)
    text(ax,103,649,'95% paired seven-day interval',size=21,color=MUTED)
    text(ax,103,691,'12.62% to 20.16%',size=26,bold=True)
    box(ax,1014,370,810,366)
    text(ax,1050,404,'Relative normalized RMSE',size=26,bold=True)
    for y,value,label,color in [(520,100,'VAR',MUTED),(651,100-gain,'SheafCode',BLUE)]:
        text(ax,1050,y-31,label,size=23,bold=True)
        ax.add_patch(Rectangle((1240,y-30),value*4.3,52,facecolor=color))
        text(ax,1240+value*4.3+15,y-30,f'{value:.2f}'.rstrip('0').rstrip('.'),size=24,bold=True,color=color)
    text(ax,1049,769,'Index: VAR = 100. Lower is better.',size=20,color=MUTED)
    box(ax,96,863,1728,106,fill='#E5F1EC')
    text(ax,132,894,f'PM2.5 alone: {pm:.2f}% lower physical RMSE',size=28,bold=True,color=TEAL)
    text(ax,1779,899,'Separate metric',size=19,color=TEAL,ha='right')
    return fig

def draw6():
    fig,ax=base(6,'Robust queries. Exact stability. Adaptive plans.',
                'Measure what separates opposite alert decisions under the stated assumptions.')
    for x,heading,formula,body,colour in [
        (96,'ROBUST REPAIR',r'$2f+1$','Separating stations per opposite pair\nfor up to f corrupted queries.',TEAL),
        (690,'STABILITY THEOREM',r'$\rho<r(R)$','Exact pattern-change limit\nwithin the label-preserving range.',ORANGE),
        (1284,'ADAPTIVE QUERIES','Answer → next query','Choose the next station from\nresponses already received.',BLUE)]:
        box(ax,x,366,540,258)
        text(ax,x+31,397,heading,size=23,bold=True,color=colour)
        text(ax,x+31,455,formula,size=36 if x<1284 else 27,bold=True,color=colour)
        text(ax,x+31,541,body,size=20,color=MUTED)
    box(ax,96,659,1728,190)
    text(ax,132,691,'25%',size=65,bold=True,color=TEAL)
    text(ax,341,708,'fewer queries to decide the alert',size=27,bold=True)
    text(ax,133,801,'10 archive settings; both tasks feasible.',size=21,color=MUTED)
    for y,label,count,color in [(729,'Full pattern',24,MUTED),(810,'Alert decision',18,TEAL)]:
        text(ax,1060,y-26,label,size=20,bold=True)
        for i in range(count):
            ax.add_patch(Circle((1274+i*17,y-15),5.5,facecolor=color))
        text(ax,1735,y-26,str(count),size=26,bold=True,color=color,ha='right')
    text(ax,96,889,'Theory: bounded clean error, declared corruption budget and independent perturbation boxes.',size=21,color=MUTED)
    text(ax,96,936,'25% result: 18 vs 24 total queries with no query corruption. Field cost savings were not measured.',size=19,color=MUTED)
    return fig

def draw7():
    fig,ax=base(7,'Support informed pollution responses.',
                'Intended benefit: fewer information gaps when reporting stations fail.')
    for x,label,color in [(96,'MONITOR',BLUE),(690,'CHECK',TEAL),(1284,'FOCUS QUERIES',ORANGE)]:
        box(ax,x,400,540,176,fill=color)
        text(ax,x+270,455,label,size=35,bold=True,color=WHITE,ha='center')
    text(ax,96,654,'Potential use: better-informed warnings during outages.',size=28,bold=True)
    text(ax,96,725,'Performance varies. Guarantees require the stated assumptions.',size=26,color=MUTED)
    text(ax,96,795,'Next step: prospective field validation.',size=27,color=MUTED)
    box(ax,96,887,1728,88,fill=INK)
    text(ax,132,910,'Pollution, emissions and health-risk reductions have not been measured.',size=24,color=WHITE)
    return fig

def make_pptx(paths):
                                                                                       
    ns='http://schemas.openxmlformats.org';D=ns+'/drawingml/2006/main';P=ns+'/presentationml/2006/main';R=ns+'/officeDocument/2006/relationships'
    header='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    def relationships(items):
        return header+'<Relationships xmlns="'+ns+'/package/2006/relationships">'+''.join(
            f'<Relationship Id="{i}" Type="{R}/{t}" Target="{target}"/>' for i,t,target in items)+'</Relationships>'
    def group():return '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    with zipfile.ZipFile(OUT/'SheafCode_Video_Visuals.pptx','w',zipfile.ZIP_DEFLATED) as z:
        content=[('presentation','presentation'),('slideMasters/slideMaster1','slideMaster'),('slideLayouts/slideLayout1','slideLayout'),('notesMasters/notesMaster1','notesMaster')]
        content += [(f'slides/slide{i}','slide') for i in range(1,8)]+[(f'notesSlides/notesSlide{i}','notesSlide') for i in range(1,8)]
        z.writestr('[Content_Types].xml',header+'<Types xmlns="'+ns+'/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="png" ContentType="image/png"/><Default Extension="gif" ContentType="image/gif"/><Default Extension="xml" ContentType="application/xml"/>'+''.join(f'<Override PartName="/ppt/{name}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.{typ}+xml"/>' for name,typ in content)+'<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/></Types>')
        z.writestr('_rels/.rels',relationships([('rId1','officeDocument','ppt/presentation.xml')]))
        z.writestr('ppt/presentation.xml',header+f'<p:presentation xmlns:a="{D}" xmlns:r="{R}" xmlns:p="{P}"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:notesMasterIdLst><p:notesMasterId r:id="rId2"/></p:notesMasterIdLst><p:sldIdLst>'+''.join(f'<p:sldId id="{255+i}" r:id="rId{i+2}"/>' for i in range(1,8))+'</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle/></p:presentation>')
        z.writestr('ppt/_rels/presentation.xml.rels',relationships([('rId1','slideMaster','slideMasters/slideMaster1.xml'),('rId2','notesMaster','notesMasters/notesMaster1.xml')]+[(f'rId{i+2}','slide',f'slides/slide{i}.xml') for i in range(1,8)]))
        theme=header+f'<a:theme xmlns:a="{D}" name="SheafCode"><a:themeElements><a:clrScheme name="SheafCode">'+''.join(f'<a:{k}><a:srgbClr val="{v}"/></a:{k}>' for k,v in [('dk1','14263E'),('lt1','FFFFFF'),('dk2','586A80'),('lt2','F6F8FC'),('accent1','245DEB'),('accent2','007B77'),('accent3','C66022'),('accent4','B74442'),('accent5','E7EDF7'),('accent6','586A80'),('hlink','245DEB'),('folHlink','007B77')])+'</a:clrScheme><a:fontScheme name="Segoe"><a:majorFont><a:latin typeface="Segoe UI"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Segoe UI"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Standard"><a:fillStyleLst>'+('<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'*3)+'</a:fillStyleLst><a:lnStyleLst>'+('<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:prstDash val="solid"/></a:ln>'*3)+'</a:lnStyleLst><a:effectStyleLst>'+('<a:effectStyle><a:effectLst/></a:effectStyle>'*3)+'</a:effectStyleLst><a:bgFillStyleLst>'+('<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'*3)+'</a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>'
        z.writestr('ppt/theme/theme1.xml',theme)
        z.writestr('ppt/slideMasters/slideMaster1.xml',header+f'<p:sldMaster xmlns:a="{D}" xmlns:r="{R}" xmlns:p="{P}"><p:cSld><p:spTree>'+group()+'</p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')
        z.writestr('ppt/slideMasters/_rels/slideMaster1.xml.rels',relationships([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','theme','../theme/theme1.xml')]))
        z.writestr('ppt/slideLayouts/slideLayout1.xml',header+f'<p:sldLayout xmlns:a="{D}" xmlns:r="{R}" xmlns:p="{P}" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>'+group()+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
        z.writestr('ppt/slideLayouts/_rels/slideLayout1.xml.rels',relationships([('rId1','slideMaster','../slideMasters/slideMaster1.xml')]))
        z.writestr('ppt/notesMasters/notesMaster1.xml',header+f'<p:notesMaster xmlns:a="{D}" xmlns:r="{R}" xmlns:p="{P}"><p:cSld><p:spTree>'+group()+'</p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:notesStyle/></p:notesMaster>')
        z.writestr('ppt/notesMasters/_rels/notesMaster1.xml.rels',relationships([('rId1','theme','../theme/theme1.xml')]))
        for i,path in enumerate(paths,1):
            media=OUT/'animation/Station_Outage_Animation.gif' if i==1 else path
            suffix=media.suffix
            z.write(media,f'ppt/media/image{i}{suffix}')
            pic=f'<p:pic><p:nvPicPr><p:cNvPr id="2" name="{html.escape(SCENES[i-1]["title"])}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
            intro=''
            if i==1:
                intro='<p:sp><p:nvSpPr><p:cNvPr id="3" name="EDIT NAME AND GRADE"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="824230" y="5690000"/><a:ext cx="10000000" cy="480000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="14263E"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap="none" lIns="0" tIns="0" rIns="0" bIns="0"/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="2700" b="1"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:latin typeface="Segoe UI"/></a:rPr><a:t>[FULL NAME]  •  Grade [GRADE]</a:t></a:r><a:endParaRPr lang="en-US"/></a:p></p:txBody></p:sp>'
            z.writestr(f'ppt/slides/slide{i}.xml',header+f'<p:sld xmlns:a="{D}" xmlns:r="{R}" xmlns:p="{P}"><p:cSld><p:spTree>'+group()+pic+intro+'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')
            z.writestr(f'ppt/slides/_rels/slide{i}.xml.rels',relationships([('rId1','slideLayout','../slideLayouts/slideLayout1.xml'),('rId2','image',f'../media/image{i}{suffix}'),('rId3','notesSlide',f'../notesSlides/notesSlide{i}.xml')]))
            note=html.escape(SCENES[i-1]['speech'])
            z.writestr(f'ppt/notesSlides/notesSlide{i}.xml',header+f'<p:notes xmlns:a="{D}" xmlns:r="{R}" xmlns:p="{P}"><p:cSld><p:spTree>'+group()+f'<p:sp><p:nvSpPr><p:cNvPr id="2" name="Narration"/><p:cNvSpPr txBox="1"/><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:t>{note}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>')
            z.writestr(f'ppt/notesSlides/_rels/notesSlide{i}.xml.rels',relationships([('rId1','notesMaster','../notesMasters/notesMaster1.xml'),('rId2','slide',f'../slides/slide{i}.xml')]))
    with zipfile.ZipFile(OUT/'SheafCode_Video_Visuals.pptx') as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.endswith(('.xml','.rels')):ET.fromstring(z.read(name))

def documents(gain,pm):
    words=sum(len(re.findall(r"\b[\w]+(?:[-’][\w]+)*\b",s['speech'])) for s in SCENES)
    seconds=sum(s['seconds'] for s in SCENES)
    script=['# SheafCode — short IRIS video script','',
        f'Approximately {words} spoken words before replacing placeholders: about {words/145*60:.0f} seconds at 145 words/minute, '
        f'or {words/140*60:.0f} seconds at 140 words/minute, before pauses. The visual cues allow {seconds} seconds. Rehearse with a timer; keep the complete final video at or below 90 seconds.','',
        'Replace [FULL NAME] and [GRADE]. If there are two researchers, use “We are [NAME 1] and [NAME 2], in grades [GRADE 1] and [GRADE 2]” and change I/my to we/our. Recheck timing after making that change.','',
        '## Read this aloud','']+[s['speech']+'\n' for s in SCENES]
    script += ['## Visual cues','',
        f'Times below describe an {seconds}-second visual sequence. Advance on the matching sentence. Do not add a silent opening or closing that pushes the complete recording over 90 seconds.','',
        '| Nominal time | Visual | Spoken cue |','|---|---|---|']
    start=0
    for s in SCENES:
        script.append(f'| {start:02}–{start+s["seconds"]:02} s | {s["id"]}.png | {s["speech"]} |');start+=s['seconds']
    script += ['','Pronunciation: “SheafCode” = “sheef code”; VAR can be spoken as “V-A-R.” '
               'Say “canonical correlation analysis” in full; pronounce CCA as “C-C-A” if discussing the visual. '
               'The narration rounds 16.4806% to 16.5%; the visual retains 16.48%.','',
               'The 25% query result is a separate offline finite-bank comparison: 18 versus 24 total queries '
               'across ten settings where both tasks were feasible, with no query corruption. '
               'It is not a field cost reduction or a reconstruction error reduction.']
    script += ['','## Main contributions covered — preparation notes, not extra narration','',
        'The spoken version explains what the results establish; it does not recite proofs. These notes make the coverage explicit for rehearsal and questions.','',
        '| Contribution | Where it appears | Technical meaning |',
        '|---|---|---|',
        '| CCA-learned spatial comparisons and causal reconstruction | Scene 3 | Learn station-to-edge restriction maps from training data; combine them with earlier observations and a regularized within-station covariance prior. |',
        '| Anchors and exact recoverability | Scene 4: “conditions for unique recovery” | Under the invertible-edge and whole-station assumptions, characterize ambiguity in completely unobserved components and the minimum station support of a homogeneous solution. |',
        '| Joint threshold certification | Scene 4 | Optimize the maximum pollutant score over every feasible fault support. Certify only when all admissible states have the same decision; abstain otherwise. |',
        '| Verified numerical bounds | Scene 4: “verified bounds” | Correct approximate dual weights using finite variable ranges and evaluate exactly or with outward rounding. |',
        '| Anytime fault search | Scene 4: “fault searches stop early” | Maintain valid global bounds using a covering branch-and-bound tree, even before the search finishes. |',
        '| Minimum-cost robust decision repair | Scene 6 | Every opposite-decision pair requires at least 2f+1 queried separators under the clean-error allowances; an exact witness oracle supports multicover optimization. |',
        '| Stability theorem and budget frontier | Scene 6 | The q-th separation margin gives the exact fixed-plan radius within the label-preserving range; endpoints are strict. Optimize that radius under a measurement budget. |',
        '| Exact adaptive planning | Scene 6 | Use remaining pattern-specific corruption budgets in a finite-state dynamic program; choose the next station from previous answers. |',
        '| Empirical checks and environmental aim | Scenes 4–7 | Separate computational checks; chronological training, selection and test roles; 24-hour simulated outages; 77 matched benchmark cases; finite-bank query comparisons; mixed external performance and a future field study. |','',
        '## Why this version is stronger for review — not extra narration','',
        'The opening gives a concrete monitoring problem and a research question. “I built,” “I derived,” “I tested,” and “I checked” identify the work being presented. The reconstruction result now states the outage duration, the separate later test data, the matched sample size and the causal baseline. The stability theorem is explained through what it establishes, rather than a string of theorem names. The separate 25% query result explains why deciding an alert can need fewer readings than identifying the full pattern.','',
        'The phrase “independent calculations” means separate computational implementations or exact checks within the research. It does not claim external replication, independent peer review, or a field trial. Match the narration to your actual contribution and be prepared to explain how the checks were performed.','',
        'This emphasis follows the [IRIS submission guidelines](https://iris.exstemplar.com/guidelines.html) on the problem, innovation, methodology and results. It improves the clarity and evidence presented; it is not a prediction of the selection outcome.','',
        '## How to describe novelty accurately','',
        'CCA, sheaf theory, linear programming, branch-and-bound and multicover are existing tools. The contribution is the learned reconstruction approach together with the problem-specific certification, repair, stability and adaptive results developed in the manuscript. Do not claim that CCA itself was invented here or that this is its first-ever application without a separate literature review. The manuscript calls the stability result a theorem; the other mathematical results are propositions. “My mathematical results” covers both naturally.','',
        'CCA maps are a studied component, not a proven universal improvement: identity maps outperform the CCA version on the reported Indian ablation. The certificates depend on the stated model and error assumptions. Archive-based planning has additional membership and perturbation assumptions, and the direct-reading guard is a separate later-date experiment.','',
        'IRIS asks for a clear project explanation and originality, with a maximum 90-second video; its rules do not require formal theorem statements or proofs in the narration. Source: [IRIS rules and guidelines](https://iris.exstemplar.com/guidelines.html), checked 13 October 2026.']
    script += ['','Play animation/Station_Outage_Animation.mp4 during the first scene. Station 8235 turns black after four seconds. '
               'Scene 3 uses the static coloured graph. Colour percentages describe relative current-reading sensitivity '
               'for one example, with history also used. See GRAPH_COLOUR_GUIDE.md.']
    (OUT/'SCRIPT.md').write_text('\n'.join(script)+'\n',encoding='utf-8')
    (OUT/'NARRATION.txt').write_text('\n\n'.join(s['speech'] for s in SCENES)+'\n',encoding='utf-8')
    guidelines='''# Recording and submission guide

The current official [registration portal](https://register.irisnationalfair.org/) and [rules page](https://iris.exstemplar.com/guidelines.html), checked on 13 October 2026, specify a video of no more than 90 seconds, submitted as a public or unlisted YouTube URL. The current portal takes a URL rather than an MP4 upload. The rules page is labelled “2025–26 Edition”; the registration portal identifies the current 2026 competition. Both agree on the 90-second limit and YouTube format.

The official rules prohibit disclosing your school, city or state in the video or presentation. Keep those details out of clothing logos, the background, screen shares and captions. The linked [earlier official submission instructions](https://www.irisnationalfair.org/images/Guidelines_IRIS_Submissions.pdf) call for a full-name/grade introduction and explain the project through innovation, methodology and conclusions. They allow visuals and specify that only the project's student researchers should appear. Their older file-upload size advice is superseded by the current portal's YouTube-link requirement.

## Use the kit

1. Fill in the name and grade in NARRATION.txt and the editable name/grade text box on PowerPoint slide 1. For a team, use the team introduction in SCRIPT.md.
2. Rehearse the script while changing slides at sentence boundaries. Aim for a finished recording around 85–88 seconds, keeping the complete recording below the 90-second limit. The 88-second cues are a guide; check the actual duration after filling in names. Read only the narration, not the contribution table or preparation notes. If the rehearsal exceeds the limit, omit “I checked the algorithms against independent calculations”; keep those checks in the supporting notes. Do not speed through the mathematical explanation to fit the time.
3. Film a brief face-to-camera introduction, then show the visuals while you continue speaking. You can keep a small live camera view of yourself alongside them. Your voice should explain the work; the silent deck alone is not the completed submission video.
4. Open SheafCode_Video_Visuals.pptx in PowerPoint, or open PRESENTATION.html locally in a browser and press F for full screen. Use the arrow keys to advance. Browser presenter notes are hidden by default; N toggles them during rehearsal. All artwork is also supplied as 1920×1080 PNGs and scalable SVGs. VISUALS.pdf has one landscape page per scene.
5. Record horizontally in a quiet, well-lit setting. Keep the camera steady and check that your voice is clearly audible. Avoid displaying the Downloads folder, manuscript file path or desktop notifications during recording.
6. Check the actual final duration, including the introduction and any closing frame. Upload the student-narrated video to YouTube as unlisted or public, then confirm the link plays while signed out before pasting it into the IRIS portal. Uploading is your final step; this kit does not publish anything.

## What each visual shows

The opening animation and coloured graph use the exact fitted station connections, with a schematic layout. Their colours show relative sensitivity to current surviving readings for one outage-hour example; past readings also enter the estimate. See GRAPH_COLOUR_GUIDE.md for the definition and verification. Station 8235 turns black when its current readings disappear. Moving dots illustrate reconstruction information. The threshold and query-selection examples are schematic teaching diagrams. The normalized-error index is calculated from the fixed Indian benchmark, with VAR set to 100. The 25% query figure is a separate finite-bank experiment. The last frame keeps the limits visible: mixed performance, conditional guarantees and the need for prospective field validation.

## Environmental emphasis

The video now leads with the loss of local pollution information, connects monitoring to warnings and responses, and explains why choosing measurements for the threshold decision can be useful. This context follows WHO's discussion of [air-quality information and public action](https://www.who.int/data/gho/data/themes/air-pollution/who-air-quality-database?hl=en-GB) and [air-quality information and health advisories](https://www.who.int/docs/default-source/air-pollution-documents/faq-personal-interventions-and-risks.pdf). These sources support the monitoring rationale; they do not validate SheafCode's predictions or error limits.

The environmental benefit is an intended application: support informed pollution responses during outages. Demonstrated outcomes remain simulated reconstruction accuracy and conditional measurement planning. Official regulatory alert recovery, pollution reduction, emissions reduction and improved health have not been established. Lower measurement counts concern the tested bank settings; practical field cost savings remain unmeasured.

## Use the animation

PRESENTATION.html automatically plays the 10-second MP4 on scene 1. The same clip is available in animation/Station_Outage_Animation.mp4 for insertion into your video editor. A GIF and three still frames are included too. PowerPoint slide 1 contains the animated GIF; use slideshow mode to see the animation. For full-resolution video editing, use the supplied 1920×1080 MP4. Introduce yourself on camera, and advance to the coloured graph during the colour-explanation sentence. The clip has no audio so you can record your own narration. Its time is part of your full recording, not an additional introduction to append beyond the time limit.

The narration now explicitly covers CCA-learned comparisons, recoverability, joint certification, verified anytime bounds, robust repair, the stability theorem and adaptive plans. SCRIPT.md maps these sentences to each proposition and theorem in the manuscript. CCA is an existing method applied to learn maps, and its ablation does not show a universal advantage over identity maps.

The script uses the fixed benchmark already in your supplied manuscript. It does not claim first-ever use of CCA, demonstrated health benefits, deployed alert recovery or field cost savings. Additional recent-data positive highlights are not substituted for the fixed benchmark.
'''
    (OUT/'RECORDING_GUIDE.md').write_text(guidelines,encoding='utf-8')
    scenes=json.dumps(SCENES,ensure_ascii=False)
    web='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SheafCode video visuals</title>
<style>html,body{margin:0;background:#14263e;color:white;font:18px system-ui;height:100%;overflow:hidden}main{height:100%;display:flex;align-items:center;justify-content:center}img,video{max-width:100vw;max-height:100vh;object-fit:contain}#tools{position:fixed;bottom:10px;right:15px;background:#14263ef0;padding:10px;border-radius:8px;font-size:14px;opacity:0;transition:opacity .2s}body:hover #tools{opacity:1}button{padding:7px 12px;border:0;border-radius:6px;margin:3px;cursor:pointer}#notes{display:none;position:fixed;bottom:66px;left:8%;right:8%;padding:20px;background:#14263ef5;border-radius:12px;line-height:1.6}</style>
<main><img id="frame" alt=""><video id="animation" muted playsinline src="animation/Station_Outage_Animation.mp4"></video></main><aside id="notes"></aside><div id="tools"><span id="count"></span> <button onclick="step(-1)">←</button><button onclick="step(1)">→</button><button onclick="replay()">Replay animation</button><button onclick="full()">Full screen (F)</button><button onclick="notes()">Notes (N)</button></div>
<script>const scenes=__SCENES__;let i=0;function show(){const s=scenes[i],v=document.getElementById('animation'),img=document.getElementById('frame');img.src='visuals/'+s.id+'.png';img.alt=s.title;img.style.display=i===0?'none':'block';v.style.display=i===0?'block':'none';if(i===0){v.currentTime=0;v.play().catch(()=>{img.style.display='block';v.style.display='none'})}else v.pause();document.getElementById('notes').textContent=s.speech;document.getElementById('count').textContent=(i+1)+' / '+scenes.length}function replay(){i=0;show()}function step(d){i=Math.max(0,Math.min(scenes.length-1,i+d));show()}function full(){if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen()}function notes(){const e=document.getElementById('notes');e.style.display=e.style.display==='block'?'none':'block'}window.addEventListener('keydown',e=>{if(['ArrowRight',' ','PageDown'].includes(e.key)){e.preventDefault();step(1)}if(['ArrowLeft','PageUp'].includes(e.key))step(-1);if(e.key.toLowerCase()==='f')full();if(e.key.toLowerCase()==='n')notes()});show();</script></html>'''.replace('__SCENES__',scenes)
    (OUT/'PRESENTATION.html').write_text(web,encoding='utf-8')
    (OUT/'scenes.json').write_text(json.dumps({'word_count':words,'nominal_seconds':seconds,'scenes':SCENES},indent=2,ensure_ascii=False),encoding='utf-8')
    return words

def main():
    OUT.mkdir(exist_ok=True);(OUT/'visuals').mkdir(exist_ok=True)
    source=ROOT/'Research_Release/recent_data_results/delhi/reconstruction_summary.json'
    benchmark=json.loads(source.read_text())['primary_every_third_day']
    gain=benchmark['var_causal']['gain_percent'];pm=benchmark['pm25_physical']['var_causal']['gain_percent']
    assert benchmark['var_causal']['cases']==77 and round(gain,2)==16.48 and round(pm,2)==10.45
    words=documents(gain,pm)
    import graph_visuals
    influence=graph_visuals.compute_influence()
    graph_visuals.animation(influence)
    figures=[graph_visuals.draw_graph(influence,scene=1,phase='down',name=True),draw2(),
             graph_visuals.draw_graph(influence,scene=3,phase='weighted'),draw4(),draw5(gain,pm),draw6(),draw7()]
    paths=[]
    with PdfPages(OUT/'VISUALS.pdf',metadata={'Title':'SheafCode IRIS video visuals','Author':''}) as pdf:
        for s,fig in zip(SCENES,figures):
            p=OUT/'visuals'/(s['id']+'.png');fig.savefig(p,dpi=100,facecolor=BG);paths.append(p)
            fig.savefig(p.with_suffix('.svg'),facecolor=BG)
            pdf.savefig(fig,facecolor=BG);plt.close(fig)
    make_pptx(paths)
    thumbs=Image.new('RGB',(1440,4*430),BG)
    for i,p in enumerate(paths):
        tile=Image.open(p).convert('RGB').resize((704,396))
        thumbs.paste(tile,(8+(i%2)*720,8+(i//2)*430))
    thumbs.save(OUT/'VISUAL_OVERVIEW.jpg',quality=92)
    claims={'checked_on':'2026-10-13','benchmark_source':str(source.relative_to(ROOT)),
            'benchmark_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'normalized_rmse_gain_percent':gain,'physical_pm25_gain_percent':pm,
            'paired_station_days':77,'benchmark_dates':16,'gain_ci95':benchmark['var_causal']['ci95'],
            'query_result':{'decision_queries':18,'pattern_queries':24,'both_feasible_settings':10,
                            'query_fault_budget':0,'gain_percent':25,'source':'Supplied full manuscript, finite-bank query results'},
            'diagrams':'Exact fitted graph connections with schematic layout; verified current-reading sensitivity; benchmark index and finite-bank query totals; schematic experiment and threshold diagrams',
            'word_count':words,'manuscript_modified':False,'video_uploaded':False}
    (OUT/'CLAIMS_AND_SOURCES.json').write_text(json.dumps(claims,indent=2)+'\n',encoding='utf-8')
    for p in paths:assert Image.open(p).size==(1920,1080)
    assert sum(s['seconds'] for s in SCENES)==88 and words<=205
    import fitz
    with fitz.open(OUT/'VISUALS.pdf') as pdf:assert len(pdf)==7
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.zip')
    bundle=OUT/'SheafCode_IRIS_Video_Kit.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    print(json.dumps({'output':str(OUT),'spoken_words':words,'visuals':len(paths),'bundle_bytes':bundle.stat().st_size},indent=2))

if __name__=='__main__':main()
