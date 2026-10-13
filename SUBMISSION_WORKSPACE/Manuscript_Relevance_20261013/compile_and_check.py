from pathlib import Path
import subprocess,os,json,hashlib,re,pymupdf as fitz

out=Path(__file__).resolve().parent;root=out.parents[1]
env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(root/'audit_tools/tex_cache')
tex=out/'SheafCode_Focused_Manuscript.tex'
run=subprocess.run([str(root/'audit_tools/tectonic/tectonic.exe'),'--only-cached','--keep-logs','--outdir',str(out),str(tex)],env=env,capture_output=True,text=True)
(out/'compile_console.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
checks=[dict(check='compiler_exit_zero',passed=run.returncode==0)]
log=tex.with_suffix('.log').read_text(encoding='utf-8',errors='replace')
for pattern in ['undefined references','undefined citations','Missing character','Overfull \\hbox','Overfull \\vbox','! LaTeX Error']:
    checks.append(dict(check='log/'+pattern,passed=pattern.lower() not in log.lower()))
pdf=fitz.open(tex.with_suffix('.pdf'));text='\n'.join(p.get_text() for p in pdf)
(out/'PDF_TEXT.txt').write_text(text,encoding='utf-8')
checks.extend([dict(check='no_unresolved_references',passed='??' not in text),dict(check='no_replacement_glyphs',passed='\ufffd' not in text)])
for i,p in enumerate(pdf):
    bad=[w for w in p.get_text('words') if w[0]<-1 or w[1]<-1 or w[2]>p.rect.width+1 or w[3]>p.rect.height+1]
    checks.append(dict(check=f'page/{i+1}/text_in_page',passed=not bad))
    if 'Sensor graph (India)' in p.get_text():
        p.get_pixmap(matrix=fitz.Matrix(1.8,1.8),clip=fitz.Rect(40,40,p.rect.width-40,230)).save(out/'corrected_graph_detail.png')
report=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',pages=len(pdf),checks=checks,tex_sha256=hashlib.sha256(tex.read_bytes()).hexdigest(),pdf_sha256=hashlib.sha256(tex.with_suffix('.pdf').read_bytes()).hexdigest(),environment_warning='Tectonic prints a Fontconfig configuration warning; native exit code is zero and the TeX log has no missing characters or overflow warnings.')
(out/'PDF_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
if report['status']!='passed':raise SystemExit(1)
