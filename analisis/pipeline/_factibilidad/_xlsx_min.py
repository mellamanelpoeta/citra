"""Lector mínimo de .xlsx (solo stdlib) para los tabulados de INEGI; el .venv no trae openpyxl."""
import zipfile, re, xml.etree.ElementTree as ET
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def sheets(path):
    z=zipfile.ZipFile(path)
    wb=ET.fromstring(z.read('xl/workbook.xml'))
    rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    rmap={r.get('Id'):r.get('Target') for r in rels}
    return [(s.get('name'), rmap[s.get('{%s}id'%NS['r'])]) for s in wb.find('m:sheets',NS)]
def read(path, sheet):
    z=zipfile.ZipFile(path)
    ss=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',NS):
            ss.append(''.join(t.text or '' for t in si.iter('{%s}t'%NS['m'])))
    tgt=dict(sheets(path))[sheet]
    tgt=tgt.lstrip('/'); tgt=tgt if tgt.startswith('xl/') else 'xl/'+tgt
    root=ET.fromstring(z.read(tgt))
    rows=[]
    for r in root.iter('{%s}row'%NS['m']):
        row={}
        for c in r.findall('m:c',NS):
            ref=c.get('r'); col=re.match(r'[A-Z]+',ref).group()
            v=c.find('m:v',NS); t=c.get('t')
            if v is None:
                is_=c.find('m:is',NS)
                val=''.join(x.text or '' for x in is_.iter('{%s}t'%NS['m'])) if is_ is not None else None
            elif t=='s': val=ss[int(v.text)]
            else: val=v.text
            row[col]=val
        rows.append((int(r.get('r')),row))
    return rows
