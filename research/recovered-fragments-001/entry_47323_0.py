\"\"\"Scientific result overview, generated only from independently recounted data.\"\"\"
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cases import FAMILIAR,HELDOUT

p=Path(__file__).parent
s=json.loads((p/'verified_summary.json').read_text())
raw=json.loads((p/'run/results.json').read_text())
columns=[('Ohne Speicher',raw['controls']['no_memory']['score']),('Faktentext',raw['controls']['text_oracle']['score'])]
labels={'affine':'Affin','bounded':'Beschränkt','coupling':'Coupling'}
columns.extend((f\"{labels[r['method']]}\\nSeed {r['seed']}\",r['score']) for r in s['runs'])
specs=list(FAMILIAR)+list(HELDOUT)
values=np.array([[col['operations'][spec]['accuracy'] for _,col in columns] for spec in specs])
fig,ax=plt.subplots(figsize=(11,8))
im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
ax.set_xticks(range(len(columns)),[x[0] for x in columns],fontsize=9)
ax.set_yticks(range(len(specs)),specs,fontsize=10)
for i,row in enumerate(values):
 for j,v in enumerate(row):ax.text(j,i,f'{round(v*16)}/16',ha='center',va='center',fontsize=9,color='black' if v>.65 else 'white')
ax.axhline(9.5,color='white',linewidth=2)
ax.set_title('QC-CR1: vollständige Antworten pro Aufgabenfamilie',loc='left',pad=18,fontsize=14)
fig.colorbar(im,ax=ax,shrink=.7,label='Anteil korrekter vollständiger Antworten')
fig.text(.11,.025,'Unterhalb der Linie: zurückgehaltene Formen. Je Zelle dieselben 4 Test-IDs × 4 Wertewelten.\\nKeine unabhängigen 16 Fakten; Faktentext ist eine einfache Kontrolle, kein starkes RAG.',fontsize=9)
fig.subplots_adjust(bottom=.13,left=.15,right=.93,top=.91)
fig.savefig(p/'quality_by_family.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.5))
x=np.arange(len(s['runs']));width=.34
single=[r['max_pair_error'] for r in s['runs']];seq=[r['max_sequence_error'] for r in s['runs']]
ax.bar(x-width/2,single,width,label='Einzelne Reparatur',color='#3986a7')
ax.bar(x+width/2,seq,width,label='Maximum über 256 Änderungen',color='#d17b40')
ax.axhline(1e-5,color='#a52c44',linestyle='--',label='Vorab festgelegte Grenze')
ax.set_yscale('log');ax.set_xticks(x,[f\"{labels[r['method']]}\\nSeed {r['seed']}\" for r in s['runs']]);ax.set_ylabel('Maximaler relativer Zustandsfehler')
ax.set_title('Lokale Reparatur auf 15 echten Decoder-Anfragezuständen',loc='left',fontsize=13)
ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(p/'repair_errors.png',dpi=180);plt.close(fig)
