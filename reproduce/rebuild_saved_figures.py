#!/usr/bin/env python3
"""Plot archived numerical evidence without rerunning detector inference.

Emits five full original figure types from saved JSON and one clearly marked
CNN-only H0-tail diagnostic. Never modifies any original result or checkpoint.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
CODE=ROOT/'code'
OUT=ROOT/'reproduced_figures'
OUT.mkdir(exist_ok=True)
for directory in (CODE, CODE/'battery', CODE/'cnns', CODE/'scripts_shared'):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

def loadmodule(name,relative):
    p=CODE/relative
    spec=importlib.util.spec_from_file_location('paper2_release_'+name,p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.FIG=OUT
    return mod

def j(rel):
    with (DATA/rel).open(encoding='utf-8') as f:
        return json.load(f)

# Full replot from the original unmodified plot functions.
d46=loadmodule('d46','battery/d46_r8c_scaled_tercile_pfa.py')
d46._fig(j('coherence_energy/d46_r8c_scaled_tercile_pfa.json'))
print('recreated Fig 2: sea-state tercile realized PFA',flush=True)

d49=loadmodule('d49','battery/d49_h0_shift.py')
d49._figs(j('h0_calibration_transfer/d49_h0_shift.json'))
print('recreated Fig 3: H0 calibration transfer',flush=True)

d47=loadmodule('d47','battery/d47_occlusion_dists.py')
d47._figs(j('coherence_energy/d47_occlusion_dists.json'))
print('recreated Fig 5: occlusion, plus extra delta-score figure',flush=True)

# Fig 4: invoke separate path-adapted copy of the ORIGINAL plotting algorithm.
# The original source remains unchanged in code/figures/.
spec=importlib.util.spec_from_file_location('paper2_frontier_saved',ROOT/'reproduce/paper2_frontier_saved.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('recreated Fig 4: two-sided frontier',flush=True)

# Fig 7: original D48 per-cell panel, isolated from its raw-data-dependent RD map.
d48=j('real_target/d48_real_target_invivo.json')
DETS=('CNN-C','CNN-IQ','Proposed','ANMF-DL0.001')
fig,axes=plt.subplots(4,8,figsize=(21,9.2),sharey=True)
for di,dn in enumerate(DETS):
    for col,fname in enumerate(sorted(d48['per_file'])):
        pf=d48['per_file'][fname]
        cells=sorted(pf['per_cell'],key=int)
        ax=axes[di][col]
        for x,c in enumerate(cells):
            cell=pf['per_cell'][c]
            r=cell['rates'][dn]
            lo,hi=cell['rate_ci'][dn]
            color='#D55E00' if cell['role']=='target' else '#0072B2'
            ax.errorbar(x,max(r,0.4/pf['n_windows']),
                        yerr=[[max(r-lo,0)],[max(hi-r,0)]],
                        fmt='o',ms=3.5,color=color,capsize=2)
        ax.set_yscale('log');ax.set_ylim(0.3e-2,1.6)
        if di==0:ax.set_title(fname[:13]+'…',fontsize=7)
        if col==0:ax.set_ylabel(f'{dn}\napparent rate',fontsize=7.5)
        ax.grid(alpha=0.25,which='both');ax.tick_params(labelsize=6)
fig.suptitle('Per-cell apparent detection rates (deployed 1e-3; orange = target cell, blue = adjacent)',fontsize=9)
fig.tight_layout(rect=(0,0,1,0.95))
fig.savefig(OUT/'fig_realtarget_cellrates.pdf')
fig.savefig(OUT/'fig_realtarget_cellrates.png',dpi=140)
plt.close(fig)
print('recreated Fig 7: target and adjacent-cell rates',flush=True)

# Diagnostic, NOT a full reproduction of Fig 1: ANMF raw phase-2 H0 scores
# are not in the original compact transfer package.
with np.load(DATA/'learned_baselines/d40_scores_phase2.npz') as z:
    samples={'CNN-C':np.asarray(z['full_c']), 'CNN-IQ':np.asarray(z['full_r'])}
fig,ax=plt.subplots(figsize=(3.5,2.7))
for name,col in (('CNN-C','#8C564B'),('CNN-IQ','#7F7F7F')):
    arr=np.sort(samples[name]);n=len(arr)
    surv=np.maximum(1.0-np.arange(1,n+1)/n,1/(2*n))
    ax.loglog(arr,surv,label=name,color=col,lw=1.1)
ax.set_xlabel('detector score (a.u.)');ax.set_ylabel('P(score > s)')
ax.set_title('CNN H0 tails — partial reconstruction',fontsize=8)
ax.legend(frameon=False,fontsize=7);ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT/'fig_cnn_h0tail_CNN_ONLY_DIAGNOSTIC.pdf')
plt.close(fig)
print('created diagnostic: CNN H0 tails ONLY; original full Fig 1 PDF preserved',flush=True)
print('Fig 6 cannot be numerically recreated without raw IPIX CDF; original PDF preserved.',flush=True)
