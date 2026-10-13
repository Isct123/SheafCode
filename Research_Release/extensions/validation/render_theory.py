import os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
def main():
    tex=r'''\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}
\title{Additional Conditional Quantity Certificates}
\author{SheafCode Research Notes}\date{}
\begin{document}\maketitle
\section{Sparse fault feasibility}
Let $z\in[\ell,u]\subset R^p$, where $R$ denotes the real numbers, and suppose
\[
 |Az+b+\sum_{j=1}^n B_j e_j|\leq\varepsilon,\qquad
 \|e_j\|_\infty\leq\beta,\qquad
 |\{j:e_j\ne0\}|\leq t.
\]
Inequalities on vectors are componentwise. The quantity of interest is $h^Tz$.
Every bound below is conditional on these assumptions, including the finite
state box. No guarantee of physical model accuracy follows from optimization.
\section{Anytime support refinement}
Introduce $a_j\in[0,1]$ with $|e_{jk}|\leq\beta a_j$ and
$\sum_j a_j\leq t$. A search node fixes selected activations to zero or one;
maximizing $h^Tz$ over its remaining linear relaxation gives an outer bound.

\textbf{Proposition.} Splitting a free activation into the two cases $a_j=0$
and $a_j=1$ preserves all sparse feasible states. At every stage, the maximum
of unexplored node bounds and attained sparse feasible quantities bounds the
original maximum. Under exact arithmetic, this upper bound cannot increase,
and the feasible incumbent cannot decrease. Exhaustive refinement is exact
after finitely many nodes; a full binary tree has at most $2^{n+1}-1$ nodes.
Apply the same argument to $-h$ to obtain a lower quantity bound.

\textit{Proof.} Each sparse feasible fault has a binary activation which is one
on its nonzero blocks and zero elsewhere. It belongs to the root relaxation.
At a split this activation belongs to one child, so the union of the children's
sparse feasible sets is the parent's set. Every child relaxation is contained
in its parent relaxation, and its optimum is therefore no larger. A node may
be pruned when its optimum cannot exceed an attained sparse feasible quantity.
An integral activation optimum permits at most $t$ nonzero fault blocks and
thus attains a value in the actual sparse union. Fixing every activation yields
the exact allowed-support linear programs. Binary branching is finite, which
proves exactness under exhaustive search. The retained outer optima cover every
unexplored feasible state throughout the search.\hfill QED

An interrupted search remains an outer bound, even if no feasible incumbent
has yet been found. The implementation uses numerical tolerances; this is not
an exact-arithmetic proof of its floating-point endpoints. This specializes
established branch-and-bound, rather than inventing that general method.
\section{Uncertain learned maps}
Suppose the true operators are $A+\Delta_A$, $B_j+\Delta_{B_j}$ and the true
offset is $b+\delta_b$. Assume entrywise radii
\[
 |\Delta_A|\leq R_A,\quad |\Delta_{B_j}|\leq R_j,\quad
 |\delta_b|\leq r_b.
\]
Let $M_k=\max(|\ell_k|,|u_k|)$ and $c_{ij}=\sum_k R_j[i,k]$.
Write $\operatorname{top}_t(c_i)$ for the sum of the $t$ largest values in
the nonnegative station vector $(c_{i1},\ldots,c_{in})$.

\textbf{Proposition.} If $\varepsilon$ bounds the true equation residual,
every state/fault vector satisfying the uncertain model satisfies the nominal
model with inflated residual budget
\[
 \varepsilon'_i=\varepsilon_i+(R_A M)_i+r_b[i]
                 +\beta\operatorname{top}_t(c_i).
\]
Every nominal sparse-fault outer quantity certificate formed with
$\varepsilon'$ consequently contains the quantities of these uncertain-model
feasible states, including certificates from interrupted support refinement.

\textit{Proof.} Subtract the perturbation terms from the true residual to obtain
the nominal residual. Let $S$ denote the active support. The componentwise
triangle inequality bounds it by
$\varepsilon+R_A M+r_b+\sum_{j\in S}R_j|e_j|$.
The last term's row $i$ is at most
$\beta\sum_{j\in S}c_{ij}\leq\beta\operatorname{top}_t(c_i)$,
since there are at most $t$ active blocks. This proves feasible-set inclusion;
maximizing or minimizing a linear quantity over a superset proves containment.
\hfill QED

Declared perturbation radii in synthetic experiments do not estimate the true
physical model error. These results specialize established perturbation and
robust optimization arguments. Historical priority and independent proof
review remain unestablished.
\begin{thebibliography}{9}
\bibitem{land} A. H. Land and A. G. Doig. An automatic method of solving
discrete programming problems. Econometrica 28(3), 1960.
DOI: 10.2307/1910129.
\bibitem{robust} D. Bertsimas and M. Sim. The Price of Robustness.
Operations Research 52(1), 2004. DOI: 10.1287/opre.1030.0065.
\end{thebibliography}
\end{document}
'''
    source=OUT/'Mathematical_Extensions.tex';source.write_text(tex,encoding='utf-8')
    compiler=ROOT.parent/'audit_tools/tectonic/tectonic.exe'
    env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(ROOT.parent/'audit_tools/tex_cache')
    subprocess.run([str(compiler),'--only-cached','--outdir',str(OUT),str(source)],check=True,env=env)
if __name__=='__main__':main()
