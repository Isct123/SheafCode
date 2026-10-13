                                                                           
import os,subprocess
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'joint_decision_results'
    tex=r'''\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage[margin=1in]{geometry}
\title{Joint Pollutant-Threshold Certificates}
\author{SheafCode Research Notes}\date{}
\begin{document}\maketitle
\section{Bounded station-fault explanations}
For each support $F$ with $|F|\leq t$, let $P_F$ consist of states and faults
such that
\[
 |Az+b+B_F e|\leq\varepsilon,\qquad |e|\leq\beta,\qquad \ell\leq z\leq u.
\]
The state box is finite; vector inequalities are componentwise. Let $Z$ be
the union of projected state sets of these supports and assume it is nonempty.
For $K$ declared affine pollutant scores, put
\[
 s_k(z)=h_k^Tz+c_k,\qquad S(z)=\max_{1\leq k\leq K}s_k(z).
\]
An illustrative exceedance means $S(z)>1$. This is not an implementation of
official AQI validity rules, pollutant averaging or policy invocation.
\section{Sharp joint score bounds}
\textbf{Proposition.} The sharp enclosing score interval is $[L,U]$, where
\begin{align*}
 L&=\min_F\ \min_{z,e,y}\ y,\qquad (z,e)\in P_F,\quad s_k(z)\leq y\ (k=1,\ldots,K),\\
 U&=\max_F\ \max_k\ \max_{z,e}s_k(z),\qquad (z,e)\in P_F.
\end{align*}
The support indices range over $|F|\leq t$; empty supports are omitted.
These are finitely many linear programs and their
endpoints are attained. Every explanation exceeds the threshold exactly when
$L>1$. Every explanation is below or equal to it exactly when $U\leq1$.
If $L\leq1<U$, there are explanations for both decisions. No universally
correct single decision is then deducible from these assumptions alone.

\textit{Proof.} Each nonempty projected polyhedron is compact because the state
and fault coordinates have finite boxes. Their finite union is compact.
The continuous maximum of finitely many affine functions attains its extrema.
The epigraph constraints minimize precisely this maximum on each support.
The finite maxima over support and pollutant indices commute, giving $U$.
The attaining endpoints prove the decision claims and the indistinguishability
statement. If all supports are empty, there is no feasible explanation; report
inconsistency rather than a vacuous decision.\hfill QED
\section{Why separate pollutant intervals can lose information}
\textbf{Corollary.} Let $\ell_k=\min_{z\in Z}s_k(z)$. Then
$L\geq\max_k\ell_k$, possibly strictly. Conversely the upper endpoint equals
the maximum of separate upper endpoints, so preserving joint dependence
does not further improve that upper endpoint.

\textit{Proof.} Every $s_k(z)$ is at least $\ell_k$, hence $S(z)$ is at least
$\max_k\ell_k$. For strictness, take $z_1+z_2=2$, $0\leq z_1,z_2\leq2$
and $s_k(z)=z_k/0.75$. Each separate lower endpoint is zero, while
\[
 L=\min\frac{\max(z_1,z_2)}{0.75}=\frac{4}{3}>1.
\]
At least one pollutant must exceed its declared threshold, although its
identity is uncertain. The upper equality follows by commuting finite maxima.
\hfill QED
\section{Uncertain maps and interpretation}
If the true uncertain-map feasible set is contained in the inflated nominal
set from the earlier operator-uncertainty proposition, a decision valid for
the larger set holds for the smaller set. Its endpoints are sharp for the
larger model, not necessarily for the original uncertain model.

The nonconvex support union can create gaps in attainable scores. The interval
is its sharp enclosure, not a claim that every intermediate score is attainable.
Exhaustive support enumeration can be expensive. The implementation checks
numerical primal feasibility and guards boundary decisions; it is not exact
rational certification. Physical membership in the box, residual and fault
model requires external evidence. Designed synthetic examples demonstrate the
strict lower-bound gain; neither learned-network synthetic study established
additional exceedance decisions. Field impact and first-ever priority remain
unestablished.

This specializes established epigraph optimization to joint pollutant decisions
under station-fault explanations. Reference: S. Boyd and L. Vandenberghe,
\textit{Convex Optimization}, Cambridge University Press, 2004, sections
3.2.3 and 4.1.3. Available from the authors' textbook website at Stanford
University; the companion Markdown proof includes its full link.
\end{document}
'''
    source=out/'Joint_Threshold_Proof.tex'
    source.write_text(tex,encoding='utf-8')
    env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(root.parent/'audit_tools/tex_cache')
    subprocess.run([str(root.parent/'audit_tools/tectonic/tectonic.exe'),
                    '--only-cached','--outdir',str(out),str(source)],env=env,check=True)


if __name__=='__main__': main()
