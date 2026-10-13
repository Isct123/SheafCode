                                                                                    
import re

SELECTED = [
    ('Generic structural rank and actual structured maps', 'prop:graph', 'Graph recoverability', 0),
    ('Unknown fault supports and functional identification', 'prop:functional', 'Quantity identification under unknown faults', 1),
    ('Support-span compression', 'prop:compression', 'Equivalent-support search compression', 0),
    ('Sharp joint pollutant-threshold scores', 'prop:joint', 'Joint threshold certification', 0),
    ('Anytime refinement of the bounded-fault relaxation', 'prop:anytime', 'Search refinement with interruption bounds', 0),
    ('Exact nominal correction of an approximate dual vector', 'prop:nominal', 'Numerically verified nominal bound', 0),
]

MOTIVATIONS = {
    'Generic structural rank and actual structured maps':
        r'An outage first raises a graph question: which missing station values are determined by surviving constraints? Tree propagation separates this question into connected components and cycle constraints. The generic matching argument below is credited background; Proposition~\ref{prop:graph} addresses the actual invertible endpoint maps.',
    'Sharp pollutant-specific residual-ball intervals':
        r'Proposition~\ref{prop:graph} diagnoses full-state ambiguity. A pollution decision often needs only one quantity, so the next supporting derivation asks whether that quantity can vary along an ambiguous direction. It supplies the interval geometry needed for unknown-support reasoning.',
    'Unknown fault supports and functional identification':
        r'The preceding quantity interval treats the constraints as trustworthy. Unknown faulty stations create several admissible explanations. We first eliminate their unrestricted nuisance values, then ask whether two such explanations can disagree on the chosen quantity. Proposition~\ref{prop:functional} answers this weaker identification question without requiring fault localization.',
    'Support-span compression':
        r'Proposition~\ref{prop:functional} compares possible fault explanations, but direct subset enumeration can repeat the same nuisance effects. Because the preceding elimination depends only on their span, Proposition~\ref{prop:compression} merges equivalent search states and retains their cheapest representatives. This is the unrestricted-nuisance branch of the theory.',
    'Bounded-fault support unions and joint calibration':
        r'Unrestricted faults can leave a decision unbounded. For interpretable certificates we now declare finite fault and state bounds. This changes the feasible model: equal spans need not have equal bounded images, so Proposition~\ref{prop:compression} cannot be reused automatically. The resulting finite support union is the input to Proposition~\ref{prop:joint}. Calibration below is a supporting statistical condition, not an additional novelty claim.',
    'Sharp joint pollutant-threshold scores':
        r'The bounded union describes all admissible pollution explanations, but separate pollutant intervals may still miss a decision implied jointly by them. Proposition~\ref{prop:joint} optimizes the maximum pollutant score over that same union. It therefore moves from quantity reconstruction to the threshold question.',
    'Support-free dual outer bounds and their strict gap':
        r'Proposition~\ref{prop:joint} is sharp but can require many support LPs. A support-free outer bound provides a cheaper starting point. Its strict-gap example explains why the refinement in Proposition~\ref{prop:anytime} is needed rather than treating the relaxation as exact.',
    'Anytime refinement of the bounded-fault relaxation':
        r'The relaxed bound can be too wide to settle the threshold question. Proposition~\ref{prop:anytime} partitions the same bounded support model, retains all sparse explanations and tracks an outer bound at every interruption. It supports upper extrema directly; lower linear extrema use the negated objective, and the joint-score minimum uses the corresponding epigraph minimization with reversed bound directions.',
    'Exact nominal correction of an approximate dual vector':
        r'Proposition~\ref{prop:anytime} assumes exact solves for its exact monotonicity and stopping claims. Floating-point dual vectors may fail the required equality slightly. Proposition~\ref{prop:nominal} adds a finite-box correction, yielding an outward-rounded upper bound for stored coefficients. It verifies linear bounds used by the search; it does not by itself certify a joint-score epigraph lower bound, an incumbent, or physical map accuracy.',
    'Conditioning and the prior-dependent error identity':
        r'The six selected propositions analyze recoverability and feasible-state decisions. The fitted point decoder also needs a separate error identity. This is a supporting derivation for the reconstruction method rather than another headline proposition.',
    'Uniform block-fault uniqueness and noisy support separation':
        r'For comparison with quantity identification, the following credited sparse-recovery statement asks the stronger question of recovering every fault block. Its noisy sufficient condition explains the omitted-block margin used in the fault experiments.',
    'Sharp worst-case repair width and Gaussian repair gain':
        r'Once a quantity envelope exists, revealing a station reading restricts its feasible set. These supporting repair calculations explain the reported repair objectives without adding another main proposition.',
    'Declared uncertainty in the learned operators':
        r'All preceding certificates condition on fixed maps. Declared map-error radii can enlarge the residual budget by the following supporting inequality. Synthetic radii do not establish actual field uncertainty.',
    'Revealed-label adaptive calibration':
        r'The deterministic feasible-set results do not establish operational coverage under unknown distribution shift. The following credited feedback argument explains the separately labelled revealed-label interval experiment.',
}

ORDER = [
    'Generic structural rank and actual structured maps',
    'Sharp pollutant-specific residual-ball intervals',
    'Unknown fault supports and functional identification',
    'Support-span compression',
    'Bounded-fault support unions and joint calibration',
    'Sharp joint pollutant-threshold scores',
    'Support-free dual outer bounds and their strict gap',
    'Anytime refinement of the bounded-fault relaxation',
    'Exact nominal correction of an approximate dual vector',
    'Conditioning and the prior-dependent error identity',
    'C1--C3 diagnostics and their statistical scope',
    'Uniform block-fault uniqueness and noisy support separation',
    'Sharp worst-case repair width and Gaussian repair gain',
    'Declared uncertainty in the learned operators',
    'Revealed-label adaptive calibration',
]

def reorganize(theory):
    pattern = re.escape(r'\subsection{') + r'([^}]+)}'
    matches = list(re.finditer(pattern, theory))
    blocks = {m.group(1): theory[m.end(): matches[i+1].start() if i+1<len(matches) else len(theory)]
              for i,m in enumerate(matches)}
    if set(blocks) != set(ORDER):
        raise RuntimeError('Theory sections changed: '+str(set(blocks)^set(ORDER)))
    selected = {h:(label,title,index) for h,label,title,index in SELECTED}
    result = theory[:matches[0].start()] + r'''
\paragraph{Progression and scope.}
Six selected propositions ask what is recoverable, what quantity survives
fault ambiguity, which fault explanations are redundant, whether a pollution
threshold is decidable, whether the support search can be interrupted safely,
and whether approximate arithmetic preserves the nominal linear bound.
This is a progression of research questions, not a claim that every statement
is a logical corollary of the preceding one. In particular, unrestricted
span compression and bounded-fault decision search use different models.
Established linear-algebra, calibration and repair facts remain supporting
derivations; they are not additional numbered novelty claims.
'''
    for h in ORDER:
        block = blocks[h]
        index = 0
        pieces = re.split('('+re.escape(r'\begin{proposition}')+'|'+re.escape(r'\end{proposition}')+')',block)
        active = False
        transformed = []
        for p in pieces:
            if p == r'\begin{proposition}':
                active=h in selected and index==selected[h][2]
                if active:
                    label,title,_=selected[h]
                    transformed.append(r'\begin{proposition}['+title+r']\label{'+label+'}')
                else:
                    transformed.append(r'\paragraph{Supporting statement.}')
                index+=1
            elif p == r'\end{proposition}':
                transformed.append(p if active else '')
                active=False
            else:
                transformed.append(p)
        block=''.join(transformed).replace(r'\begin{corollary}',r'\paragraph{Supporting consequence.}').replace(r'\end{corollary}','')
        block=block.replace('the residual-ball proposition','the residual-ball supporting statement')
        result += '\n'+r'\subsection{'+h+'}\n'+MOTIVATIONS.get(h,'')+'\n'+block
    if result.count(r'\begin{proposition}') != 6:
        raise RuntimeError('Expected six selected propositions')
    return result

def main_narrative():
    return r'''\section{Mathematical results: from graph recovery to verified decisions}
The mathematical treatment follows one practical question: what can a
monitoring network still establish when readings disappear or some stations
are faulty? Six selected propositions are stated and proved in
Appendix~\ref{app:proofs}. Their progression connects graph structure,
indistinguishable explanations, search-state compression and threshold
decisions. General first-ever priority is not established; each result
specializes established mathematics to the declared monitoring model.

\subsection{Step 1: what does the surviving graph determine?}
Proposition~\ref{prop:graph} uses a rooted spanning tree in each connected
component. With invertible selected endpoint maps, a root state propagates
to every station. Remaining rows impose cycle and additional consistency
conditions on that root. Their surviving degrees of freedom determine
exact outage rank and minimum nonzero kernel support.
The graph traversal is elementary; checking the remaining constraints
requires rank and kernel calculations.
Generic Hall matching is retained as credited background and does not
automatically establish rank for tied or fitted maps. Exact recoverability
also does not guarantee accurate reconstruction of nonzero pollution:
the original fitted operators have zero numerical kernel at the recorded
tolerance, leaving only the zero exactly consistent full state.

\subsection{Step 2: must the whole state and every fault be identified?}
Full-state recovery is stronger than answering a pollution question.
Proposition~\ref{prop:functional} asks whether a chosen quantity $h^Tz$
can differ between any two admissible explanations with at most $t$
faulty station blocks. Two explanations involve a union of at most $2t$
stations; the quantity is identifiable precisely when none of the
corresponding null directions changes it, with the uniform quantifiers
and exact-consistency assumptions stated in the appendix.
The proof compares explanations rather than requiring fault localization.
Supporting residual-ball geometry supplies sharp quantity intervals
when residuals are allowed. The classical stronger sparse fault-uniqueness
argument is credited \citep{donoho2003}; a small second fault cannot be
protected merely by a large total fault norm.

\subsection{Step 3: which fault explanations repeat the same search state?}
The unknown-support model in Step 2 suggests enumerating fault subsets,
but different subsets can produce the same nuisance span.
Proposition~\ref{prop:compression} identifies these equivalent states.
Breadth-first expansion retains a minimum-cardinality representative
for each span, so replacing a more expensive equivalent support cannot
remove a possible future transition. This is a BFS and dominance argument
supported by exact subspace comparisons. The rank-dependent counting bound
is polynomial for fixed total nuisance rank, not a fixed-parameter
tractability claim. This result requires unrestricted nuisance coefficients.

\subsection{Step 4: can all explanations imply a pollution exceedance?}
Unrestricted nuisance values can make intervals unbounded. We therefore
now declare finite fault magnitudes, residual budgets and trusted state
bounds. This is a different feasible model: span equivalence from Step 3
does not generally preserve bounded fault images.
For its finite support union, Proposition~\ref{prop:joint} gives the sharp
enclosing range of the maximum pollutant score. Separate pollutant
intervals can miss a jointly forced exceedance. For example, if
$z_1+z_2=2$, $0\leq z_k\leq2$ and the scores are $z_k/0.75$,
both separate lower bounds are zero but the joint lower bound is $4/3$.
Every state then has an exceedance although neither particular pollutant
must exceed. Epigraph LP optimization formalizes this argument using
established convex optimization \citep{boyd2004}.
The current actual-state certificate tests mostly abstain or lack
eligible calibration; this constructive benefit is not a demonstrated
regulatory decision improvement.

\subsection{Step 5: what if enumerating bounded fault supports is too costly?}
Step 4 is sharp but may require many support LPs. A support-free relaxation
is cheaper and can be strictly wider than the sparse feasible union.
Proposition~\ref{prop:anytime} branches on station activations and keeps
outer bounds over unexplored nodes together with attained sparse
incumbents. Under exact solves an interrupted upper-bound search remains
sound, and exhaustive branching is exact.
The branch-and-bound invariant is classical \citep{land1960}; its
specialization supplies an interpretable computation-budget trade-off.
This search does not inherit the unrestricted span-compression guarantee.
Linear lower bounds use negated objectives; the joint epigraph minimum
requires the corresponding minimization bounds.

\subsection{Step 6: is a computed linear bound numerically trustworthy?}
Step 5 separates the search approximation from exact sparse optimization,
but approximate arithmetic introduces another issue. An approximate dual
vector $w$ may leave a mismatch $\delta=h-A^Tw$.
Proposition~\ref{prop:nominal} bounds $\delta^Tz$ over the finite trusted
box, adds the residual contribution and the largest $t$ station-fault
contributions, and evaluates the result with exact rational arithmetic and
outward rounding. It verifies a linear upper bound for stored coefficients,
not physical map accuracy or LP optimality \citep{jansson2004}.
It also does not by itself validate the epigraph lower-bound calculation
or a floating-point incumbent. Those need their own verified constraints.

\paragraph{Supporting mathematics and empirical scope.}
Decoder conditioning, C1--C3 diagnostics, residual-ball intervals, sparse
uniqueness, calibration, repair objectives and declared operator-error
inflation remain supporting derivations. They explain the methods and
experiments without being promoted to additional numbered novelty claims.
All empirical comparisons, unfavorable outcomes and conditional-certificate
failures remain included. Six linked propositions provide a focused
theoretical narrative; their number is not evidence of mathematical priority
or realized emissions reduction.
'''
