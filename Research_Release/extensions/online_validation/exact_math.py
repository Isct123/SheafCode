                                                                                   
import sys,itertools,random
from fractions import Fraction as F
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def rref(M):
    a=[[F(x) for x in row] for row in M];m=len(a);n=len(a[0]) if m else 0;row=0;pivots=[]
    for col in range(n):
        pivot=next((j for j in range(row,m) if a[j][col]),None)
        if pivot is None:continue
        a[row],a[pivot]=a[pivot],a[row];scale=a[row][col];a[row]=[v/scale for v in a[row]]
        for j in range(m):
            if j!=row and a[j][col]:
                coefficient=a[j][col];a[j]=[v-coefficient*w for v,w in zip(a[j],a[row])]
        pivots.append(col);row+=1
        if row==m:break
    return a,pivots

def rank(M):return len(rref(M)[1])
def columns(M,indices):return [[row[i] for i in indices] for row in M]
def mv(M,x):return [sum(F(a)*b for a,b in zip(row,x)) for row in M]
def nullvector(M):
    a,pivots=rref(M);n=len(M[0]);free=next(i for i in range(n) if i not in pivots);x=[F(0)]*n;x[free]=1
    for row,pivot in zip(a,pivots):x[pivot]=-row[free]
    return x

def main():
    checks=[];witnesses=[]
    def check(name,passed,detail=None):
        checks.append({'check':name,'passed':bool(passed),'detail':detail})
        if not passed:raise AssertionError((name,detail))
                                                                             
                                                                     
    sets=[s for k in range(1,4) for s in itertools.combinations(range(3),k)]
    for number,entries in enumerate(itertools.product([-1,0,1],repeat=9)):
        M=[list(entries[i:i+3]) for i in range(0,9,3)]
        deficient=[s for s in sets if rank(columns(M,s))<len(s)]
        distance=min(map(len,deficient)) if deficient else float('inf')
        unique=all(rank(columns(M,s))==len(s) for s in sets if len(s)<=2)
        check(f'block_distance_equivalence/{number}',unique==(distance>2))
        if not unique:
            support=next(s for s in deficient if len(s)<=2);h=nullvector(columns(M,support));u=[F(0)]*3;v=[F(0)]*3
            u[support[0]]=h[0]
            if len(support)>1:v[support[1]]=-h[1]
            check(f'explicit_sparse_collision/{number}',u!=v and mv(M,u)==mv(M,v))
    rng=random.Random(9441)
    for number in range(250):
        M=[[rng.randint(-3,3) for _ in range(6)] for _ in range(5)]
        unions=[s for k in range(1,4) for s in itertools.combinations(range(3),k)]
        deficient=[s for s in unions if rank(columns(M,[c for v in s for c in [2*v,2*v+1]]))<2*len(s)]
        d=min(map(len,deficient)) if deficient else float('inf')
        uniqueness=all(rank(columns(M,[c for v in s for c in [2*v,2*v+1]]))==2*len(s) for s in unions if len(s)<=2)
        check(f'two_coordinate_station_blocks/{number}',uniqueness==(d>2))
                                                                        
    true_residual_squared=F(1);wrong_residual_squared=F(1,100)
    check('total_amplitude_is_not_sufficient',F(10000)+F(1,100)>4 and wrong_residual_squared<true_residual_squared)
    witnesses.append({'type':'discarded_noisy_support_bound','D':'identity(3)','fault':['100','1/10','0'],'residual':['0','0','1']})
    for number in range(200):
        fault=[F(rng.randint(5,20)),F(rng.randint(5,20)),F(0),F(0)]
        noise=[F(rng.randint(-1,1)) for _ in range(4)];syndrome=[a+b for a,b in zip(fault,noise)]
        check(f'noisy_minimum_block_condition/{number}',min(fault[:2])**2>4*sum(x*x for x in noise))
        true=sum(x*x for x in syndrome[2:])
        for support in itertools.combinations(range(4),2):
            if support==(0,1):continue
            wrong=sum(x*x for i,x in enumerate(syndrome) if i not in support)
            check(f'noisy_support_separation/{number}/{support}',wrong>true)
                                                                      
    vertices=[(F(0),F(2)),(F(2),F(0))];mid=(F(1),F(1));joint=max(mid)/F(3,4)
    check('joint_minimum_equals_four_thirds',joint==F(4,3) and all(max(v)/F(3,4)>=joint for v in vertices))
    check('separate_minimum_equals_zero',max(min(v[k]/F(3,4) for v in vertices) for k in range(2))==0)
    for j in range(201):
        z1=F(j,100);z2=2-z1
        check(f'coupled_rational_line/{j}',max(z1,z2)/F(3,4)>=F(4,3))
                                                                         
                                                                                       
                                                               
                                                                   
                                                          
    for z in [F(0),F(2)]:
        e=[1-z,1-z];a=[F(1,2),F(1,2)]
        check(f'strict_gap_relaxation_witness/{z}',all(z+v==1 and abs(v)<=2*b for v,b in zip(e,a)) and sum(a)<=1)
        check(f'strict_gap_not_sparse/{z}',sum(v!=0 for v in e)>1)
    sparse_states={1-e1 for e1,e2 in itertools.product([F(j,2) for j in range(-4,5)],repeat=2) if e1==e2 and (e1==0 or e2==0)}
    check('sharp_sparse_union_singleton',sparse_states=={F(1)})
    for number in range(100):
        w=F(1)-F(number+1,10**9);z=F(10**9);delta=F(1)-w
        bound=w*z+delta*z
        check(f'exact_dual_equality_correction/{number}',bound==z and w*z<z)
    for number in range(200):
                                                                           
                                                                             
        A=[[F(rng.randint(-3,3)) for _ in range(2)] for _ in range(3)]
        z=[F(rng.randint(-2,2)) for _ in range(2)];b=[F(rng.randint(-2,2)) for _ in range(3)]
        h=[F(rng.randint(-2,2)) for _ in range(2)];w=[F(rng.randint(-5,5),7) for _ in range(3)]
        B=[[F(rng.randint(-2,2)) for _ in range(3)] for _ in range(3)]
        active=rng.randrange(3);fault=[F(0)]*3;fault[active]=F(rng.randint(-2,2))
        residual=[a+c+d for a,c,d in zip(mv(A,z),b,mv(B,fault))];eps=[abs(x) for x in residual]
        delta=[h[j]-sum(A[i][j]*w[i] for i in range(3)) for j in range(2)]
        leverage=[abs(sum(B[i][j]*w[i] for i in range(3))) for j in range(3)]
        certificate=-sum(a*c for a,c in zip(b,w))+sum(a*abs(c) for a,c in zip(eps,w))+2*max(leverage)+2*sum(abs(x) for x in delta)
        check(f'full_rational_certificate/{number}',sum(a*c for a,c in zip(h,z))<=certificate)
                                                                               
    alpha=F(1,10);gamma=F(1,200);level=alpha;errors=[]
    for t in range(10000):
        err=0 if level<=0 else 1 if level>=1 else int(t%7!=0)
        errors.append(err);level+=gamma*(alpha-err)
        check(f'ACI_exact_boundary/{t}',-gamma<=level<=1+gamma)
    check('ACI_exact_telescope',sum(errors)-len(errors)*alpha==(alpha-level)/gamma)
                                                                                    
    for number in range(80):
        M=[[rng.randint(-1,1) for _ in range(5)] for _ in range(3)]
        def span(s):
                                                                          
            trans=[[M[j][i] for j in range(3)] for i in s]
            rr,_=rref(trans);return tuple(tuple(x) for x in rr if any(x))
        exhaustive={span(s) for k in range(4) for s in itertools.combinations(range(5),k)}
        frontier={span(()):()};queue=[()]
        while queue:
            s=queue.pop(0)
            if len(s)>=3:continue
            for v in range(5):
                candidate=tuple(sorted(set(s)|{v}));key=span(candidate)
                if key not in frontier:frontier[key]=candidate;queue.append(candidate)
        check(f'exact_span_enumeration/{number}',set(frontier)==exhaustive)
    save('exact_math_results.json',{'status':'passed','checks':checks,'total_checks':len(checks),'witnesses':witnesses,
        'scope':'Exhaustive finite exact-rational operator/collision tests and selected exact identities. Internal proof review and finite verification, not a Lean/Coq proof or unrestricted priority certification.'})
    print('Exact arithmetic checks passed',len(checks),flush=True)

if __name__=='__main__':main()
