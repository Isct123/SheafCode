                                                                      

                                                                              
                                                                       
   
from fractions import Fraction
import numpy as np

def canonical_span(rows,ambient):
    M=[[Fraction(str(x)) for x in row] for row in rows]
    if any(len(row)!=ambient for row in M):raise ValueError('Invalid ambient dimension')
    pivot=0
    for col in range(ambient):
        candidate=next((i for i in range(pivot,len(M)) if M[i][col]),None)
        if candidate is None:continue
        M[pivot],M[candidate]=M[candidate],M[pivot];scale=M[pivot][col]
        M[pivot]=[x/scale for x in M[pivot]]
        for i in range(len(M)):
            if i!=pivot and M[i][col]:
                scale=M[i][col];M[i]=[a-scale*b for a,b in zip(M[i],M[pivot])]
        pivot+=1
        if pivot==len(M):break
    return tuple(tuple(row) for row in M[:pivot])

def exact_lattice(blocks,ambient,t):
    if not isinstance(t,int) or t<0 or t>len(blocks):raise ValueError('Invalid fault budget')
    converted=[]
    for B in blocks:
        B=np.asarray(B)
        if B.ndim!=2 or B.shape[0]!=ambient:raise ValueError('Invalid block shape')
        converted.append(B.T.tolist())
    states={():[]};queue=[()];cursor=0
    while cursor<len(queue):
        key=queue[cursor];cursor+=1;support=states[key]
        if len(support)>=t:continue
        for j,rows in enumerate(converted):
            if j in support:continue
            newkey=canonical_span(list(key)+rows,ambient)
            if newkey not in states:states[newkey]=support+[j];queue.append(newkey)
    return states
