                                                                                       
from pathlib import Path
import sys,json,pickle,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'joint_certification_results'
sys.path.insert(0,str(ROOT/'extensions/online_validation'))
PRIME=1000000007

def full_rank_mod(matrix):
                                                                         
                                                                              
    rows=[]
    for row in matrix:
        values=[float(x).as_integer_ratio() for x in row];den=max((b for a,b in values),default=1)
        assert all(den%b==0 for a,b in values)
        rows.append([(a*(den//b))%PRIME for a,b in values])
    rank=0;width=len(matrix[0]) if len(matrix) else 0;pivots=[]
    for col in range(width):
        pivot=next((i for i in range(rank,len(rows)) if rows[i][col]),None)
        if pivot is None:continue
        rows[rank],rows[pivot]=rows[pivot],rows[rank];inv=pow(rows[rank][col],-1,PRIME)
        for j in range(col,width):rows[rank][j]=rows[rank][j]*inv%PRIME
        for i in range(rank+1,len(rows)):
            factor=rows[i][col]
            if factor:
                for j in range(col,width):rows[i][j]=(rows[i][j]-factor*rows[rank][j])%PRIME
        pivots.append(col);rank+=1
        if rank==len(rows):break
                                                                       
                                                                              
    return rank,pivots

def main():
    assert all(PRIME%d for d in range(2,math.isqrt(PRIME)+1)), 'Modulus must be prime'
    assert full_rank_mod(np.array([[1.,2.],[2.,4.]]))[0]==1
    assert full_rank_mod(np.eye(3))[0]==3
    source=ROOT/'math_integration_results/ROLE_STATUS.json';roles=json.loads(source.read_text());records=[];endpoint_checks=0
    for role in roles:
        name=role['cohort'];folder=ROOT/'recent_data_results'/('delhi' if name=='original_india' else 'epa_la') if name in ['original_india','original_usa'] else ROOT/'expanded_validation_results'/name
        with (folder/'models.pkl').open('rb') as f:model=pickle.load(f)
        D=model['sheaf'].D;n,p=model['sheaf'].n,model['sheaf'].p
        rank,pivots=full_rank_mod(D);assert rank==n*p,'Modular test inconclusive; exact rank not certified'
        endpoint=[]
        for e,(u,v) in enumerate(model['edges']):
            for station in [u,v]:
                r,cols=full_rank_mod(D[e*p:(e+1)*p,station*p:(station+1)*p]);assert r==p,'Endpoint test inconclusive';endpoint_checks+=1;endpoint.append({'edge':e,'station':station,'rank':r,'pivot_columns':cols})
        records.append({'cohort':name,'rows':D.shape[0],'columns':D.shape[1],'exact_column_rank':rank,'pivot_columns':pivots,'endpoint_certificates':endpoint,'input_sha256':hashlib.sha256((folder/'models.pkl').read_bytes()).hexdigest()})
    report={'status':'passed','modulus':PRIME,'operators':len(records),'endpoint_blocks':endpoint_checks,'scope':'Full exact rank of stored binary coefficient matrices certified by modular full rank after exact row scaling. This certifies nominal algebra, not physical model truth or prediction accuracy.','records':records,'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if '--check-only' not in sys.argv:(OUT/'EXACT_GRAPH_RANK.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))

if __name__=='__main__':main()
