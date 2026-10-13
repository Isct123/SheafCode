from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parent))
from operational_guard import assess

def test_witness_and_abstention():
    assert assess(['A','B'],{'A':37})['label']=='exceedance'
    assert assess(['A','B'],{'A':33})['label']=='abstain'
    assert assess(['A','B'],{'A':34,'B':34})['label']=='safe'
    assert assess(['A','B'],{'A':36,'B':34})['label']=='abstain'
    assert assess(['A','B'],{'A':135,'B':34},faults=1)['label']=='abstain'
    r=assess(['A','B','C'],{'A':135,'B':37},faults=1)
    assert r['label']=='exceedance' and r['witness_stations']==['A','B']
    assert assess(['A','B'],{'A':0,'B':0},faults=1)['label']=='abstain'

@pytest.mark.parametrize('kwargs',[
    {'stations':['A','A'],'readings':{}}, {'stations':['A'],'readings':{'B':0}},
    {'stations':['A'],'readings':{'A':float('nan')}}, {'stations':['A'],'readings':{},'noise':-1},
    {'stations':['A'],'readings':{},'faults':.5}, {'stations':['A'],'readings':{},'faults':True},
])
def test_invalid_inputs(kwargs):
    with pytest.raises(ValueError):assess(**kwargs)

def test_heterogeneous_bounds():
    assert assess(['A','B'],{'A':36,'B':37},noise={'A':0,'B':3})['witness_stations']==['A']
