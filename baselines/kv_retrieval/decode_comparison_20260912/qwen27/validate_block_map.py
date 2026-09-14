"""CPU-only equivalence test invoking both actual upstream buffer-map functions."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from fast_block_map import install


def state(topk,blocks):
    calls=[]
    class Unit:
        def __init__(self,index):self.index=index
        def load(self,target):
            calls.append((self.index,target[0].storage_offset(),target[1].storage_offset()))
            target[0].fill_(self.index+1);target[1].fill_(-self.index-1)
    return SimpleNamespace(num_units=1,n_local=4,block_size=2,topk=topk,
        global_buffer=torch.zeros(2,1,1,topk*2+32,2),global_buffer_block_id_list=[[-1]*topk],
        global_buffer_init_st=-1,global_buffer_init_ed=-1,
        global_remainder=(torch.full((1,1,16,2),.25),torch.full((1,1,16,2),.75)),
        _global_remainder_ed=8,_global_remainder_st=0,
        init_k=torch.full((1,1,2,2),2.),init_v=torch.full((1,1,2,2),3.),
        cached_blocks=[dict.fromkeys(range(blocks))],global_blocks=[[Unit(i) for i in range(blocks)]],calls=calls)


def main():
    torch.set_num_threads(1)
    original,accelerated,provenance=install()
    report={'status':'running','provenance':provenance,'checks':[]}
    rng=random.Random(61)
    for budget in [1,4,16,64,128,991]:
        left,right=state(budget,budget*2+3),state(budget,budget*2+3)
        choices=[[],list(range(min(3,budget))),list(range(budget)),list(range(budget))]
        if budget<=128:
            choices += [rng.sample(range(budget*2+3),budget) for _ in range(12)]
        else:
            choices += [list(range(1,budget+1)),list(range(2,budget+2)),list(range(1,budget+1))]
        for selected in choices:
            a=original(left,1,[selected.copy()]);b=accelerated(right,1,[selected.copy()])
            torch.testing.assert_close(a[0],b[0],rtol=0,atol=0);torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
            assert a[2:]==b[2:]
            assert left.global_buffer_block_id_list==right.global_buffer_block_id_list
            assert left.calls==right.calls
            torch.testing.assert_close(left.global_buffer,right.global_buffer,rtol=0,atol=0)
        report['checks'].append({'topk':budget,'transitions':len(choices),'block_ids_slots_copy_order_and_outputs_exact':True})
    report['status']='complete'
    (HERE/'block_map_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
