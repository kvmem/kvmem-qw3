"""Same block=128, 128K policy, with validated exact CPU map bookkeeping."""
import run_large_infllm as entry
from fast_block_map import install

_,_,provenance=install()
original_save=entry.h.save


def save(path,report):
    report['block_map_optimization']=provenance
    report['fast_wrapper_sha256']=entry.h.digest(__file__)
    original_save(path,report)


entry.h.save=save
if __name__=='__main__':entry.h.main()
