"""Author-supported large-window configurations; preserve 2x GPU cache ratio."""
import argparse
import sys
import run_hf as h

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--large-block',type=int,choices=[128,2048],default=128)
args,rest=parser.parse_known_args()
sys.argv=[sys.argv[0],*rest]
original_config=h.InfLLMConfig
original_save=h.save


def config(*positional,**kwargs):
    assert kwargs['topk']==991,'This control requires --infllm-window 128k'
    if args.large_block==2048:
        # 3968 recent + 128 initial + 62*2048 retrieved = exactly 131072.
        kwargs.update(n_local=3968,block_size=2048,topk=62)
    kwargs['max_cached_block']=2*kwargs['topk']
    return original_config(*positional,**kwargs)


def save(path,report):
    report['large_window_block_size']=args.large_block
    report['large_window_cache_ratio']=2
    report['wrapper_sha256']=h.digest(__file__)
    original_save(path,report)


h.InfLLMConfig=config
h.save=save
if __name__=='__main__':h.main()
