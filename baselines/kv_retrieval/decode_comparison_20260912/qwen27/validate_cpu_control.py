"""Validate full CPU offload against GPU FP8 KV on the real BF16 model."""
import argparse
import json
from pathlib import Path
import time
import run_hf as h


@h.torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists()
    report={'status':'running','started':time.time(),'checks':[]}
    a.out.parent.mkdir(exist_ok=True,parents=True);h.save(a.out,report)
    h.torch.set_num_threads(2);h.enable_conv()
    model,report['model']=h.load_text_model(h.ROOT/'models/Qwen3.8-27B-BF16')
    backend=h.FullOffloadBackend(model)
    try:
        ids=json.loads((h.PARENT/'inputs.json').read_text())['evaluation']['input_ids'][:333]
        gpu=h.Backend.cache(backend,h.Config(capacity=352,dense=True))
        cpu=backend.cache(h.Config(capacity=352))
        for start in range(0,len(ids),128):
            part=h.torch.tensor([ids[start:start+128]],device='cuda')
            x=backend.forward(part,gpu);y=backend.forward(part,cpu)
        h.torch.testing.assert_close(x,y,rtol=0,atol=0)
        backend.finalize(gpu);backend.finalize(cpu)
        for _ in range(8):
            token=x[:,-1].argmax(-1,keepdim=True)
            x=backend.forward(token,gpu);y=backend.forward(token,cpu)
            h.torch.cuda.synchronize()
            h.torch.testing.assert_close(x,y,rtol=0,atol=0)
            report['checks'].append({'logits_byte_exact':True})
        for i in backend.modules:
            state=cpu.layers[i]
            h.torch.testing.assert_close(state.cpu_kv[:,:341],gpu.layers[i].gpu_kv[:,:341].cpu(),rtol=0,atol=0)
        report.update(status='complete',cpu_kv_byte_exact=True)
    except BaseException as exc:
        report.update(status='error',error=repr(exc));raise
    finally:
        backend.close();report['ended']=time.time();h.save(a.out,report)


if __name__=='__main__':main()
