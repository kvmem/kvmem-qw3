from common import HERE
import argparse,json
import torch
from run import Harness
from fastpaths import enable_conv
from verify import verify

def diff(a,b):
 return dict(max=float((a-b).abs().max()),neq=int((a!=b).sum()),n=a.numel())

@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);enable_conv()
 p=argparse.ArgumentParser();p.add_argument('--method',default='infllm');args=p.parse_args()
 h=Harness(argparse.Namespace(method=args.method,window='small',tokens=128,graphs=False))
 cache,pl,ph,_=h.prefix(8190,prime=False)
 norms={};handles=[]
 for i,layer in enumerate(h.model.model.layers):
  handles.append(layer.input_layernorm.register_forward_hook(lambda m,a,o,i=i:norms.setdefault(i,[]).append(o.clone())))
 ids=torch.tensor([h.history[9000:9005]],device='cuda')
 for i in range(5):h.forward(ids[:,i:i+1],cache)
 for handle in handles:handle.remove()
 h.cleanup(cache);del cache
 cache,ql,qh,_=h.prefix(8190,prime=False)
 print(json.dumps(dict(prefill_logits=diff(pl,ql),prefill_hidden=diff(ph,qh))),flush=True)
 first=torch.cat(norms[0],dim=1)
 for name in ('in_proj_qkv','in_proj_a','in_proj_b','in_proj_z'):
  module=getattr(h.model.model.layers[0].linear_attn,name)
  batch=module(first);serial=torch.cat([module(first[:,i:i+1]) for i in range(5)],dim=1)
  print(json.dumps(dict(component=name,comparison=diff(batch,serial))),flush=True)
 def compare(m,a,o,i):
  ref=torch.cat(norms[i],dim=1)
  print(json.dumps(dict(layer=i,norm=diff(o,ref),first=diff(o[:,:1],ref[:,:1]))),flush=True)
 for i,layer in enumerate(h.model.model.layers):
  handles.append(layer.input_layernorm.register_forward_hook(lambda m,a,o,i=i:compare(m,a,o,i)))
 for name, module in h.model.named_modules():
  if type(module).__name__ not in ('Q8Linear','Qwen3_5RMSNorm','Qwen3_5MLP'):continue
  fn=module.forward
  def check(x,fn=fn,name=name):
   y=fn(x)
   if x.ndim==3 and x.shape[1]==5:
    z=torch.cat([fn(x[:,i:i+1]) for i in range(5)],dim=1)
    if not torch.equal(y,z):print(json.dumps(dict(pure_component=name,comparison=diff(y,z))),flush=True)
   return y
  module.forward=check
 verify(h.model,ids,cache,h.backend)
 if h.backend:h.backend.close()

if __name__=='__main__':main()
