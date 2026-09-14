import unittest
import numpy as np
from analyze_layers import LAYERS, interval_average
from metrics import js


class LayerControlTest(unittest.TestCase):
    def test_opposing_layer_shifts_are_not_hidden_by_layer_control(self):
        before={};after={}
        for i,layer in enumerate(LAYERS):
            a=np.array([0.,1.,0.]) if i<len(LAYERS)//2 else np.array([0.,0.,1.])
            b=a[[0,2,1]]
            before[layer]={p:a for p in [0,1]};after[layer]={p:b for p in [0,1]}
        pooled_before=interval_average(before,None,0,2,3)
        pooled_after=interval_average(after,None,0,2,3)
        self.assertEqual(js(pooled_before,pooled_after),0)
        self.assertEqual(js(interval_average(before,LAYERS[0],0,2,3),
                            interval_average(after,LAYERS[0],0,2,3)),1)


if __name__=='__main__':unittest.main()
