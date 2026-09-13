import unittest
import numpy as np
from repair_litert_zero_blocks import zero_block_plan

class ZeroBlockRepairTest(unittest.TestCase):
    def test_zero_codes_and_scale_are_both_repaired(self):
        weights=np.ones((2,64),dtype=np.float32);weights[0,:32]=0
        packed=np.full(64,0x88,dtype=np.uint8)
        scales=np.array([0,1,1,1],dtype=np.float16)
        edits,scale_edits,blocks,codes=zero_block_plan(weights,[2,64],[2,2],32,packed,scales)
        self.assertEqual(scale_edits,[0]);self.assertEqual(blocks,1);self.assertEqual(codes,32)
        self.assertEqual(edits,{i:0 for i in range(16)})

    def test_prior_scale_only_repair_still_clears_phantom_weights(self):
        weights=np.zeros((1,32),dtype=np.float32)
        edits,scale_edits,_,_=zero_block_plan(weights,[1,32],[1,1],32,np.full(16,0x88,dtype=np.uint8),np.array([.001]))
        self.assertEqual(edits,{i:0 for i in range(16)});self.assertEqual(scale_edits,[])

    def test_nonzero_reference_with_invalid_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            zero_block_plan(np.ones((1,32)),[1,32],[1,1],32,np.zeros(16,dtype=np.uint8),np.array([0.0]))

    def test_bf16_values_below_half_of_smallest_fp16_scale_round_to_zero(self):
        weights=np.full((1,32),1e-24,dtype=np.float32)
        edits,scale_edits,_,_=zero_block_plan(weights,[1,32],[1,1],32,np.full(16,0x88,dtype=np.uint8),np.array([0.0],dtype=np.float16))
        self.assertEqual(edits,{i:0 for i in range(16)});self.assertEqual(scale_edits,[0])

if __name__=='__main__':unittest.main()
