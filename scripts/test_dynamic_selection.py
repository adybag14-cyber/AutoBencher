#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import copy
import unittest
from build_dynamic_litert import select_refinement

class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.calibration={'selection':{'attention_int8_layers':list(range(42)),
            'mlp_int8_layers':[0,41]},'mlp_ranking':[{'layer':i,'relative_gain':i/100} for i in range(42)]}
        self.floor={'family':'test','selection':{'attention_int8_layers':list(range(42)),
            'mlp_int8_layers':[1,2,3,41],'embedding_int8':True,'lm_head_int8':True}}

    def test_existing_precision_never_lowered(self):
        output=select_refinement(self.calibration,self.floor,12)
        selected=output['selection']['mlp_int8_layers']
        self.assertEqual(len(selected),12)
        self.assertTrue(set(self.floor['selection']['mlp_int8_layers']).issubset(selected))
        self.assertIn(0,selected)
        self.assertEqual(output['selection']['attention_int8_layers'],list(range(42)))

    def test_rejects_impossible_quota(self):
        for quota in (2,43):
            with self.assertRaises(ValueError): select_refinement(self.calibration,self.floor,quota)

    def test_input_not_mutated(self):
        original=copy.deepcopy(self.calibration)
        select_refinement(self.calibration,self.floor,12)
        self.assertEqual(self.calibration,original)

    def test_rejects_incomplete_ranking(self):
        self.calibration['mlp_ranking']=[]
        with self.assertRaises(ValueError): select_refinement(self.calibration,self.floor,12)

if __name__=='__main__': unittest.main(verbosity=2)
