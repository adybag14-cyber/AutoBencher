#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from calibrate_dynamic_litert import quantize_weight

class ActualQuantizerTests(unittest.TestCase):
    def test_octav_and_int8_shapes_finite_and_zero_rows(self):
        rng=np.random.default_rng(19)
        weight=rng.normal(0,0.1,size=(16,64)).astype(np.float32)
        weight[0]=0
        for bits in (4,8):
            with self.subTest(bits=bits):
                reconstructed=quantize_weight(weight,bits)
                self.assertEqual(reconstructed.shape,weight.shape)
                self.assertTrue(np.isfinite(reconstructed).all())
                self.assertTrue(np.array_equal(reconstructed[0],np.zeros(64)))
                self.assertLess(float(np.mean((weight-reconstructed)**2)),0.0005)

    def test_int8_usually_more_precise_on_fixed_dense_fixture(self):
        weight=np.random.default_rng(3).normal(size=(16,64)).astype(np.float32)
        e4=np.mean((weight-quantize_weight(weight,4))**2)
        e8=np.mean((weight-quantize_weight(weight,8))**2)
        self.assertLess(e8,e4)

if __name__=='__main__': unittest.main(verbosity=2)
