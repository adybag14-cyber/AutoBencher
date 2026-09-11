#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import unittest
from export_full_precision_control import explicit_unquantized_kwargs

class FullPrecisionControlTests(unittest.TestCase):
    def test_none_is_explicit_and_half_precision_disabled(self):
        result=explicit_unquantized_kwargs({'quantization_recipe':None,'experimental_use_fp16':True,
            'experimental_use_mixed_precision':True,'cache_length':32768})
        self.assertEqual(result['quantization_recipe'],'none')
        self.assertFalse(result['experimental_use_fp16'])
        self.assertFalse(result['experimental_use_mixed_precision'])
        self.assertEqual(result['cache_length'],32768)

    def test_does_not_mutate_caller(self):
        kwargs={'quantization_recipe':None}
        explicit_unquantized_kwargs(kwargs)
        self.assertIsNone(kwargs['quantization_recipe'])

    def test_explicit_quantized_recipe_is_rejected(self):
        with self.assertRaises(ValueError):
            explicit_unquantized_kwargs({'quantization_recipe':'MINICPM_DYNV3'})

if __name__=='__main__': unittest.main(verbosity=2)
