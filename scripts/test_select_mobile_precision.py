import copy, unittest
from select_mobile_precision import select

class SelectionTests(unittest.TestCase):
    def calibration(self):
        return {'calibration':{'no_benchmark_data':True},
                'groups':[{'group':kind,'layer':i,'int4_err':1000.0,
                           'int8_err':.1,'signal':100,'relative_gain':9.999}
                          for kind in ['attn','mlp'] for i in range(42)]}
    def test_strict_budget_does_not_force_any_int4_layers(self):
        result=select(self.calibration())
        self.assertEqual(result['int4_mlp_layers'],[])
        self.assertEqual(result['selection']['mlp_int8_layers'],list(range(42)))
    def test_duplicate_groups_or_benchmark_calibration_are_rejected(self):
        value=self.calibration();value['groups'].append(copy.deepcopy(value['groups'][0]))
        with self.assertRaises(ValueError):select(value)
        value=self.calibration();value['calibration']['no_benchmark_data']=False
        with self.assertRaises(ValueError):select(value)

if __name__=='__main__':unittest.main()
