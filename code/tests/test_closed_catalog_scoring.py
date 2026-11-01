"""Synthetic fixtures test scoring semantics, not biological performance."""
import unittest
import numpy as np
import pandas as pd
from athpathfinder.scoring import closed_catalog_mask,end_to_end_metrics,prediction_metrics
from athpathfinder.problem import NON_METABOLIC

class ClosedCatalogTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'label':[1,1,1,0], 'true_family':['A','X','B','Control'],
            'family_eligible':[True,False,False,False],
            'stage1_score':[.9,.8,.7,.1], 'stage1_pred':[1,1,1,0],
            'stage2_pred':['A','A','A','A'], 'end_to_end_pred':['A','A','A',NON_METABOLIC]})
    def test_ineligible_is_excluded_even_when_label_is_in_alphabet(self):
        x=self.frame();self.assertEqual(closed_catalog_mask(x,['A','B']).tolist(),[True,False,False,True])
        m=prediction_metrics(x,['A','B']);self.assertEqual(m['stage1_test_n'],4);self.assertEqual(m['stage2_test_n'],1);self.assertEqual(m['end_to_end_n'],2)
    def test_all_controls_retained_and_absent_labels_fixed(self):
        m=end_to_end_metrics(self.frame(),['A','B']);self.assertEqual(m['end_to_end_accuracy'],1)
        self.assertAlmostEqual(m['end_to_end_macro_f1'],2/3)
    def test_wrong_output_alphabet_fails(self):
        x=self.frame();x.loc[0,'end_to_end_pred']='UNSEEN'
        with self.assertRaises(ValueError):end_to_end_metrics(x,['A','B'])
    def test_string_eligibility(self):
        x=self.frame();x['family_eligible']=x.family_eligible.astype(str)
        self.assertEqual(closed_catalog_mask(x,['A','B']).tolist(),[True,False,False,True])
if __name__=='__main__':unittest.main()
