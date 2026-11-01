"""Synthetic tests of invariance, API bounds and unchanged family decisions."""
import contextlib,io,unittest
from unittest.mock import patch
import numpy as np
from athpathfinder.embedding import phi,FEATURE_VERSION
from athpathfinder.predict import AthPathFinder,validate_query,predict
from athpathfinder.algorithms.family import JaccardFamily
from athpathfinder.__main__ import main
class ConstantScorer:
    def predict_proba(self,x):return np.tile([.1,.9],(len(x),1))
class GeneSetSeedTests(unittest.TestCase):
    def setUp(self):
        self.genes=[f'AT1G{i:05d}' for i in range(1,25)]
        self.go={g:{'GO:A',f'GO:{i%4}'} for i,g in enumerate(self.genes)}
        self.terms=['GO:A','GO:0','GO:1','GO:2','GO:3']
        self.head=JaccardFamily().fit([self.genes[:5],self.genes[5:10]],['family-first','family-second'])
        self.model=AthPathFinder(self.terms,self.go,ConstantScorer(),self.head,.5,self.head.families,family_kind='jaccard')
    def test_long_set_features_ignore_names_order_duplicates_and_case(self):
        a=phi(self.genes,self.terms,self.go,'file-a')
        b=phi([g.lower() for g in self.genes[::-1]]+self.genes,self.terms,self.go,'file-b')
        np.testing.assert_array_equal(a,b)
    def test_api_predictions_ignore_name_order_duplicates(self):
        a=self.model.predict_one(self.genes,'a').as_dict();b=self.model.predict_one(self.genes[::-1]+self.genes,'b').as_dict()
        self.assertEqual(a,b)
    def test_four_retained_is_error_even_with_duplicates_or_unknown(self):
        for genes in [[],self.genes[:4],self.genes[:4]*4+["AT1G99999"]]:
            with self.assertRaises(ValueError):self.model.predict_one(genes)
    def test_unknown_genes_discarded_and_reported(self):
        p=self.model.predict_one(self.genes[:5]+['AT1G99999','invalid']).as_dict()
        self.assertEqual(p['genes'],self.genes[:5]);self.assertEqual(p['discarded_genes'],['AT1G99999','INVALID'])
    def test_five_retained_is_valid(self):self.assertEqual(len(self.model.predict_one(self.genes[:5]).genes),5)
    def test_zero_overlap_keeps_first_label_and_sets_flag(self):
        p=self.model.predict_one(self.genes[15:20]);self.assertEqual(p.family,'family-first');self.assertTrue(p.no_gene_overlap_support);self.assertEqual(p.family_jaccard,0)
    def test_tie_keeps_stored_reference_order(self):
        self.assertEqual(self.head.predict([self.genes[:10]])[0],'family-first')
    def test_empty_catalog_is_error(self):
        with self.assertRaises(ValueError):JaccardFamily().predict([self.genes])
    def test_cli_invalid_input_is_nonzero_and_clear(self):
        with patch('athpathfinder.__main__.predict',side_effect=ValueError('At least 5 distinct retained genes')):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as caught:main(self.genes[:4])
        self.assertEqual(caught.exception.code,2);self.assertIn('At least 5',err.getvalue())
    def test_rice_retains_low_support_families_explicitly(self):
        from evaluate_rice_catalog import build_rice_rows
        genes={str(i) for i in range(1,7)};go={g:{'GO:1'} for g in genes}
        pos,_,audit=build_rice_rows({'00010':genes,'00020':genes},{},go,{'00010':'Carbohydrate metabolism','00020':'Energy metabolism'},set())
        self.assertTrue(all(r['family_eligible'] for r in pos))
        self.assertEqual(audit['families_passing_FAMILY_MIN'],[])
        self.assertIn('all ten mapped',audit['family_support_policy'])

    def test_legacy_model_is_not_silently_loaded(self):
        import tempfile,pickle
        with tempfile.NamedTemporaryFile() as f:
            f.write(pickle.dumps({'selected_go':self.terms}));f.flush()
            with self.assertRaises(ValueError):AthPathFinder.load(f.name,self.go)
if __name__=='__main__':unittest.main()
