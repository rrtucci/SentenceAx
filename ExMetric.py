from carb_subset.matcher import Matcher
from carb_subset.carb import Benchmark
import re
from sax_globals import *
from sax_utils import *
from SaxExtraction import *
from sax_extraction_utils import get_ex_from_ilabels
from words_tags_ilabels_translation import *
from MOutput import *
from AllenTool import *


class ExMetric():
    """
    similar to Openie6.metric.Carb


    """

    def __init__(self, fix_d=None):
        self.dev_benchmark = Benchmark('carb_subset/data/gold/dev.tsv')
        self.test_benchmark = Benchmark('carb_subset/data/gold/test.tsv')
        self.matchingFunc = Matcher.binary_linient_tuple_match
        self.osentL_to_exs = {}
        # self.l_osent_pos_mask = [] # not used
        # self.l_osent_verb_mask = [] # not used
        self.score_d = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}
        self.fix_d = fix_d

    def __call__(self, l_osentL, lll_ilabel, ll_score):
        self.osentL_to_exs = \
            AllenTool.get_osentL_to_exs_from_lll_ilabel(l_osentL,
                                                        lll_ilabel,
                                                        ll_score,
                                                        self.fix_d)
 
 
    def reset(self):
        self.osentL_to_exs = {}
        self.score_d = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}

    def get_metric_values(self, mode, do_reset=True):
        # similar to Openie6.metric.Carb.get_metric()
        if MAX_EX_DEPTH:
            for osentL in self.osentL_to_exs:
                self.osentL_to_exs[osentL] = \
                    sorted(self.osentL_to_exs[osentL],
                           key=lambda x: x.score,
                           reverse=True)[:MAX_EX_DEPTH]
        openie6_osentL_to_exs = {}
        for osentL, sax_exs in self.osentL_to_exs.items():
            openie6_osentL_to_exs[osentL] = [sax_ex.convert_to_carb_ex
                                           for sax_ex in sax_exs]

        out_fp = "/dev/null"
        if mode == 'dev':
            bmark = self.dev_benchmark
        elif mode == 'test':
            bmark = self.test_benchmark
        else:
            assert False
        auc, optimal_f1_point, last_f1_point = \
            bmark.compare(
                predicted=openie6_osentL_to_exs,
                matchingFunc=self.matchingFunc,
                output_fn=out_fp,
                error_file=None,
                binary=False)

        self.score_d = {
            'carb_auc': auc,
            'carb_f1': optimal_f1_point[2],
            'carb_lastf1': last_f1_point[2]}
        score_d = self.score_d
        if mode == 'dev' and do_reset:
            # this resets score_d
            self.reset()
        return score_d

if __name__ == "__main__":

    def main():
        mode = "test"
        em = ExMetric()
        pred_in_fp = "carb_subset/data/test_gold_allennlp_format.txt"
        at = AllenTool(pred_in_fp)
        osentL_to_exs = at.osentL_to_exs
        l_osentL, lll_ilabel, ll_score =\
            AllenTool.get_lll_ilabel_from_osentL_to_exs(osentL_to_exs)
        em(l_osentL, lll_ilabel, ll_score)
        score_d = em.get_metric_values(mode, do_reset=True)
        print(score_d)