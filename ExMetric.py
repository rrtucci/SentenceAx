from carb_subset.matcher import Matcher
from carb_subset.carb import Benchmark
import re
from sax_globals import *
from sax_utils import *
from SaxExtraction import *
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

    def __call__(self, l_osentL, lll_ilabel, ll_confi):
        """


        Parameters
        ----------
        l_osentL
        lll_ilabel
        ll_confi

        Returns
        -------

        """
        self.osentL_to_exs = \
            AllenTool.get_osent2_to_exs_from_lll_ilabel(l_osentL,
                                                        lll_ilabel,
                                                        ll_confi,
                                                        self.fix_d)
        print("Just enetered samples into ExMetric instance via its "
              "__call__() method.")
        print("number of samples=", len(lll_ilabel))

    def reset(self):
        self.osentL_to_exs = {}
        self.score_d = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}

    def get_metric_values(self, mode, do_reset=True):
        # similar to Openie6.metric.Carb.get_metric()
        if MAX_EX_DEPTH:
            for osentL in self.osentL_to_exs:
                self.osentL_to_exs[osentL] = \
                    sorted(self.osentL_to_exs[osentL],
                           key=lambda x: x.confi,
                           reverse=True)[:MAX_EX_DEPTH]
        osent_to_exs = SaxExtraction.shorten_osentL_to_exs(self.osentL_to_exs)
        carb_osent_to_exs = \
            SaxExtraction.get_carb_osent2_to_exs(osent_to_exs)
        for osent, exs in carb_osent_to_exs.items():
            if "Philip Russell" in osent:
                for ex in exs:
                    print("deft67", ex.pred, ex.args)

        # no /dev/null in Windows
        # out_fp = "/dev/null"
        out_fp = "dev_null.txt"
        if mode == 'dev':
            bmark = self.dev_benchmark
        elif mode == 'test':
            bmark = self.test_benchmark
        else:
            assert False
        auc, optimal_f1_point, last_f1_point = \
            bmark.compare(
                predicted=carb_osent_to_exs,
                matchingFunc=self.matchingFunc,
                output_fn=out_fp,
                error_file=None,
                binary=False)

        self.score_d = {
            'carb_auc': auc,
            'carb_f1': optimal_f1_point[2],
            'carb_last_f1': last_f1_point[2]}
        score_d = self.score_d
        if mode == 'dev' and do_reset:
            # this resets score_d
            self.reset()
        return score_d




if __name__ == "__main__":

    # main1() didn't work.
    # "carb_subset/data/test_gold_allennlp_format.txt"
    # and "carb_subset/data/gold/test.tsv:" are DIFFERENT
    def main1():
        in_fp = "carb_subset/data/test_gold_allennlp_format.txt"
        at = AllenTool(in_fp)
        osent_to_exs = SaxExtraction.shorten_osentL_to_exs(
            at.osentL_to_exs)
        ex_met = ExMetric()
        mode = "test"
        pred_l_osent, pred_lll_ilabel, pred_ll_confi = \
            AllenTool.get_lll_ilabel_from_osent2_to_exs(osent_to_exs)

        ex_met(pred_l_osent, pred_lll_ilabel, pred_ll_confi)
        score_d = ex_met.get_metric_values(mode, do_reset=True)
        print(score_d)

    def main2():
        ex_met = ExMetric()
        mode = "test"
        bm = ex_met.test_benchmark
        carb_osent_to_exs = bm.gold
        sax_osent_to_exs = \
            SaxExtraction.get_sax_osent2_to_exs(carb_osent_to_exs)
        for osent, exs in sax_osent_to_exs.items():
            if 'Philip Russell' in osent:
                for k, sax_ex in enumerate(exs):
                    carb_ex = carb_osent_to_exs[osent][k]
                    print("sax******************", sax_ex.arg2)
                    print("carb******************", carb_ex.args[1:])

        pred_l_osent, pred_lll_ilabel, pred_ll_confi = \
            AllenTool.get_lll_ilabel_from_osent2_to_exs(sax_osent_to_exs)

        ex_met(pred_l_osent, pred_lll_ilabel, pred_ll_confi)
        score_d = ex_met.get_metric_values(mode, do_reset=True)
        print(score_d)

    # main1() # no good
    main2()
