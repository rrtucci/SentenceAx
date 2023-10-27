from carb_subset.matcher import Matcher
from carb_subset.carb import Benchmark
import re
from Params import *
from sax_utils import *
from SaxExtraction import *
from words_tags_ilabels_translation import *
from MOutput import *
from AllenTool import *


class ExMetric:
    """
    similar to Openie6.metric.Carb
    
    Attributes 
    ----------
    tune_benchmark: Benchmark
    sent_to_sent: dict[str, str]
    matchingFunc: Matcher.binary_linient_tuple_match
    osentL_to_exs: dict[str, listt[SaxExtraction]]
    score_d: dict[str, float]
    test_benchmark: Benchmark
    use_carb_ex: bool
    
    """

    def __init__(self,
                 osentL_to_exs=None,
                 sent_to_sent=None,
                 use_carb_ex=False):
        """
        
        Parameters
        ----------
        osentL_to_exs: dict[str, list[SaxExtraction]]
        sent_to_sent: dict[str, str]
        use_carb_ex: bool
        """
        self.tune_benchmark = Benchmark('carb_subset/data/gold/dev.tsv')
        self.test_benchmark = Benchmark('carb_subset/data/gold/test.tsv')
        self.matchingFunc = Matcher.binary_linient_tuple_match
        self.osentL_to_exs = osentL_to_exs
        # self.ll_osent_pos_bool = [] # not used
        # self.ll_osent_verb_bool = [] # not used
        self.score_d = ExMetric.get_zero_score_d()
        self.sent_to_sent = sent_to_sent
        self.use_carb_ex = use_carb_ex

    def __call__(self,
                 l_osentL,  # meta data
                 lll_ilabel,  # predictions
                 ll_confi):  # scores
        """
        similar to Openie6.metric.Carb.__call__

        Parameters
        ----------
        l_osentL: list[str]
        lll_ilabel: list[list[list[int]]]
        ll_confi: list[list[float]]

        Returns
        -------
        None

        """
        assert not self.use_carb_ex
        if not self.osentL_to_exs:
            self.osentL_to_exs = \
                AllenTool.get_osent2_to_exs_from_lll_ilabel(
                    l_osentL,
                    lll_ilabel,
                    ll_confi,
                    self.sent_to_sent)
        else:
            assert False, "This __call__ is redundant. osentL_to_exs" \
                          " has already been entered in the " \
                          "ExMetric constructor"
        print("Just entered samples into ExMetric instance via its "
              "__call__() method.")
        print("number of samples=", len(lll_ilabel))

    @staticmethod
    def get_zero_score_d():
        """

        Returns
        -------
        dict[str, float]

        """
        score_d = OrderedDict({'AUC': 0.0,
                               'F1': 0.0,
                               'last_F1': 0.0})
        return score_d

    def reset(self):
        """
        similar to Openie6.metric.Carb.reset()
        
        Returns
        -------
        None

        """
        self.osentL_to_exs = {}
        self.score_d = ExMetric.get_zero_score_d()

    def get_score_d(self, ttt, do_reset=True):
        """
        similar to Openie6.metric.Carb.get_metric()
        
        Parameters
        ----------
        ttt: str
        do_reset: bool

        Returns
        -------
        dict[str, float]

        """

        def fun(x):
            if hasattr(x, "confi"):
                return x.confi
            elif hasattr(x, "confidence"):
                return x.confidence

        if EX_NUM_DEPTHS:
            for osentL in self.osentL_to_exs:
                self.osentL_to_exs[osentL] = \
                    sorted(self.osentL_to_exs[osentL],
                           key=fun,
                           reverse=True)[:EX_NUM_DEPTHS]
        if not self.use_carb_ex:
            osent_to_exs = \
                SaxExtraction.shorten_osentL_to_exs(self.osentL_to_exs)
            carb_osent_to_exs = \
                SaxExtraction.get_carb_osent2_to_exs(osent_to_exs)
        else:
            carb_osent_to_exs = self.osentL_to_exs

        # no /dev/null in Windows
        # out_fp = "/dev/null"
        out_fp = "dev_null.txt"
        if ttt == "tune":
            bmark = self.tune_benchmark
        elif ttt == 'test':
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

        score_d = OrderedDict({'AUC': auc,
                               'F1': optimal_f1_point[2],
                               'last_F1': last_f1_point[2]})
        self.score_d = copy(score_d)
        if ttt == "tune" and do_reset:
            # this resets score_d
            self.reset()
        return score_d


if __name__ == "__main__":
    # main1() didn't work.
    # "carb_subset/data/test_gold_allennlp_format.txt"
    # and "carb_subset/data/gold/test.tsv:" are DIFFERENT
    # def main1():
    #     in_fp = "carb_subset/data/test_gold_allennlp_format.txt"
    #     at = AllenTool(in_fp)
    #     osent_to_exs = SaxExtraction.shorten_osentL_to_exs(
    #         at.osentL_to_exs)
    #     ex_met = ExMetric()
    #     ttt = "test"
    #     pred_l_osent, pred_lll_ilabel, pred_ll_confi = \
    #         AllenTool.get_lll_ilabel_from_osent2_to_exs(osent_to_exs)
    #
    #     ex_met(pred_l_osent, pred_lll_ilabel, pred_ll_confi)
    #     score_d = ex_met.get_score_d(ttt, do_reset=True)
    #     print(score_d)

    def main2():
        bm = Benchmark('carb_subset/data/gold/test.tsv')
        carb_osent_to_exs = bm.gold
        sax_osent_to_exs = \
            SaxExtraction.get_sax_osent2_to_exs(carb_osent_to_exs)
        sax_osentL_to_exs = \
            SaxExtraction.elongate_osent_to_exs(sax_osent_to_exs)
        ex_met = ExMetric(sax_osentL_to_exs)
        # unnecessary
        # ex_met()
        ttt = "test"
        score_d = ex_met.get_score_d(ttt, do_reset=True)
        print(score_d)


    def main3():
        bm = Benchmark('carb_subset/data/gold/test.tsv')
        carb_osent_to_exs = bm.gold
        ex_met = ExMetric(carb_osent_to_exs, use_carb_ex=True)
        # unnecessary
        # ex_met()
        score_d = ex_met.get_score_d(ttt="test", do_reset=True)
        print(score_d)


    # main1() # no good
    main2()
    main3()
