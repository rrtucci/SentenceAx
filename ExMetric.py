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
        if not osentL_to_exs:
            # important to initialize it as empty dictionary if using
            # __call__
            osentL_to_exs = {}
        self.osentL_to_exs = osentL_to_exs
        # self.ll_osent_pos_bool = [] # not used
        # self.ll_osent_verb_bool = [] # not used
        self.score_d = ExMetric.get_zero_score_d()
        self.sent_to_sent = sent_to_sent
        self.use_carb_ex = use_carb_ex

    @staticmethod
    def get_osent2_to_exs_from_lll_ilabel(l_osent2,
                                          lll_ilabel,
                                          ll_confi,
                                          sent_to_sent):
        """
        similar to Openie6.metric.Carb.__call__()

        This method takes as `lll_ilabel` and other variables and returns

        `osent2_to_exs`

        osent = original sentence
        osentL = osent + UNUSED_TOKEN_STR

        This method does not care internally whether we are using `osentL,
        lll_ilabels` or `osent, lll_ilabels`. that is why we are introducing
        the symbol `osent2`, which can stand for `osent` or `osentL`


        Parameters
        ----------
        l_osent2: list[str]
        lll_ilabel: list[list[list[int]]]
        ll_confi: list[list[float]]
        sent_to_sent: dict[str, str]
            a dictionary that makes small fixes on osent2

        Returns
        -------
        dict[str, list[SaxExtraction]]

        """

        osent2_to_exs = {}
        for sam_id, osent2 in enumerate(l_osent2):
            add_key_to_target_d(key=osent2,
                                fix_d=sent_to_sent,
                                target_d=osent2_to_exs)

            num_exs = len(ll_confi[sam_id])
            for depth in range(num_exs):
                ilabels = lll_ilabel[sam_id][depth]
                # all ilabels=0 once no more extractions
                if sum(ilabels) == 0:
                    break
                ex0 = SaxExtraction.get_ex_from_ilabels(
                    ilabels,
                    osent2,
                    ll_confi[sam_id][depth])
                if ex0.arg1 and ex0.rel:
                    add_key_value_pair_to_target_d(
                        key=osent2,
                        value=ex0,
                        fix_d=sent_to_sent,
                        target_d=osent2_to_exs)
        return osent2_to_exs

    def __call__(self,
                 l_osentL,  # meta data
                 lll_ilabel,  # predictions
                 ll_confi):  # scores
        """
        similar to Openie6.metric.Carb.__call__

        This method  can be called multiple times for the same class instance.
        Each time, self.osentL_to_exs grows.

        Parameters
        ----------
        l_osentL: list[str]
        lll_ilabel: list[list[list[int]]]
        ll_confi: list[list[float]]

        Returns
        -------
        None

        """
        if DEBUG: print("Entering ExMetric.__call__() method.")
        assert not self.use_carb_ex
        if DEBUG: print("ll_confi", ll_confi)
        if DEBUG: print("len(self.osentL_to_exs)", len(self.osentL_to_exs))
        dominant_d = \
            ExMetric.get_osent2_to_exs_from_lll_ilabel(
                l_osentL,
                lll_ilabel,
                ll_confi,
                self.sent_to_sent)
        self.osentL_to_exs = merge_dicts(dominant_d, self.osentL_to_exs)
        if DEBUG:
            print("len(self.osentL_to_exs) after merge",
                  len(self.osentL_to_exs))

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

    def reset_exs_dict(self):
        """
        similar to Openie6.metric.Carb.reset()

        Returns
        -------
        None

        """
        self.osentL_to_exs = {}

    def reset_score_d(self):
        """
        similar to Openie6.metric.Carb.reset()
        
        Returns
        -------
        None

        """
        for name in self.score_d.keys():
            self.score_d[name] = 0.0

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
        if DEBUG: print("Entering ExMetric.get_score_d() method.")

        def fun(x):
            # if x is sax extraction
            if hasattr(x, "confi"):
                return x.confi
            # if ex is carb extraction
            elif hasattr(x, "confidence"):
                return x.confidence

        for osentL, exs in self.osentL_to_exs.items():
            # print("confi", [fun(ex) for ex in exs])
            if DEBUG: print("len(exs)", len(exs))
            self.osentL_to_exs[osentL] = \
                sorted(exs,
                       key=fun,
                       reverse=True)[:EX_NUM_DEPTHS]
        if not self.use_carb_ex:
            osent_to_exs = undoL(self.osentL_to_exs)
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

        self.score_d = OrderedDict(
            {'AUC': auc,
             'F1': optimal_f1_point[2],
             'last_F1': last_f1_point[2]})
        if do_reset:
            self.reset_exs_dict()
        return self.score_d


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
        sax_osentL_to_exs = redoL(sax_osent_to_exs)
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
