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

    This class does scoring for task="ex". Class CCMetric does scoring for
    task="cc".
    
    Attributes 
    ----------
    tune_benchmark: Benchmark
    sent_to_sent: dict[str, str]
    matchingFunc: Matcher.binary_linient_tuple_match
    osentL_to_exs: dict[str, listt[SaxExtraction]]
    score_d: dict[str, float]
    test_benchmark: Benchmark
    input_carb_exs: bool
    
    """

    def __init__(self,
                 osentL_to_exs=None,
                 sent_to_sent=None,
                 input_carb_exs=False):
        """
        Constructor
        
        Parameters
        ----------
        osentL_to_exs: dict[str, list[SaxExtraction]]
        sent_to_sent: dict[str, str]
        input_carb_exs: bool
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
        self.input_carb_exs = input_carb_exs

    def __call__(self,
                 l_osentL,  # meta data
                 lll_ilabel,  # predictions
                 ll_confi):  # scores
        """
        similar to Openie6.metric.Carb.__call__

        A __call__() method is a new chance to load attributes into the
        class after the __init__() has been called.

        Whereas __init__() is called only once, __call__() can be called
        multiple times for the same class instance. For ExMetric,
        this __call__() method is called for each batch of an epoch. Each
        time, self.osentL_to_exs grows.  At the end of an epoch,
        get_score_d() is called. That method averages, saves and resets the
        cummulative scores, before commencing a new epoch.


        Parameters
        ----------
        l_osentL: list[str]
        lll_ilabel: list[list[list[int]]]
        ll_confi: list[list[float]]

        Returns
        -------
        None

        """
        if VERBOSE: print("Entering ExMetric.__call__() method.")
        assert not self.input_carb_exs
        if VERBOSE: print("ll_confi", ll_confi)
        if VERBOSE: print("len(self.osentL_to_exs)", len(self.osentL_to_exs))
        dominant_d = \
            AllenTool.get_osent2_to_exs_from_lll_ilabel(
                l_osentL,
                lll_ilabel,
                ll_confi,
                self.sent_to_sent)
        self.osentL_to_exs = merge_dicts(dominant_d, self.osentL_to_exs)
        if VERBOSE:
            print("len(self.osentL_to_exs) after merge",
                  len(self.osentL_to_exs))

    @staticmethod
    def get_zero_score_d():
        """
        This method returns a new copy of the `score_d` dictionary with all
        values zero.

        Returns
        -------
        dict[str, float]

        """
        score_d = OrderedDict({'AUC': 0.0,
                               'F1': 0.0,
                               'last_F1': 0.0})
        return score_d

    def reset_score_d(self):
        """
        similar to Openie6.metric.Carb.reset()

        Unlike the method get_zero_score_d(), this method does not create a
        new `score_d` dictionary. Instead, it sets to zero all values of the
        existing `self.score_d`.

        Returns
        -------
        None

        """
        for name in self.score_d.keys():
            self.score_d[name] = 0.0

    def reset_exs_dict(self):
        """
        similar to Openie6.metric.Carb.reset()

        This method sets to {} (resets) self.osentL_to_exs.

        Note that reset_exs_dict() and reset_score_d() are separate methods.
        Openie6 lumps them together.

        Returns
        -------
        None

        """
        self.osentL_to_exs = {}

    def get_score_d(self, ttt, do_reset=True):
        """
        similar to Openie6.metric.Carb.get_metric()

        This method returns the current `score_d`. It calls reset_exs_dict()
        iff do_reset=True.
        
        Parameters
        ----------
        ttt: str
        do_reset: bool

        Returns
        -------
        dict[str, float]

        """
        if VERBOSE: print("Entering ExMetric.get_score_d() method.")

        def fun(x):
            # if x is sax extraction
            if hasattr(x, "confi"):
                return x.confi
            # if ex is carb extraction
            elif hasattr(x, "confidence"):
                return x.confidence

        assert self.osentL_to_exs
        for osentL, exs in self.osentL_to_exs.items():
            # print("confi", [fun(ex) for ex in exs])
            if VERBOSE: print("len(exs)", len(exs))
            self.osentL_to_exs[osentL] = \
                sorted(exs,
                       key=fun,
                       reverse=True)[:EX_NUM_DEPTHS]
        if not self.input_carb_exs:
            osent_to_exs = undoL(self.osentL_to_exs)
            carb_osent_to_exs = \
                SaxExtraction.get_carb_osent2_to_exs(osent_to_exs)
        else:
            carb_osent_to_exs = undoL(self.osentL_to_exs)

        # no /dev/null in Windows
        # out_fp = "/dev/null"
        out_fp = "dev_null.txt"

        # this means that the exs of
        # EXTAGS_TEST_FP and EXTAGS_TUNE_FP
        # are ignored. In the actual files, they are
        # just a single extraction with only NONE extags
        if ttt == "tune":
            bmark = self.tune_benchmark
        elif ttt == "test":
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
        # necessary because __call__ only
        # appends to self.osentL_to_exs
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
        ex_met = ExMetric(carb_osent_to_exs, input_carb_exs=True)
        # unnecessary
        # ex_met()
        score_d = ex_met.get_score_d(ttt="test", do_reset=True)
        print(score_d)


    # main1() # no good
    main2()
    main3()
