from CCReport import *
import os
import pickle
from sax_utils import *
from AllenTool import *


class CCMetric:
    """
    similar to Openie6.metric.Conjunction

    Attributes
    ----------
    report_exact: CCReport
    report_inner: CCReport
    report_outer: CCReport
    report_whole: CCReport
    score_d: dict[str, float]
    sent_to_words: dict[str, list[str]]
    store: bool


    """

    def __init__(self, store=True, sent_to_words=None):
        """

        Parameters
        ----------
        store: bool
        sent_to_words: dict[str, list[str]]

        Returns
        -------
        CCMetric
        """
        self.store = store
        self.report_whole = CCReport("whole")
        self.report_outer = CCReport("outer")
        self.report_inner = CCReport("inner")
        self.report_exact = CCReport("exact")
        # self.n_complete = 0 # not used
        # self.n_sentence = 0 # not used
        if store:
            print("Deleting previous cc metric pkl files")
            di = CC_METRIC_STORAGE_DIR
            if os.path.exists(di + '/l_osent.pkl'):
                os.remove(di + '/l_osent.pkl')
            if os.path.exists(di + '/l_pred_ccnodes.pkl'):
                os.remove(di + '/l_pred_ccnodes.pkl')
            if os.path.exists(di + '/l_true_ccnodes.pkl'):
                os.remove(di + '/l_true_ccnodes.pkl')

        self.sent_to_words = sent_to_words

        self.score_d = CCMetric.get_zero_score_d()

    def __call__(self,
                 l_osent,  # meta data
                 lll_pred_ex_ilabel,  # predicted
                 lll_ilabel):  # ground truth
        """
        similar to Openie6.metric.Conjunction.__call__

        ccnodes  when we give it the complete ccnodes
        happens when we want to evaluate on the original system outputs
        meta_data same as osent

        Parameters
        ----------
        l_osent: list[str]
        lll_pred_ex_ilabel: list[list[list[[int]]]
        lll_ilabel: list[list[list[[int]]]

        Returns
        -------
        None

        """

        num_samples = len(lll_ilabel)
        print("Entering samples into CCMetric instance via its __call__() "
              "method.")
        print("number of samples=", num_samples)
        for k in range(num_samples):
            pred_ccnodes = CCTree(l_osent[k],
                                  lll_pred_ex_ilabel[k],
                                  calc_tree_struc=True).ccnodes
            true_ccnodes = CCTree(l_osent[k],
                                  lll_ilabel[k],
                                  calc_tree_struc=True).ccnodes

            self.report_whole.absorb_new_sample(pred_ccnodes, true_ccnodes)
            self.report_outer.absorb_new_sample(pred_ccnodes, true_ccnodes)
            self.report_inner.absorb_new_sample(pred_ccnodes, true_ccnodes)
            self.report_exact.absorb_new_sample(pred_ccnodes, true_ccnodes)

            if self.store:
                # this happens for each sample
                # print("Storing new cc metric pkl files.")
                di = CC_METRIC_STORAGE_DIR
                pickle.dump(l_osent[k], open(
                    di + '/l_osent.pkl', 'ab'))
                pickle.dump(pred_ccnodes, open(
                    di + '/l_pred_ccnodes.pkl', 'ab'))
                pickle.dump(true_ccnodes, open(
                    di + '/l_true_ccnodes.pkl', 'ab'))

    @staticmethod
    def get_zero_score_d():
        """

        Returns
        -------
        dict[str, float]

        """
        score_d = OrderedDict({
            'F1_whole': 0,
            'F1_outer': 0,
            'F1_inner': 0,
            'F1_exact': 0,
            'P_exact': 0,
            'R_exact': 0
        })
        return score_d

    def reset_reports(self):
        """
        similar to Openie6.metric.Conjunction.reset()

        Returns
        -------
        None

        """
        self.report_whole.reset()
        self.report_outer.reset()
        self.report_inner.reset()
        self.report_exact.reset()
        # self.n_complete = 0
        # self.n_sentence = 0

    def get_score_d(self, ttt, do_reset=True):
        """
        similar to Openie6.metric.Conjunction.get_metric()


        Parameters
        ----------
        ttt: str
            never used,  except as placeholder.
            ExMetric.get_score_d() has same signature and uses it.
        do_reset: bool

        Returns
        -------
        dict[str, float]

        """

        score_d = OrderedDict({
            'F1_whole': self.report_whole.overall_scorer.f1_score(),
            'F1_outer': self.report_outer.overall_scorer.f1_score(),
            'F1_inner': self.report_inner.overall_scorer.f1_score(),
            'F1_exact': self.report_exact.overall_scorer.f1_score(),
            'P_exact': self.report_exact.overall_scorer.precision(),
            'R_exact': self.report_exact.overall_scorer.recall()
        })
        self.score_d = copy(score_d)
        if do_reset:
            self.score_d = CCMetric.get_zero_score_d()
        return score_d

    def get_overall_score(self, report_category='exact'):
        """
        Similar to Openie6.metric.Conjunction.get_overall_score().

        Parameters
        ----------
        report_category: str

        Returns
        -------
        int

        """
        if report_category == 'whole':
            report = self.report_whole
        elif report_category == 'outer':
            report = self.report_outer
        elif report_category == 'inner':
            report = self.report_inner
        elif report_category == 'exact':
            report = self.report_exact
        else:
            raise ValueError(
                'invalid report_category: {}'.format(report_category))
        # print("mkcd", report, self.report_inner.overall_scorer)
        return report.overall_scorer.f1_score()


if __name__ == "__main__":
    def main():
        # dump file just saves all pred_ccnodes and true_ccnodes
        cc_met = CCMetric()
        in_fp = "tests/cc_ilabels.txt"
        with open(in_fp, "r", encoding="utf-8") as f:
            in_lines = f.readlines()

        l_osent = []
        lll_ilabel = []
        ll_ilabel = []
        for in_line in in_lines:
            if in_line:
                if in_line[0].isalpha():
                    l_osent.append(in_line.strip())
                    if ll_ilabel:
                        lll_ilabel.append(ll_ilabel)
                    ll_ilabel = []
                elif in_line[0].isdigit():
                    words = get_words(in_line)
                    # print("lkll", words)
                    ll_ilabel.append([int(x) for x in words])
        # last one
        if ll_ilabel:
            lll_ilabel.append(ll_ilabel)
        cc_met(l_osent, lll_ilabel, lll_ilabel)
        score_d = cc_met.get_score_d(ttt="train", do_reset=True)
        print(score_d)
        print("overall-exact score:", cc_met.get_overall_score("exact"))


    main()
