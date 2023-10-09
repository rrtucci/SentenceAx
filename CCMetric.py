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
    store_dir: str
    fix_d: dict[str, str]
    report_exact: CCReport
    report_inner: CCReport
    report_outer: CCReport
    report_whole: CCReport


    """

    def __init__(self, store_dir=None, fix_d=None):
        """

        Parameters
        ----------
        store_dir: str
        fix_d: dict[str, str]

        Returns
        -------
        CCMetric
        """
        self.report_whole = CCReport("whole")
        self.report_outer = CCReport("outer")
        self.report_inner = CCReport("inner")
        self.report_exact = CCReport("exact")
        # self.n_complete = 0 # not used
        # self.n_sentence = 0 # not used
        self.store_dir = store_dir
        if self.store_dir:
            if os.path.exists(store_dir + '/l_osent.pkl'):
                os.remove(store_dir + '/l_osent.pkl')
            if os.path.exists(store_dir + '/l_pred_ccnodes.pkl'):
                os.remove(store_dir + '/l_pred_ccnodes.pkl')
            if os.path.exists(store_dir + '/l_true_ccnodes.pkl'):
                os.remove(store_dir + '/l_true_ccnodes.pkl')

        self.fix_d = fix_d

    def __call__(self,
                 l_osent,  # meta data
                 lll_pred_ex_ilabel,  # predicted
                 lll_ex_ilabel):  # ground truth
        """

        ccnodes  when we give it the complete ccnodes
        happens when we want to evaluate on the original system outputs
        meta_data same as osent

        Parameters
        ----------
        l_osent: list[str]
        lll_pred_ex_ilabel: list[list[list[[int]]]
        lll_ex_ilabel: list[list[list[[int]]]

        Returns
        -------
        None

        """

        num_samples = len(lll_ex_ilabel)
        print("Entering samples into CCMetric instance via its __call__() "
              "method.")
        print("number of samples=", num_samples)
        for k in range(num_samples):
            pred_ccnodes = CCTree(l_osent[k],
                                  lll_pred_ex_ilabel[k],
                                  calc_tree_struc=True).ccnodes
            true_ccnodes = CCTree(l_osent[k],
                                  lll_ex_ilabel[k],
                                  calc_tree_struc=True).ccnodes

            self.report_whole.absorb_new_sample(pred_ccnodes, true_ccnodes)
            self.report_outer.absorb_new_sample(pred_ccnodes, true_ccnodes)
            self.report_inner.absorb_new_sample(pred_ccnodes, true_ccnodes)
            self.report_exact.absorb_new_sample(pred_ccnodes, true_ccnodes)

            if self.store_dir:
                pickle.dump(l_osent[k],
                            open(self.store_dir + '/l_osent.pkl', 'ab'))
                pickle.dump(pred_ccnodes, open(
                    self.store_dir + '/l_pred_ccnodes.pkl', 'ab'))
                pickle.dump(true_ccnodes, open(
                    self.store_dir + '/l_true_ccnodes.pkl', 'ab'))

    def reset(self):
        """

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

    def get_score_d(self, do_reset=False):
        """
        similar to Openie6.metric.Conjunction.get_metric()


        Parameters
        ----------
        do_reset: bool

        Returns
        -------
        dict[str, float]

        """

        score_d = dict()
        score_d['P_exact'] = self.report_exact.overall_scorer.precision()
        score_d['R_exact'] = self.report_exact.overall_scorer.recall()
        score_d['F1_whole'] = self.report_whole.overall_scorer.f1_score()
        score_d['F1_outer'] = self.report_outer.overall_scorer.f1_score()
        score_d['F1_inner'] = self.report_inner.overall_scorer.f1_score()
        score_d['F1_exact'] = self.report_exact.overall_scorer.f1_score()
        if do_reset:
            self.reset()
        return score_d

    def get_overall_score(self, report_category='exact'):
        """

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

    @staticmethod
    def load_fix_d(fix_fp):
        """
        similar to Openie6.data_processing.load_conj_mapping()
        Our fix_d is similar to Openie6 mapping and conj_mapping.
        This method works equally well for ExMetric.fix_d and CCMetric.fix_d

        Returns
        -------
        dict[str, str]

        """
        fix_d = {}
        with open(fix_fp, "r") as f:
            content = f.read()
            fixed_sent = ''
            for sample in content.split('\n\n'):
                for i, line in enumerate(sample.strip('\n').split('\n')):
                    if i == 0:
                        fixed_sent = line
                    else:
                        fix_d[line] = fixed_sent
        return fix_d


if __name__ == "__main__":
    def main():
        # dump file just saves all pred_ccnodes and true_ccnodes
        cc_met = CCMetric()
        in_fp = "testing_files/cc_ilabels.txt"
        with open(in_fp, "r", encoding="utf-8") as f:
            in_lines = f.readlines()

        l_osent = []
        lll_ex_ilabel = []
        ll_ilabel = []
        for in_line in in_lines:
            if in_line:
                if in_line[0].isalpha():
                    l_osent.append(in_line.strip())
                    if ll_ilabel:
                        lll_ex_ilabel.append(ll_ilabel)
                    ll_ilabel = []
                elif in_line[0].isdigit():
                    words = get_words(in_line)
                    # print("lkll", words)
                    ll_ilabel.append([int(x) for x in words])
        # last one
        if ll_ilabel:
            lll_ex_ilabel.append(ll_ilabel)
        cc_met(l_osent, lll_ex_ilabel, lll_ex_ilabel)
        score_d = cc_met.get_score_d(do_reset=False)
        print(score_d)
        # this gives nan if do_reset = True
        print("overall-exact score:", cc_met.get_overall_score("exact"))


    main()
