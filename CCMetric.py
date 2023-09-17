from CCReport import *
import os
import pickle
from sax_utils import *


class CCMetric():
    """
    similar to Openie6.metric.Conjunction


    """

    def __init__(self, dump_dir=None, fix_d=None):
        self.report_whole = CCReport("whole")
        self.report_outer = CCReport("outer")
        self.report_inner = CCReport("inner")
        self.report_exact = CCReport("exact")
        # self.n_complete = 0 # not used
        # self.n_sentence = 0 # not used
        self.dump_dir = dump_dir
        if self.dump_dir:
            if os.path.exists(dump_dir + '/l_osent.pkl'):
                os.remove(dump_dir + '/l_osent.pkl')
            if os.path.exists(dump_dir + '/pred_ccnodes.pkl'):
                os.remove(dump_dir + '/pred_ccnodes.pkl')
            if os.path.exists(dump_dir + '/true_ccnodes.pkl'):
                os.remove(dump_dir + '/true_ccnodes.pkl')

        self.fix_d = fix_d

    def __call__(self,
                 l_osent,
                 pred_lll_ilabel,
                 true_lll_ilabel,
                 ccnodes=None):
        # ccnodes  when we give it the complete ccnodes
        # happens when we want to evaluate on the original system outputs
        # meta_data same as l_osent

        if not ccnodes:
            pred_ccnodes = CCTree(l_osent, pred_lll_ilabel).ccnodes
            true_ccnodes = CCTree(l_osent, true_lll_ilabel).ccnodes
        else:
            pred_ccnodes = pred_lll_ilabel
            true_ccnodes = true_lll_ilabel

        self.report_whole.grow(pred_ccnodes, true_ccnodes)
        self.report_outer.grow(pred_ccnodes, true_ccnodes)
        self.report_inner.grow(pred_ccnodes, true_ccnodes)
        self.report_exact.grow(pred_ccnodes, true_ccnodes)

        if self.dump_dir:
            pickle.dump(l_osent,
                        open(self.dump_dir + '/l_osent.pkl', 'ab'))
            pickle.dump(pred_ccnodes, open(
                self.dump_dir + '/pred_ccnodes.pkl', 'ab'))
            pickle.dump(true_ccnodes, open(
                self.dump_dir + '/true_ccnodes.pkl', 'ab'))

    def reset(self):
        self.report_whole.reset()
        self.report_outer.reset()
        self.report_inner.reset()
        self.report_exact.reset()
        # self.n_complete = 0
        # self.n_sentence = 0

    def get_metric_values(self, do_reset=False):
        # similar to Openie6.metric.Conjunction.get_metric()

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

    def get_overall_score(self, report_name='exact'):
        if report_name == 'whole':
            report = self.report_whole
        elif report_name == 'outer':
            report = self.report_outer
        elif report_name == 'inner':
            report = self.report_inner
        elif report_name == 'exact':
            report = self.report_exact
        else:
            raise ValueError('invalid report_name: {}'.format(report_name))
        return report.overall_scorer.f1_score

    @staticmethod
    def load_fix_d(fix_fp):
        """
        similar to Openie6.data_processing.load_conj_mapping()
        Our fix_d is similar to Openie6 mapping and conj_mapping.
        This method works equally well for ExMetric.fix_d and CCMetric.fix_d

        Returns
        -------

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
        cc_met = CCMetric()
        cc_met(l_osent, pred_lll_ilabel, true_lll_ilabel)
        score_d = cc_met.get_metric_values(do_reset=True)
        print(score_d)
