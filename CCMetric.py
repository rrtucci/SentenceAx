from CCReport import *
import os
import pickle


class CCMetric():
    """
    similar to metric.Conjunction


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
            if os.path.exists(dump_dir + '/tokens.pkl'):
                os.remove(dump_dir + '/tokens.pkl')
            if os.path.exists(dump_dir + '/pred_it_ccnodes.pkl'):
                os.remove(dump_dir + '/pred_it_ccnodes.pkl')
            if os.path.exists(dump_dir + '/gt_it_ccnodes.pkl'):
                os.remove(dump_dir + '/gt_it_ccnodes.pkl')

        self.fix_d = fix_d

    def __call__(self,
                 ll_pred_ccnode,
                 ll_truth_ccnode,
                 meta_data=None,
                 ccnodes=None,
                 tokens=None):
        # ccnodes  when we give it the complete ccnodes
        # happens when we want to evaluate on the original system outputs
        for i in range(len(ll_pred_ccnode)):
            if not ccnodes:
                pred_ccnodes = CCTree(
                    ll_pred_ccnode[i], meta_data[i], correct=True).ccnodes
                true_ccnodes = get_ccnodes(ll_truth_ccnode[i], meta_data[i])
            else:
                pred_ccnodes = ll_pred_ccnode[i]
                true_ccnodes = ll_truth_ccnode[i]

            self.report_whole.append(pred_ccnodes, true_ccnodes)
            self.report_outer.append(pred_ccnodes, true_ccnodes)
            self.report_inner.append(pred_ccnodes, true_ccnodes)
            self.report_exact.append(pred_ccnodes, true_ccnodes)

            if self.dump_dir:
                pickle.dump(tokens, open(self.dump_dir + '/tokens.pkl', 'ab'))
                pickle.dump(pred_ccnodes, open(
                    self.dump_dir + '/pred_it_ccnodes.pkl', 'ab'))
                pickle.dump(true_ccnodes, open(
                    self.dump_dir + '/gt_it_ccnodes.pkl', 'ab'))

    def reset(self):
        self.report_whole.reset()
        self.report_outer.reset()
        self.report_inner.reset()
        self.report_exact.reset()
        # self.n_complete = 0
        # self.n_sentence = 0

    def get_metric_values(self, reset = False):
        # similar to metric.Conjunction.get_metric()

        name_to_score = dict()
        name_to_score['P_exact'] = self.report_exact.overall_scorer.precision()
        name_to_score['R_exact'] = self.report_exact.overall_scorer.recall()
        name_to_score['F1_whole'] = self.report_whole.overall_scorer.f1_score()
        name_to_score['F1_outer'] = self.report_outer.overall_scorer.f1_score()
        name_to_score['F1_inner'] = self.report_inner.overall_scorer.f1_score()
        name_to_score['F1_exact'] = self.report_exact.overall_scorer.f1_score()
        if reset:
            self.reset()
        return name_to_score

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
        similar to data_processing.load_conj_mapping()
        Our fix_d is similar to mapping and conj_mapping.
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
