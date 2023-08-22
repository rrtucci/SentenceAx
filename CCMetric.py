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
        self.complete = 0
        self.sentence = 0
        self.dump_dir = dump_dir
        if self.dump_dir :
            if os.path.exists(dump_dir + '/tokens.pkl'):
                os.remove(dump_dir + '/tokens.pkl')
            if os.path.exists(dump_dir + '/pred_it_ccnodes.pkl'):
                os.remove(dump_dir + '/pred_it_ccnodes.pkl')
            if os.path.exists(dump_dir + '/gt_it_ccnodes.pkl'):
                os.remove(dump_dir + '/gt_it_ccnodes.pkl')

        self.fix_d = fix_d

    def __call__(self, ll_pred_ccnode, ll_truth_ccnode, meta_data=None,
                 ccnodes=None):
        # ccnodes  when we give it the complete ccnodes
        # happens when we want to evaluate on the original system outputs
        for i in range(len(ll_pred_ccnode)):
            if not ccnodes:
                pred_ccnodes = get_ccnodes(
                    ll_pred_ccnode[i], meta_data[i], correct=True)
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
        return

    def reset(self):
        self.report_whole.reset()
        self.report_outer.reset()
        self.report_inner.reset()
        self.report_exact.reset()
        self.complete = 0
        self.sentence = 0

    def get_metric_values(self, reset: bool = False, mode=None):
        # similar to metric.Conjunction.get_metric()
        pairs = [("whole", self.report_whole),
                    ("outer", self.report_outer),
                    ("inner", self.report_inner),
                    ("exact", self.report_exact)]

        l_metric_value = dict()
        l_metric_value['P_exact'] = pairs[3][1].overall_scorer.precision()
        l_metric_value['R_exact'] = pairs[3][1].overall_scorer.recall()
        l_metric_value['F1_whole'] = pairs[0][1].overall_scorer.f1_score()
        l_metric_value['F1_outer'] = pairs[1][1].overall_scorer.f1_score()
        l_metric_value['F1_inner'] = pairs[1][1].overall_scorer.f1_score()
        l_metric_value['F1_exact'] = pairs[3][1].overall_scorer.f1_score()
        if reset:
            self.reset()
        return l_metric_value

    def get_overall_score(self, metric='exact'):
        if metric == 'whole':
            report = self.report_whole
        elif metric == 'outer':
            report = self.report_outer
        elif metric == 'inner':
            report = self.report_inner
        elif metric == 'exact':
            report = self.report_exact
        else:
            raise ValueError('invalid metric: {}'.format(metric))
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
