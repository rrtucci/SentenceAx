from CCReport import *
import os
import pickle


class CCMetric():
    def __init__(self, dump_dir=None):
        self.report_whole = CCReport("whole")
        self.report_outer = CCReport("outer")
        self.report_inner = CCReport("inner")
        self.report_exact = CCReport("exact")
        self.n_complete = 0
        self.n_sentence = 0
        self.dump_dir = dump_dir
        if self.dump_dir != None:
            if os.path.exists(dump_dir + '/tokens.pkl'):
                os.remove(dump_dir + '/tokens.pkl')
            if os.path.exists(dump_dir + '/pred_it_ccnodes.pkl'):
                os.remove(dump_dir + '/pred_it_ccnodes.pkl')
            if os.path.exists(dump_dir + '/gt_it_ccnodes.pkl'):
                os.remove(dump_dir + '/gt_it_ccnodes.pkl')

    def __call__(self, ll_pred_ccnode, ll_truth_ccnode, meta_data=None,
                 ccnodes=None):
        # ccnodes != None when we give it the complete ccnodes
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
        self.n_complete = 0
        self.n_sentence = 0

    def get_metric(self, reset: bool = False, mode=None):
        pairs = [("whole", self.report_whole),
                    ("outer", self.report_outer),
                    ("inner", self.report_inner),
                    ("exact", self.report_exact)]

        all_metrics = dict()
        all_metrics['P_exact'] = pairs[3][1].overall_scorer.precision()
        all_metrics['R_exact'] = pairs[3][1].overall_scorer.recall()
        all_metrics['F1_whole'] = pairs[0][1].overall_scorer.f1_score()
        all_metrics['F1_outer'] = pairs[1][1].overall_scorer.f1_score()
        all_metrics['F1_inner'] = pairs[1][1].overall_scorer.f1_score()
        all_metrics['F1_exact'] = pairs[3][1].overall_scorer.f1_score()
        if reset:
            self.reset()
        return all_metrics

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
