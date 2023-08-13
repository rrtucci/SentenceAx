from CCReportCard import *
import os
import pickle


class CCMetric():
    def __init__(self, dump_dir=None):
        self.sc_whole = CCReportCard("whole")
        self.sc_outer = CCReportCard("outer")
        self.sc_inner = CCReportCard("inner")
        self.sc_exact = CCReportCard("exact")
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

    def __call__(self, predictions, ground_truth, meta_data=None,
                 ccnodes=None):
        # ccnodes != None when we give it the complete ccnodes
        # happens when we want to evaluate on the original system outputs
        for i in range(len(predictions)):
            if not ccnodes:
                pred_ccnodes = get_ccnodes(
                    predictions[i], meta_data[i], correct=True)
                true_ccnodes = get_ccnodes(ground_truth[i], meta_data[i])
            else:
                pred_ccnodes = predictions[i]
                true_ccnodes = ground_truth[i]

            self.sc_whole.append(pred_ccnodes, true_ccnodes)
            self.sc_outer.append(pred_ccnodes, true_ccnodes)
            self.sc_inner.append(pred_ccnodes, true_ccnodes)
            self.sc_exact.append(pred_ccnodes, true_ccnodes)

            if self.dump_dir:
                pickle.dump(tokens, open(self.dump_dir + '/tokens.pkl', 'ab'))
                pickle.dump(pred_ccnodes, open(
                    self.dump_dir + '/pred_it_ccnodes.pkl', 'ab'))
                pickle.dump(true_ccnodes, open(
                    self.dump_dir + '/gt_it_ccnodes.pkl', 'ab'))
        return

    def reset(self):
        self.sc_whole.reset()
        self.sc_outer.reset()
        self.sc_inner.reset()
        self.sc_exact.reset()
        self.n_complete = 0
        self.n_sentence = 0

    def get_metric(self, reset: bool = False, mode=None):
        ccscorers = [("whole", self.sc_whole),
                    ("outer", self.sc_outer),
                    ("inner", self.sc_inner),
                    ("exact", self.sc_exact)]

        all_metrics = dict()
        all_metrics['P_exact'] = ccscorers[3][1].overall.precision
        all_metrics['R_exact'] = ccscorers[3][1].overall.recall
        all_metrics['F1_whole'] = ccscorers[0][1].overall.f1_score
        all_metrics['F1_outer'] = ccscorers[1][1].overall.f1_score
        all_metrics['F1_inner'] = ccscorers[1][1].overall.f1_score
        all_metrics['F1_exact'] = ccscorers[3][1].overall.f1_score
        if reset:
            self.reset()
        return all_metrics

    def get_overall_score(self, metric='exact'):
        if metric == 'whole':
            ccscorer = self.sc_whole
        elif metric == 'outer':
            ccscorer = self.sc_outer
        elif metric == 'inner':
            ccscorer = self.sc_inner
        elif metric == 'exact':
            ccscorer = self.sc_exact
        else:
            raise ValueError('invalid metric: {}'.format(metric))
        return ccscorer.overall.f1_score
