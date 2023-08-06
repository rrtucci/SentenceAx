from CCScorer import *
import os
import pickle


class CCMetric():
    def __init__(self, dump_dir=None):
        self.ccscorer_whole = CCScorer("whole")
        self.ccscorer_outer = CCScorer("outer")
        self.ccscorer_inner = CCScorer("inner")
        self.ccscorer_exact = CCScorer("exact")
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
                 ccnodes=False):
        # ccnodes == True when we give it the complete ccnodes
        # happens when we want to evaluate on the original system outputs
        for i in range(len(predictions)):
            if not ccnodes:
                pred_ccnodes = get_ccnodes(
                    predictions[i], meta_data[i], correct=True)
                true_ccnodes = get_ccnodes(ground_truth[i], meta_data[i])
            else:
                pred_ccnodes = predictions[i]
                true_ccnodes = ground_truth[i]

            self.ccscorer_whole.append(pred_ccnodes, true_ccnodes)
            self.ccscorer_outer.append(pred_ccnodes, true_ccnodes)
            self.ccscorer_inner.append(pred_ccnodes, true_ccnodes)
            self.ccscorer_exact.append(pred_ccnodes, true_ccnodes)

            if self.dump_dir:
                pickle.dump(tokens, open(self.dump_dir + '/tokens.pkl', 'ab'))
                pickle.dump(pred_ccnodes, open(
                    self.dump_dir + '/pred_it_ccnodes.pkl', 'ab'))
                pickle.dump(true_ccnodes, open(
                    self.dump_dir + '/gt_it_ccnodes.pkl', 'ab'))
        return

    def reset(self):
        self.ccscorer_whole.reset()
        self.ccscorer_outer.reset()
        self.ccscorer_inner.reset()
        self.ccscorer_exact.reset()
        self.n_complete = 0
        self.n_sentence = 0

    def get_metric(self, reset: bool = False, mode=None):
        ccscorers = [("whole", self.ccscorer_whole),
                    ("outer", self.ccscorer_outer),
                    ("inner", self.ccscorer_inner),
                    ("exact", self.ccscorer_exact)]

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
            ccscorer = self.ccscorer_whole
        elif metric == 'outer':
            ccscorer = self.ccscorer_outer
        elif metric == 'inner':
            ccscorer = self.ccscorer_inner
        elif metric == 'exact':
            ccscorer = self.ccscorer_exact
        else:
            raise ValueError('invalid metric: {}'.format(metric))
        return ccscorer.overall.f1_score
