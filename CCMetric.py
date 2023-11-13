from CCScoreManager import *
import os
import pickle
from sax_utils import *
from AllenTool import *


class CCMetric:
    """
    similar to Openie6.metric.Conjunction
    
    This class does scoring for task="cc". Class ExMetric does scoring for 
    task="ex".
    
    This class stores scores of model weights. There are 4 types of scoring 
    managers, called CCScoreManager's: "manager_whole", "manager_outer", 
    "manager_inner" and "manager_exact". Each of these uses a different 
    scoring procedure.

    Setting the parameter `save` to True makes this class store the scores 
    for the current set of weights.

    Attributes
    ----------
    manager_exact: CCScoreManager
    manager_inner: CCScoreManager
    manager_outer: CCScoreManager
    manager_whole: CCScoreManager
    score_d: dict[str, float]
    save: bool


    """

    def __init__(self):
        """
        Constructor

        """
        self.save = CC_METRIC_SAVE
        self.manager_whole = CCScoreManager("whole")
        self.manager_outer = CCScoreManager("outer")
        self.manager_inner = CCScoreManager("inner")
        self.manager_exact = CCScoreManager("exact")
        # self.n_complete = 0 # not used
        # self.n_sentence = 0 # not used
        if self.save:
            print("CCMetric deleting previous pkl files.")
            di = CC_METRIC_STORAGE_DIR
            if os.path.exists(di + '/l_osent.pkl'):
                os.remove(di + '/l_osent.pkl')
            if os.path.exists(di + '/l_pred_ccnodes.pkl'):
                os.remove(di + '/l_pred_ccnodes.pkl')
            if os.path.exists(di + '/l_true_ccnodes.pkl'):
                os.remove(di + '/l_true_ccnodes.pkl')

        self.score_d = CCMetric.get_zero_score_d()

    def __call__(self,
                 l_osent,  # Openie6.meta_data
                 lll_pred_ilabel,  # Openie6.predictions
                 lll_ilabel):  # Openie6.ground_truth
        """
        similar to Openie6.metric.Conjunction.__call__

        A __call__() method is a new chance to load attributes into the 
        class after the __init__() has been called.

        Whereas __init__() is called only once, __call__() can be called
        multiple times for the same class instance. For CCMetric,
        this __call__() method is called for each batch of an epoch. Each
        time, the scores in the managers grow. At the end of an epoch,
        get_score_d() is called. That method averages, saves and resets the
        cummulative scores, before commencing a new epoch.


        Parameters
        ----------
        l_osent: list[str]
        lll_pred_ilabel: list[list[list[[int]]]
        lll_ilabel: list[list[list[[int]]]

        """
        if VERBOSE: print("Entering CCMetric.__call__() method.")
        num_samples = len(lll_ilabel)
        print("number of samples=", num_samples)
        for k in range(num_samples):
            pred_ccnodes = CCTree(l_osent[k],
                                  lll_pred_ilabel[k],
                                  calc_tree_struc=True).ccnodes
            true_ccnodes = CCTree(l_osent[k],
                                  lll_ilabel[k],
                                  calc_tree_struc=True).ccnodes

            self.manager_whole.absorb_new_sample(pred_ccnodes, true_ccnodes)
            self.manager_outer.absorb_new_sample(pred_ccnodes, true_ccnodes)
            self.manager_inner.absorb_new_sample(pred_ccnodes, true_ccnodes)
            self.manager_exact.absorb_new_sample(pred_ccnodes, true_ccnodes)

            if self.save:
                # we append to pickle files for each sample.
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
        This method returns a new copy of the `score_d` dictionary with all 
        values zero.

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

    def reset_score_d(self):
        """
        Unlike the method get_zero_score_d(), this method does not create a 
        new `score_d` dictionary. Instead, it sets to zero all values of the 
        existing `self.score_d`.

        Returns
        -------
        None

        """
        for name in self.score_d.keys():
            self.score_d[name] = 0.0

    def reset_managers(self):
        """
        similar to Openie6.metric.Conjunction.reset()
        
        This method sets to zero (resets) the 4 managers.

        Note that reset_managers() and reset_score_d() are separate methods.
        Openie6 lumps them together.

        Returns
        -------
        None

        """
        self.manager_whole.reset()
        self.manager_outer.reset()
        self.manager_inner.reset()
        self.manager_exact.reset()
        # self.n_complete = 0
        # self.n_sentence = 0

    def get_score_d(self, ttt, do_reset=True):
        """
        similar to Openie6.metric.Conjunction.get_metric()
        
        This method returns the current `score_d`. It resets the managers iff
        do_reset=True.

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
        if VERBOSE: print("Entering CCMetric.get_score_d method.")
        score_d = OrderedDict({
            'F1_whole': self.manager_whole.overall_scores.f1_score(),
            'F1_outer': self.manager_outer.overall_scores.f1_score(),
            'F1_inner': self.manager_inner.overall_scores.f1_score(),
            'F1_exact': self.manager_exact.overall_scores.f1_score(),
            'P_exact': self.manager_exact.overall_scores.precision(),
            'R_exact': self.manager_exact.overall_scores.recall()
        })
        self.score_d = copy(score_d)
        if do_reset:
            self.reset_score_d()
        return score_d

    def get_overall_score(self, manager_kind='exact'):
        """
        Similar to Openie6.metric.Conjunction.get_overall_score().
        
        There are 4 kinds of managers produced by this class, and each kind
        has an overall_scores. This method returns the F1 score of the
        overall_scores.

        Parameters
        ----------
        manager_kind: str

        Returns
        -------
        int

        """
        if manager_kind == 'whole':
            manager = self.manager_whole
        elif manager_kind == 'outer':
            manager = self.manager_outer
        elif manager_kind == 'inner':
            manager = self.manager_inner
        elif manager_kind == 'exact':
            manager = self.manager_exact
        else:
            raise ValueError(
                'invalid manager_kind: {}'.format(manager_kind))
        # print("mkcd", manager, self.manager_inner.overall_scores)
        return manager.overall_scores.f1_score()


if __name__ == "__main__":
    def main():
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
