from CCScoreManager import *
import os
import pickle
from utils_gen import *
from AllenTool import *


class CCMetric:
    """
    similar to Openie6.metric.Conjunction

    This class does scoring for task="cc". Class ExMetric does scoring for
    task="ex".

    This class stores scores of model weights. There are 4 types of scoring
    managers, instances of CCScoreManager: `kind_to_manager[kind]`,
    where kind\in SCORE_KINDS. Each of these managers uses a different
    scoring procedure.

    Setting the parameter `save` to True makes this class store the tree
    structure.

    Attributes
    ----------
    kind_to_manager: dict[str, CCScoreManager]
    save: bool
    score_d: dict[str, float]
    verbose: bool

    """

    def __init__(self, verbose=False):
        """
        Constructor

        Parameters
        ----------
        verbose: bool

        """

        self.kind_to_manager = {}
        for kind in SCORE_KINDS:
            self.kind_to_manager[kind] = CCScoreManager(kind)
        self.save = CC_METRIC_SAVE
        self.score_d = CCMetric.get_zero_score_d()
        self.verbose = verbose

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
        num_samples = len(lll_ilabel)
        if self.verbose:
            print("Entering CCMetric.__call__() method.")
            print("number of samples=", num_samples)
        for k in range(num_samples):
            pred_ccnodes = CCTree(l_osent[k],
                                  lll_pred_ilabel[k],
                                  calc_tree_struc=True).ccnodes
            true_ccnodes = CCTree(l_osent[k],
                                  lll_ilabel[k],
                                  calc_tree_struc=True).ccnodes

            for kind in SCORE_KINDS:
                self.kind_to_manager[kind]. \
                    absorb_new_sample(pred_ccnodes, true_ccnodes)

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

    def get_all_node_ccscore(self, kind='exact'):
        """
        Similar to Openie6.metric.Conjunction.get_overall_score().

        This method returns the all_node_ccscore` for kind `kind`.

        Parameters
        ----------
        kind: str

        Returns
        -------
        CCScore

        """
        return self.kind_to_manager[kind].all_node_ccscore

    @staticmethod
    def get_zero_score_d():
        """
        This method returns a new copy of the `score_d` dictionary with all
        values zero.

        Returns
        -------
        dict[str, float]

        """
        score_d = {}
        for kind in SCORE_KINDS:
            score_d[f"acc_nsam_{kind}"] = (0, 0)
            score_d[f"acc_nsam_{kind}11"] = (0, 0)
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
            self.score_d[name] = (0, 0)

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
        for kind in SCORE_KINDS:
            self.kind_to_manager[kind].reset()

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
        if self.verbose:
            print("Entering CCMetric.get_score_d method.")
        score_d = {}
        for kind in SCORE_KINDS:
            score_d[f"acc_nsam_{kind}"] = \
                self.get_all_node_ccscore(kind).get_acc_nsam()
            score_d[f"acc_nsam_{kind}11"] = \
                self.get_all_node_ccscore(kind).get_acc_nsam11()

        self.score_d = copy(score_d)
        if do_reset:
            self.reset_score_d()
        return score_d


if __name__ == "__main__":
    def main():
        cc_met = CCMetric(verbose=True)
        in_fp = "tests/cc_ilabels.txt"
        with open(in_fp, "r", encoding="utf-8") as f:
            in_lines = get_ascii(f.readlines())

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
        print("acc-nsam score:", cc_met.get_all_node_ccscore(
            "exact").get_acc_nsam())


    main()
