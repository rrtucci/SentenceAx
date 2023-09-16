from carb_subset.matcher import Matcher
from carb_subset.carb import Benchmark
import re
from sax_globals import *
from sax_utils import *
from SaxExtraction import *
from sax_extraction_utils import get_extraction
from words_tags_ilabels_translation import *
from MOutput import *


class ExMetric():
    """
    similar to Openie6.metric.Carb


    """

    def __init__(self, fix_d=None):
        self.dev_benchmark = Benchmark('carb-data/gold/dev.tsv')
        self.test_benchmark = Benchmark('carb-data/gold/test.tsv')
        self.matchingFunc = Matcher.binary_linient_tuple_match
        self.osent_to_exs = {}
        # self.l_osent_pos_mask = [] # not used
        # self.l_osent_verb_mask = [] # not used
        self.score_d = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}
        self.fix_d = fix_d

    def __call__(self,
                 osent_to_exs,
                 l_orig_sent,
                 ll_score):

        for sam_id, orig_sent in enumerate(l_orig_sent):
            osentL_words = get_words(orig_sent) + UNUSED_TOKENS
            if self.fix_d:
                if self.fix_d[orig_sent] not in self.osent_to_exs:
                    self.osent_to_exs[self.fix_d[orig_sent]] = []
            else:
                if orig_sent not in self.osent_to_exs:
                    self.osent_to_exs[orig_sent] = []

            num_exs = len(osent_to_exs[orig_sent])
            for depth in range(num_exs):
                ex = osent_to_exs[orig_sent][depth]
                extags = translate_words_to_extags(ex)
                ilabels = translate_extags_to_ilabels(extags)
                # all ilabels=0 once no more extractions
                if sum(ilabels) == 0:
                    break
                ex0 = get_extraction(
                    ilabels,
                    orig_sent + UNUSED_TOKENS_STR,
                    ll_score[sam_id][depth])
                if ex0.arg1 and ex0.rel:
                    if self.fix_d:
                        if ex0.is_not_in(self.osent_to_exs[
                                             self.fix_d[orig_sent]]):
                            self.osent_to_exs[
                                self.fix_d[orig_sent]].append(ex0)
                    else:
                        if ex0.is_not_in(self.osent_to_exs[
                                             self.osent_to_exs[orig_sent]]):
                            self.osent_to_exs[orig_sent].append(ex0)

    def reset(self):
        self.osent_to_exs = {}
        self.score_d = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}

    def get_metric_values(self, mode, do_reset=True):
        # similar to Openie6.metric.Carb.get_metric()
        if MAX_EX_DEPTH:
            for sent in self.osent_to_exs:
                self.osent_to_exs[sent] = \
                    sorted(self.osent_to_exs[sent],
                           key=lambda x: x.score,
                           reverse=True)[:MAX_EX_DEPTH]
        openie6_osent_to_exs = {}
        for osent, sax_exs in self.osent_to_exs.items():
            openie6_osent_to_exs[osent] = [sax_ex.convert_to_carb_ex
                                           for sax_ex in sax_exs]

        out_fp = "/dev/null"
        if mode == 'dev':
            bmark = self.dev_benchmark
        elif mode == 'test':
            bmark = self.test_benchmark
        else:
            assert False
        auc, optimal_f1_point, last_f1_point = \
            bmark.compare(
                predicted=openie6_osent_to_exs,
                matchingFunc=self.matchingFunc,
                output_fn=out_fp,
                error_file=None,
                binary=False)

        self.score_d = {
            'carb_auc': auc,
            'carb_f1': optimal_f1_point[2],
            'carb_lastf1': last_f1_point[2]}
        score_d = self.score_d
        if mode == 'dev' and do_reset:
            # this resets score_d
            self.reset()
        return score_d

if __name__ == "__main__":

    def main():
        em = ExMetric()
        m_out = MOutput(task="ex")

        em(m_out.lll_ilabel, m_out.l_orig_sent, m_out.ll_score)
        score_d = em.get_metric_values(mode="dev", do_reset=True)


