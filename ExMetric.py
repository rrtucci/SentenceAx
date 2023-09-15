from carb_subset.matcher import Matcher
from carb_subset.carb import Benchmark
import re
from carb_subset.oie_readers.extraction import Extraction
from sax_globals import *


def contains_extraction(ex, l_ex):
    """

    Parameters
    ----------
    ex: Extraction
        This is a carb class, not SaxExtraction
    l_ex: list[Extraction]

    Returns
    -------

    """
    ex_str = ' '.join(ex.args) + ' ' + ex.pred
    for ex0 in l_ex:
        if ex_str == ' '.join(ex0.args) + ' ' + ex0.pred:
            return True
    return False


class ExMetric():
    """
    similar to Openie6.metric.Carb


    """

    def __init__(self, fix_d=None):
        self.dev_benchmark = Benchmark('carb/data/gold/dev.tsv')
        self.test_benchmark = Benchmark('carb/data/gold/test.tsv')
        self.matchingFunc = Matcher.binary_linient_tuple_match
        self.sent_to_extractions = {}
        self.osent_to_pos_mask = {}
        self.osent_to_verb_mask = {}
        self.score_d = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}
        self.fix_d = fix_d

    def __call__(self,
                 sent_to_extractions,
                 l_orig_sent,
                 ll_score,
                 pos_mask=None,
                 verb_mask=None):
        num_samples, num_extractions, max_sentence_len = \
            sent_to_extractions.shape
        assert num_samples == len(l_orig_sent)

        for sam, orig_sent in enumerate(l_orig_sent):
            osentL_words = orig_sent.split() + UNUSED_TOKENS
            if self.fix_d:
                if self.fix_d[orig_sent] not in self.sent_to_extractions:
                    self.sent_to_extractions[self.fix_d[orig_sent]] = []
            else:
                if orig_sent not in self.sent_to_extractions:
                    self.sent_to_extractions[orig_sent] = []
            if pos_mask:
                self.osent_to_pos_mask[orig_sent] = pos_mask[sam]
            if verb_mask:
                self.osent_to_verb_mask[orig_sent] = verb_mask[sam]

            for depth in range(num_extractions):
                l_ex = sent_to_extractions[sam][depth][:len(osentL_words)]
                if sum(l_ex) == 0:  # extractions completed
                    break
                ex0 = self.get_extraction(
                    l_ex, osentL_words, ll_score[sam][depth])
                if ex0.args[0] != '' and ex0.pred != '':
                    if self.fix_d:
                        if not contains_extraction(ex0,
                                                   self.sent_to_extractions[
                                                       self.fix_d[orig_sent]]):
                            self.sent_to_extractions[
                                self.fix_d[orig_sent]].append(ex0)
                    else:
                        if not contains_extraction(
                                ex0,
                                self.sent_to_extractions[orig_sent]):
                            self.sent_to_extractions[orig_sent].append(ex0)

        # if self.fix_d or self.conj_word_mapping:
        #     for sentence in self.sent_to_extractions:
        #         dextractions = dedup_extractions(
        #             self.sent_to_extractions[sentence], self.conj_word_mapping[sentence])
        #         self.sent_to_extractions[sentence] = dextractions

        return

    def reset(self):
        self.sent_to_extractions = {}
        self.score_d = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}

    def get_metric_values(self, reset, mode):
        # similar to Openie6.metric.Carb.get_metric()
        if MAX_EX_DEPTH:
            for sent in self.sent_to_extractions:
                self.sent_to_extractions[sent] = sorted(
                    self.sent_to_extractions[sent],
                    key=lambda x: x.score, reverse=True)[
                                                 :MAX_EX_DEPTH]

        out_fp = "/dev/null"
        if mode == 'dev':
            auc, optimal_f1_point, last_f1_point = \
                self.dev_benchmark.compare(
                    predicted=self.sent_to_extractions,
                    matchingFunc=self.matchingFunc,
                    output_fn=out_fp,
                    error_file=None,
                    binary=False)
        elif mode == 'test':
            auc, optimal_f1_point, last_f1_point = \
                self.test_benchmark.compare(
                    predicted=self.sent_to_extractions,
                    matchingFunc=self.matchingFunc,
                    output_fn=out_fp,
                    error_file=None,
                    binary=False)
        else:
            assert False

        self.score_d = {
            'carb_auc': auc,
            'carb_f1': optimal_f1_point[2],
            'carb_lastf1': last_f1_point[2]}
        score_d = self.score_d
        if mode == 'dev' and reset:
            # this resets score_d
            self.reset()
        return score_d

    def get_extraction(self, l_ilabel, osentL_words, score):
        rel = []
        arg1 = []
        arg2 = []
        loc_time = []
        args = []
        tag_mode = 'none'
        rel_case = 0
        for i, word in enumerate(osentL_words):
            if '[unused' in word:
                if l_ilabel[i].item() == 2:
                    rel_case = int(re.search(
                        '\[unused(.*)\]', word).group(1))
                continue
            if l_ilabel[i] == 1:
                arg1.append(word)
            if l_ilabel[i] == 2:
                rel.append(word)
            if l_ilabel[i] == 3:
                arg2.append(word)
            if l_ilabel[i] == 4:
                loc_time.append(word)

        rel = ' '.join(rel).strip()
        if rel_case == 1:
            rel = 'is ' + rel
        elif rel_case == 2:
            rel = 'is ' + rel + ' of'
        elif rel_case == 3:
            rel = 'is ' + rel + ' from'

        arg1 = ' '.join(arg1).strip()
        arg2 = ' '.join(arg2).strip()
        args = ' '.join(args).strip()
        loc_time = ' '.join(loc_time).strip()
        arg2 = (arg2 + ' ' + loc_time + ' ' + args).strip()
        orig_sentL = ' '.join(osentL_words).strip()

        extraction = Extraction(
            pred=rel,
            head_pred_index=None,
            sent=orig_sentL,
            confidence=score,
            index=0)
        extraction.addArg(arg1)
        extraction.addArg(arg2)

        return extraction
