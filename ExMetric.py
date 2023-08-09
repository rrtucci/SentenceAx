from carb_subset.matcher import Matcher
from carb_subset.carb import Benchmark
import re
from carb_subset.oie_readers.extraction import Extraction


def contains_extraction(extr, list_extr):
    str = ' '.join(extr.args) + ' ' + extr.pred
    for extraction in list_extr:
        if str == ' '.join(extraction.args) + ' ' + extraction.pred:
            return True
    return False


class ExMetric():  # formerly metric.Carb
    def __init__(self, hparams, mapping=None):
        self.dev_benchmark = Benchmark('carb/data/gold/dev.tsv')
        self.test_benchmark = Benchmark('carb/data/gold/test.tsv')
        self.matchingFunc = Matcher.binary_linient_tuple_match
        self.all_predictions, self.all_pos_words, self.all_verb_words = \
            {}, {}, {}
        self.score = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}
        self.hparams = hparams
        self.num_extractions = self.hparams.num_extractions
        self.mapping = None
        self.conj_word_mapping = None

    def __call__(self, predictions, sentences, scores, pos_words=None,
                 verb_words=None):
        num_sentences, extractions, max_sentence_len = predictions.shape
        assert num_sentences == len(sentences)

        for i, sentence_str in enumerate(sentences):
            words = sentence_str.split() + ['[unused1]', '[unused2]',
                                            '[unused3]']
            orig_sentence = sentence_str.split('[unused1]')[0].strip()
            if self.mapping:
                if self.mapping[orig_sentence] not in self.all_predictions:
                    self.all_predictions[self.mapping[orig_sentence]] = []
            else:
                if orig_sentence not in self.all_predictions:
                    self.all_predictions[orig_sentence] = []
            if pos_words != None:
                self.all_pos_words[orig_sentence] = pos_words[i]
            if verb_words != None:
                self.all_verb_words[orig_sentence] = verb_words[i]

            for j in range(extractions):
                extraction = predictions[i][j][:len(words)]
                if sum(extraction) == 0:  # extractions completed
                    break
                pro_extraction = self.process_extraction(
                    extraction, words, scores[i][j].item())
                if pro_extraction.args[0] != '' and pro_extraction.pred != '':
                    if self.mapping:
                        if not contains_extraction(
                                pro_extraction,
                                self._all_predictions[
                                    self.mapping[orig_sentence]]):
                            self._all_predictions[
                                self.mapping[orig_sentence]]. \
                                append(pro_extraction)
                    else:
                        if not contains_extraction(
                                pro_extraction,
                                self._all_predictions[orig_sentence]):
                            self._all_predictions[orig_sentence]. \
                                append(pro_extraction)

        # if self.mapping or self.conj_word_mapping:
        #     for sentence in self.all_predictions:
        #         dextractions = dedup_extractions(
        #             self.all_predictions[sentence], self.conj_word_mapping[sentence])
        #         self.all_predictions[sentence] = dextractions

        return

    def get_metric(self, reset, mode):
        if self.num_extractions:
            for sentence in self.all_predictions:
                self.all_predictions[sentence] = sorted(
                    self.all_predictions[sentence],
                    key=lambda x: x.confidence, reverse=True)[
                                                 :self.num_extractions]

        out_filename = "/dev/null"
        if mode == 'dev':
            auc, optimal_f1_point, last_f1_point = self.dev_benchmark.compare(
                predicted=self.all_predictions,
                matchingFunc=self.matchingFunc,
                output_fn=out_filename, error_file=None,
                binary=False)
        elif mode == 'test':
            auc, optimal_f1_point, last_f1_point = \
                self.test_benchmark.compare(
                    predicted=self.all_predictions,
                    matchingFunc=self.matchingFunc,
                    output_fn=out_filename, error_file=None,
                    binary=False)
        else:
            assert False

        self.score = {
            'carb_auc': auc, 'carb_f1': optimal_f1_point[2],
            'carb_lastf1': last_f1_point[2]}
        score = self.score
        if mode == 'dev' and reset:
            self.reset()
        return score

    def reset(self):
        self.all_predictions = {}
        self.score = {'carb_auc': 0.0, 'carb_f1': 0.0, 'carb_sum': 0.0}

    def process_extraction(self, extraction, sentence, score):
        # rel, arg1, arg2, loc, time = [], [], [], [], []
        rel, arg1, arg2, loc_time, args = [], [], [], [], []
        tag_mode = 'none'
        rel_case = 0
        for i, token in enumerate(sentence):
            if '[unused' in token:
                if extraction[i].item() == 2:
                    rel_case = int(re.search(
                        '\[unused(.*)\]', token).group(1))
                continue
            if extraction[i] == 1:
                arg1.append(token)
            if extraction[i] == 2:
                rel.append(token)
            if extraction[i] == 3:
                arg2.append(token)
            if extraction[i] == 4:
                loc_time.append(token)

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
        if not self.hparams.no_lt:
            arg2 = (arg2 + ' ' + loc_time + ' ' + args).strip()
        sentence_str = ' '.join(sentence).strip()

        extraction = Extraction(
            pred=rel, head_pred_index=None, sent=sentence_str,
            confidence=score, index=0)
        extraction.addArg(arg1)
        extraction.addArg(arg2)

        return extraction

    def process_allenlp_format(self, lines):
        assert self.all_predictions == {}
        for line in lines:
            extr = line.split('\t')
            sentence = extr[0]
            confidence = float(extr[2])

            arg1 = re.findall("<arg1>.*</arg1>",
                              extr[1])[0].strip('<arg1>').strip(
                '</arg1>').strip()
            rel = re.findall("<rel>.*</rel>",
                             extr[1])[0].strip('<rel>').strip('</rel>').strip()
            arg2 = re.findall("<arg2>.*</arg2>",
                              extr[1])[0].strip('<arg2>').strip(
                '</arg2>').strip()

            extraction = Extraction(pred=rel, head_pred_index=None,
                                    sent=sentence, confidence=confidence,
                                    index=0)
            extraction.addArg(arg1)
            extraction.addArg(arg2)

            if sentence not in self.all_predictions:
                self.all_predictions[sentence] = []
            self.all_predictions[sentence] = extraction
