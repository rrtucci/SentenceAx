from my_globals import *
from allen_tool import *
from transformers import AutoTokenizer
import spacy


# Refs:
# https://spacy.io/usage/spacy-101/

class ModelInputWriter:

    def __init__(self, allen_fpath):

        # ttt = train, tune, test
        assert abs(sum(self.ttt_fractions) - 1) < 1e-8
        self.ttt_fractions = (.6, .2, .2)

        self.ztz_to_extractions = read_allen_file(allen_fpath)

        self.light_verbs = [
            "take", "have", "give", "do", "make", "has", "have",
            "be", "is", "were", "are", "was", "had", "being",
            "began", "am", "following", "having", "do",
            "does", "did", "started", "been", "became",
            "left", "help", "helped", "get", "keep",
            "think", "got", "gets", "include", "suggest",
            "used", "see", "consider", "means", "try",
            "start", "included", "lets", "say", "continued",
            "go", "includes", "becomes", "begins", "keeps",
            "begin", "starts", "said", "stop", "begin",
            "start", "continue", "say"]

        tokenizer = AutoTokenizer.from_pretrained(
            model_str="uncased",
            do_lower_case=True,
            use_fast=True,
            data_dir='data',
            add_special_tokens=False,
            additional_special_tokens= UNUSED_TOKENS)

        nlp = spacy.load("en_core_web_sm")

        self.spacy_tokens = ""
        self.simple_to_complex_sents = None  # analogous to conj_mapping
        self.set_simple_to_complex_sents_dict()

    def get_sentences(self):
        return self.ztz_to_extractions.keys()

    def get_num_sents(self):
        return len(self.ztz_to_extractions.keys())

    def get_num_ttt_sents(self):
        num_sents = self.get_num_sents()
        num_train_sents = floor(self.ttt_fractions[0] * num_sents)
        num_tune_sents = floor(self.ttt_fractions[1] * num_sents)
        num_test_sents = floor(self.ttt_fractions[2] * num_sents)
        num_extra_sents = num_sents - num_train_sents - \
                          num_tune_sents - num_test_sents
        num_train_sents += num_extra_sents
        return num_train_sents, num_tune_sents, num_test_sents

    @staticmethod
    def get_tag_to_int(tag_type):
        if tag_type == "extags":
            tag_to_int = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                          'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}
        elif tag_type == "cctags":

            tag_to_int = {'CP_START': 2, 'CP': 1,
                          'CC': 3, 'SEP': 4, 'OTHERS': 5, 'NONE': 0}
        else:
            assert False

        return tag_to_int

    def remerge_sent(self):
        # merges self.spacy_tokens which are not separated by white-space
        # does this recursively until no further changes
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(self.spacy_tokens) - 1:
                tok = self.spacy_tokens[i]
                if not tok.whitespace_:
                    next_tok = self.spacy_tokens[i + 1]
                    # in-place operation.
                    self.spacy_tokens.merge(tok.idx,
                                            next_tok.idx + len(next_tok))
                    changed = True
                i += 1
        return self.spacy_tokens

    def pos_tags(self):
        pos_bool_list, pos_indices, pos_words = [], [], []
        for token_index, token in enumerate(self.spacy_tokens):
            if token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
                pos_bool_list.append(1)
                pos_indices.append(token_index)
                pos_words.append(token.lower_)
            else:
                pos_bool_list.append(0)
        pos_bool_list.append(0)
        pos_bool_list.append(0)
        pos_bool_list.append(0)
        return pos_bool_list, pos_indices, pos_words

    def verb_tags(self):
        verb_bool_list, verb_indices, verb_words = [], [], []
        for token_index, token in enumerate(self.spacy_tokens):
            if token.pos_ in ['VERB'] and token.lower_ not in self.light_verbs:
                verb_bool_list.append(1)
                verb_indices.append(token_index)
                verb_words.append(token.lower_)
            else:
                verb_bool_list.append(0)
        verb_bool_list.append(0)
        verb_bool_list.append(0)
        verb_bool_list.append(0)
        return verb_bool_list, verb_indices, verb_words

    def set_simple_to_complex_sents_dict(self):
        simple_to_complex_sents = {}
        content = open(EXT_SAMPLES_PATH).read()
        complex_ztz = ''
        for sample in content.split('\n\n'):
            for i, line in enumerate(sample.strip('\n').split('\n')):
                if i == 0:
                    complex_ztz = line
                else:
                    simple_to_complex_sents[line] = complex_ztz
        return simple_to_complex_sents

    def write_tags_file(self,
                        tag_type,
                        out_fpath,
                        ztz_id_range):

        num_sents = len(self.ztz_to_extractions.keys())
        assert 0 <= ztz_id_range[0] <= ztz_id_range[1] <= num_sents - 1

        with open(out_fpath, 'w') as f:
            prev_ztz = ''
            top_of_file = True
            ztz_id = -1
            for ztz, ex in self.ztz_to_extractions:
                ztz_id += 1
                if ztz_id < ztz_id_range[0] or ztz_id > ztz_id_range[1]:
                    continue
                if ztz != prev_ztz:
                    new_in_ztz = True
                    prev_ztz = ztz
                    if top_of_file:
                        top_of_file = False
                    else:
                        f.write('\n')
                else:
                    new_in_ztz = False
                if ex.name_is_tagged["ARG2"] and \
                        ex.name_is_tagged["REL"] and \
                        ex.name_is_tagged["ARG1"]:
                    if 'REL' in ex.ztz_tags and 'ARG1' in ex.ztz_tags:
                        if (not ex.arg2) or 'ARG2' in ex.ztz_tags:
                            assert len(ex.in3_tokens) == len(ex.ztz_tags)
                            if new_in_ztz:
                                f.write(' '.join(ex.in3_tokens))
                                f.write('\n')
                            f.write(' '.join(ex.ztz_tags))
                            f.write('\n')

    def write_tags_ttt_files(self, tag_type, out_dir):

        extags_train_fpath = out_dir + "/extags_train.txt"
        # dev=development=validation=tuning
        extags_tune_fpath = out_dir + "/extags_tune.txt"
        extags_test_fpath = out_dir + "/extags_test.txt"

        num_train_sents, num_tune_sents, num_test_sents = \
            self.get_num_ttt_sents()

        for fpath in [extags_train_fpath, extags_tune_fpath,
                      extags_test_fpath]:
            self.write_extags_file(fpath)

        print(
