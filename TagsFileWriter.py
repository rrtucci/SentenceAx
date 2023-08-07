from sax_globals import *
from allen_tool import *
from transformers import AutoTokenizer
import spacy
import nltk
from math import floor


# Refs:
# https://spacy.io/usage/spacy-101/

class TagsFileWriter:

    def __init__(self, allen_fp):


        # ttt = train, tune, test
        assert abs(sum(self.ttt_fractions) - 1) < 1e-8
        self.ttt_fractions = (.6, .2, .2)

        self.ztz_to_extractions = read_allen_file(allen_fp)


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

    def get_extags(hparams, model, sentences,
                   orig_sentences, sentence_indices_list):
        label_dict = {0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2', 4: 'ARG2',
                      5: 'NONE'}
        lines = []
        outputs = model.outputs
        idx1, idx2, idx3 = 0, 0, 0
        count = 0
        prev_orig_sentence = ''

        for i in range(0, len(sentence_indices_list)):
            if len(sentence_indices_list[i]) == 0:
                sentence = orig_sentences[i].split('[unused1]')[
                    0].strip().split()
                sentence_indices_list[i].append(list(range(len(sentence))))

            lines.append(
                '\n' + orig_sentences[i].split('[unused1]')[0].strip())
            for j in range(0, len(sentence_indices_list[i])):
                assert len(sentence_indices_list[i][j]) == len(
                    outputs[idx1]['meta_data'][
                        idx2].strip().split()), ipdb.set_trace()
                sentence = outputs[idx1]['meta_data'][
                               idx2].strip() + ' [unused1] [unused2] [unused3]'
                assert sentence == sentences[idx3]
                orig_sentence = orig_sentences[i]
                predictions = outputs[idx1]['predictions'][idx2]

                all_extractions, all_str_labels, len_exts = [], [], []
                for prediction in predictions:
                    if prediction.sum().item() == 0:
                        break

                    labels = [0] * len(orig_sentence.strip().split())
                    prediction = prediction[:len(sentence.split())].tolist()
                    for idx, value in enumerate(
                            sorted(sentence_indices_list[i][j])):
                        labels[value] = prediction[idx]

                    labels = labels[:-3]
                    if 1 not in prediction and 2 not in prediction:
                        continue

                    str_labels = ' '.join([label_dict[x] for x in labels])
                    lines.append(str_labels)

                idx3 += 1
                idx2 += 1
                if idx2 == len(outputs[idx1]['meta_data']):
                    idx2 = 0
                    idx1 += 1

        lines.append('\n')
        return lines

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

    def write_extags_file(self,
                          tag_type,
                          out_fp,
                          ztz_id_range):

        num_sents = len(self.ztz_to_extractions.keys())
        assert 0 <= ztz_id_range[0] <= ztz_id_range[1] <= num_sents - 1

        with open(out_fp, 'w') as f:
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

        extags_train_fp = out_dir + "/extags_train.txt"
        # dev=development=validation=tuning
        extags_tune_fp = out_dir + "/extags_tune.txt"
        extags_test_fp = out_dir + "/extags_test.txt"

        num_train_sents, num_tune_sents, num_test_sents = \
            self.get_num_ttt_sents()

        for fpath in [extags_train_fp, extags_tune_fp,
                      extags_test_fp]:
            self.write_extags_file(fpath)


    def get_extags(self, model, sentences,
                   orig_sentences, sentence_indices_list):
        label_dict = {0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2', 4: 'ARG2',
                      5: 'NONE'}
        lines = []
        outputs = model.outputs
        idx1, idx2, idx3 = 0, 0, 0
        count = 0
        prev_orig_sentence = ''

        for i in range(0, len(sentence_indices_list)):
            if len(sentence_indices_list[i]) == 0:
                sentence = orig_sentences[i].split('[unused1]')[
                    0].strip().split()
                sentence_indices_list[i].append(list(range(len(sentence))))

            lines.append(
                '\n' + orig_sentences[i].split('[unused1]')[0].strip())
            for j in range(0, len(sentence_indices_list[i])):
                assert len(sentence_indices_list[i][j]) == len(
                    outputs[idx1]['meta_data'][
                        idx2].strip().split())
                sentence = outputs[idx1]['meta_data'][
                               idx2].strip() + ' [unused1] [unused2] [unused3]'
                assert sentence == sentences[idx3]
                orig_sentence = orig_sentences[i]
                predictions = outputs[idx1]['predictions'][idx2]

                all_extractions, all_str_labels, len_exts = [], [], []
                for prediction in predictions:
                    if prediction.sum().item() == 0:
                        break

                    labels = [0] * len(orig_sentence.strip().split())
                    prediction = prediction[:len(sentence.split())].tolist()
                    for idx, value in enumerate(
                            sorted(sentence_indices_list[i][j])):
                        labels[value] = prediction[idx]

                    labels = labels[:-3]
                    if 1 not in prediction and 2 not in prediction:
                        continue

                    str_labels = ' '.join([label_dict[x] for x in labels])
                    lines.append(str_labels)

                idx3 += 1
                idx2 += 1
                if idx2 == len(outputs[idx1]['meta_data']):
                    idx2 = 0
                    idx1 += 1

        lines.append('\n')
        return lines

