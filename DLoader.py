from sax_globals import *
from allen_tool import *
from transformers import AutoTokenizer
import spacy
import torch
from torch.utils.data import DataLoader

import pickle
import os
# use of
# tt.data.Field,
# tt.data.Example
# are deprecated
# and Dataset signature has changed
import torchtext as tt
import nltk
from copy import deepcopy
from DSet import *

class DLoader:
    """
    Classes Example and Field from tt were used in the Openie6 code,
    but they are now deprecated, so they are not used Mappa Mundi. Here is
    link explaining a migration route ot of them.

    https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb#scrollTo=kBV-Wvlo07ye
    """

    def __init__(self):

        self.params_d = PARAMS_D
        do_lower_case = 'uncased' in self.params_d["model_str"]
        self.auto_tokenizer = AutoTokenizer.from_pretrained(
            self.params_d["model_str"],
            do_lower_case=do_lower_case,
            use_fast=True,
            data_dir='data/pretrained_cache',
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)

        self.spacy_model = None

        self.sent_pad_id = self.auto_tokenizer.convert_tokens_to_ids(
            self.auto_tokenizer.pad_token)

    @staticmethod
    def remerge_sent(tokens):
        # merges spacy tokens which are not separated by white-space
        # does this recursively until no further changes
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(tokens) - 1:
                tok = tokens[i]
                if not tok.whitespace_:
                    next_tok = tokens[i + 1]
                    # in-place operation.
                    tokens.merge(tok.idx,
                                 next_tok.idx + len(next_tok))
                    changed = True
                i += 1
        return tokens

    @staticmethod
    def pos_mask(tokens):
        pos_mask = []
        pos_indices = []
        pos_words = []
        for token_index, token in enumerate(tokens):
            if token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
                pos_mask.append(1)
                pos_indices.append(token_index)
                pos_words.append(token.lower_)
            else:
                pos_mask.append(0)
        pos_mask.append(0)
        pos_mask.append(0)
        pos_mask.append(0)
        return pos_mask, pos_indices, pos_words

    @staticmethod
    def verb_mask(tokens):
        verb_mask, verb_indices, verb_words = [], [], []
        for token_index, token in enumerate(tokens):
            if token.pos_ in ['VERB'] and \
                    token.lower_ not in LIGHT_VERBS:
                verb_mask.append(1)
                verb_indices.append(token_index)
                verb_words.append(token.lower_)
            else:
                verb_mask.append(0)
        verb_mask.append(0)
        verb_mask.append(0)
        verb_mask.append(0)
        return verb_mask, verb_indices, verb_words

    def get_example_ds(self, inp_fp):
        # formerly _process_data()
        """
        this reads a file of the form

        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        the tags may be extags or cctags
        each original sentence and its tag sequences constitute a new example
        """

        # example_ds = []  # list[example_d]
        l_example_d = []  # list[example_d]
        ilabels_for_each_ex = []  # a list of a list of ilabels, list[list[in]]
        orig_sents = []

        if type(inp_fp) == type([]):
            inp_lines = None
        else:
            with(inp_fp, "r") as f:
                inp_lines = f.readlines()

        prev_line = ""
        for line in inp_lines:
            line = line.strip()
            if '[used' in line:  # it's the  beginning of an example
                sent_plus = line
                encoding = self.auto_tokenizer.batch_encode_plus(
                    sent_plus.split())
                sent_plus_ids = [BOS_TOKEN_ID]
                word_starts = []
                for ids in encoding['input_ids']:
                    # special spacy tokens like \x9c have zero length
                    if len(ids) == 0:
                        ids = [100]
                    word_starts.append(len(sent_plus_ids))
                    sent_plus_ids += ids  # same as sent_plus_ids.extend(ids)
                sent_plus_ids.append(EOS_TOKEN_ID)

                orig_sent = sent_plus.split('[unused1]')[0].strip()
                orig_sents.append(orig_sent)

            elif line and '[used' not in line:  # it's a line of tags
                ilabels = [TAG_TO_ILABEL[tag] for tag in line.split()]
                # take away last 3 ids for unused tokens
                ilabels = ilabels[:len(word_starts)]
                ilabels_for_each_ex.append(ilabels)
                prev_line = line
            # last line of file or empty line after example
            # line is either "" or None
            elif len(prev_line) != 0 and not line:
                if len(ilabels_for_each_ex) == 0:
                    ilabels_for_each_ex = [[0]]
                # note that if li=[2,3]
                # then li[:100] = [2,3]
                example_d = {
                    'sent_plus_ids': sent_plus_ids,
                    'l_ilabels': ilabels_for_each_ex[:MAX_EXTRACTION_LENGTH],
                    'word_starts': word_starts,
                    'meta_data': orig_sent
                }
                if len(sent_plus.split()) <= 100:
                    l_example_d.append(example_d)
                ilabels_for_each_ex = []
                prev_line = line

            else:
                assert False

        # so far, we haven't assumed any spacy derived data nanalysis
        # if spacy is allowed, the example_d can carry more info.
        if self.spacy_model:
            sents = [example_d['meta_data'] for example_d in l_example_d]
            for sent_index, spacy_tokens in enumerate(
                    self.spacy_model.pipe(sents, batch_size=10000)):
                spacy_tokens = DLoader.remerge_sent(spacy_tokens)
                assert len(sents[sent_index].split()) == len(
                    spacy_tokens)
                example_d = l_example_d[sent_index]

                pos_mask, pos_indices, pos_words = \
                    DLoader.pos_mask(spacy_tokens)
                example_d['pos_mask'] = pos_mask
                example_d['pos_indices'] = pos_indices

                verb_mask, verb_indices, verb_words = \
                    DLoader.verb_mask(spacy_tokens)
                example_d['verb_mask'] = verb_mask
                if len(verb_indices) != 0:
                    example_d['verb_indices'] = verb_indices
                else:
                    example_d['verb_indices'] = [0]

        # example_d = {
        #     'sent_plus_ids': sent_plus_ids,
        #     'l_ilabels': ilabels_for_each_ex[:MAX_EXTRACTION_LENGTH],
        #     'word_starts': word_starts,
        #     'meta_data': orig_sent,
        #     # if spacy_model:
        #     'pos_mask': pos_mask,
        #     'pos_indices': pos_indices,
        #     'verb_mask': verb_mask,
        #     'verb_indices': verb_indices
        # }

        # use of tt.Example is deprecated
        # for example_d in l_example_d:
        #     example = tt.data.Example.fromdict(example_d, fields)
        #     examples.append(example)
        # return examples, orig_sents

        return l_example_d, orig_sents

    def get_ttt_datasets(self, predict_sentences=None):
        # formerly process_data()

        train_fp = self.params_d["train_fp"]
        dev_fp = self.params_d["dev_fp"]
        test_fp = self.params_d["test_fp"]

        model_str = self.params_d["model_str"].replace("/", "_")
        cached_train_fp = f'{train_fp}.{model_str}.pkl'
        cached_dev_fp = f'{dev_fp}.{model_str}.pkl'
        cached_test_fp = f'{test_fp}.{model_str}.pkl'

        orig_sents = []
        if 'predict' in self.params_d["mode"]:
            # no caching used in predict mode
            if predict_sentences == None:  # predict
                if self.params_d["inp"] != None:
                    predict_fp = self.params_d["inp"]
                else:
                    predict_fp = self.params_d["predict_fp"]
                with open(predict_fp, "r") as f:
                    predict_lines = f.readlines()

                predict_sentences = []
                for line in predict_lines:
                    # Normalize the quotes - similar to that in training data
                    line = line.replace('’', '\'')
                    line = line.replace('”', '\'\'')
                    line = line.replace('“', '\'\'')

                    # tokenized_line = line.split()

                    # Why use both nltk and spacy to word tokenize.
                    # get_ttt_datasets() uses nltk.word_tokenize()
                    # get_examples() uses spacy_model.pipe(sents...)

                    words = ' '.join(nltk.word_tokenize(line))
                    predict_sentences.append(
                        words + UNUSED_TOKENS_STR)
                    predict_sentences.append('\n')

            # openie 6 is wrong here. Uses wrong arguments for
            # process_data() which is get_examples() for us.
            # get_examples()
            # returns: examples, orig_sents
            predict_example_ds, orig_sents = \
                self.get_example_ds(predict_fp)
            META_DATA.build_vocab(
                tt.data.Dataset(predict_example_ds, fields=fields.values()))

            predict_dataset = DSet(predict_example_ds,
                                   self.spacy_model,
                                   self.sent_pad_id)
            train_dataset, dev_dataset, test_dataset = \
                predict_dataset, predict_dataset, predict_dataset
        else: # 'predict' not in self.params_d["mode"]
            self.spacy_model = spacy.load("en_core_web_sm")
            # spacy usage:
            # doc = spacy_model("This is a text")
            # spacy_model.pipe()
            # spacy_model usually abbreviated as nlp
            if not os.path.exists(
                    cached_train_fp) or self.params_d["build_cache"]:
                train_example_ds, _ = self.get_example_ds(train_fp)
                pickle.dump(train_example_ds, open(cached_train_fp, 'wb'))
            else:
                train_example_ds = pickle.load(open(cached_train_fp, 'rb'))

            if not os.path.exists(cached_dev_fp) or \
                    self.params_d["build_cache"]:
                dev_example_ds, _ = self.get_example_ds(dev_fp)
                pickle.dump(dev_example_ds, open(cached_dev_fp, 'wb'))
            else:
                dev_example_ds = pickle.load(open(cached_dev_fp, 'rb'))

            if not os.path.exists(cached_test_fp) or\
                    self.params_d["build_cache"]:
                test_example_ds, _ = self.get_example_ds(test_fp)
                pickle.dump(test_example_ds, open(cached_test_fp, 'wb'))
            else:
                test_example_ds = pickle.load(open(cached_test_fp, 'rb'))

            META_DATA.build_vocab(
                DSet(train_example_ds),
                DSet(dev_example_ds),
                DSet(test_example_ds))

            train_dataset = DSet(train_example_ds,
                                   self.spacy_model,
                                   self.sent_pad_id)
            dev_dataset = DSet(dev_example_ds,
                                   self.spacy_model,
                                   self.sent_pad_id)
            test_dataset = DSet(test_example_ds,
                                   self.spacy_model,
                                   self.sent_pad_id)
            train_dataset.sort()  # to simulate bucket sort (along with pad_data)

        return train_dataset, dev_dataset, test_dataset, \
            META_DATA.vocab, orig_sents

    def get_ttt_dataloaders(self, type, predict_sentences=None):

        train_dataset, val_dataset, test_dataset, \
            meta_data_vocab, orig_sents = self.get_ttt_datasets(
            predict_sentences)
        # this method calls DataLoader

        if type == "train":
            return DataLoader(train_dataset,
                              batch_size=self.params_d["batch_size"],
                              # collate_fn=self.pad_data,
                              shuffle=True,
                              num_workers=1)
        elif type == "val":
            return DataLoader(val_dataset,
                              batch_size=self.params_d["batch_size"],
                              # collate_fn=self.pad_data,
                              num_workers=1)
        elif type == "test":
            return DataLoader(test_dataset,
                              batch_size=self.params_d["batch_size"],
                              # collate_fn=self.pad_data,
                              num_workers=1)
        else:
            assert False
