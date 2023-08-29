from sax_globals import *
from AllenTool import *
from transformers import AutoTokenizer
import spacy
import torch
from torch.utils.data import DataLoader

import pickle
import os
# use of
# tt.data.Field, Field.build_vocab
# tt.data.Example
# are deprecated
# and tt.data.Dataset signature has changed
import torchtext as tt

import nltk
from copy import deepcopy
from SaxDataSet import *

class SaxDataLoader:
    """
    Classes Example and Field from tt were used in the Openie6 code,
    but they are now deprecated, so they are not used Mappa Mundi. Here is
    link explaining a migration route ot of them.

    https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb#scrollTo=kBV-Wvlo07ye
    """

    def __init__(self, auto_tokenizer, sent_pad_id,
                 train_fp, dev_fp, test_fp):

        self.params_d = PARAMS_D
        self.auto_tokenizer = auto_tokenizer
        self.sent_pad_id = sent_pad_id
        self.train_fp = train_fp
        self.dev_fp = dev_fp
        self.test_fp = test_fp
        self.spacy_model = None

    @staticmethod
    def remerge_sent(tokens):
        """
        similar to data.remerge_sent()

        Parameters
        ----------
        tokens

        Returns
        -------

        """
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
        """
        similar to data.pos_tags()

        Parameters
        ----------
        tokens

        Returns
        -------

        """
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
        """
        similar to data.verb_tags()

        Parameters
        ----------
        tokens

        Returns
        -------

        """
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

    def get_l_sample_d(self, in_fp):
        """
        similar to data._process_data()


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

        # l_sample_d = []  # list[example_d]
        l_sample_d = []  # list[example_d]
        ilabels_for_each_ex = []  # a list of a list of labels, list[list[in]]
        orig_sents = []

        if type(in_fp) == type([]):
            in_lines = None
        else:
            with(in_fp, "r") as f:
                in_lines = f.readlines()

        prev_line = ""
        for line in in_lines:
            line = line.strip()
            if '[used' in line:  # it's the  beginning of an example
                sentL = line
                encoding = self.auto_tokenizer.batch_encode_plus(
                    sentL.split())
                sentL_ids = [BOS_ILABEL]
                word_starts = []
                for ids in encoding['input_ids']:
                    # special spacy tokens like \x9c have zero length
                    if len(ids) == 0:
                        ids = [100]
                    word_starts.append(len(sentL_ids))
                    sentL_ids += ids  # same as sentL_ids.extend(ids)
                sentL_ids.append(EOS_ILABEL)

                orig_sent = sentL.split('[unused1]')[0].strip()
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
                sample_d = {
                    'sentL_ids': sentL_ids,
                    'l_ilabels': ilabels_for_each_ex[:MAX_EX_DEPTH],
                    'word_starts': word_starts,
                    'orig_sent': orig_sent
                }
                if len(sentL.split()) <= 100:
                    l_sample_d.append(sample_d)
                ilabels_for_each_ex = []
                prev_line = line

            else:
                assert False

        # so far, we haven't assumed any spacy derived data nanalysis
        # if spacy is allowed, the example_d can carry more info.
        if self.spacy_model:
            sents = [sample_d['orig_sent'] for sample_d in l_sample_d]
            for sent_index, spacy_tokens in enumerate(
                    self.spacy_model.pipe(sents, batch_size=10000)):
                spacy_tokens = SaxDataLoader.remerge_sent(spacy_tokens)
                assert len(sents[sent_index].split()) == len(
                    spacy_tokens)
                sample_d = l_sample_d[sent_index]

                pos_mask, pos_indices, pos_words = \
                    SaxDataLoader.pos_mask(spacy_tokens)
                sample_d['pos_mask'] = pos_mask
                sample_d['pos_indices'] = pos_indices

                verb_mask, verb_indices, verb_words = \
                    SaxDataLoader.verb_mask(spacy_tokens)
                sample_d['verb_mask'] = verb_mask
                if len(verb_indices) != 0:
                    sample_d['verb_indices'] = verb_indices
                else:
                    sample_d['verb_indices'] = [0]

        # example_d = {
        #     'sentL_ids': sentL_ids,
        #     'll_label': labels_for_each_ex[:MAX_EX_DEPTH],
        #     'word_starts': word_starts,
        #     'orig_sent': orig_sent,
        #     # if spacy_model:
        #     'pos_mask': pos_mask,
        #     'pos_indices': pos_indices,
        #     'verb_mask': verb_mask,
        #     'verb_indices': verb_indices
        # }

        # use of tt.data.Example is deprecated
        # for example_d in l_sample_d:
        #     example = tt.data.Example.fromdict(example_d, fields)
        #     examples.append(example)
        # return examples, orig_sents

        return l_sample_d, orig_sents

    def get_ttt_datasets(self, pred_in_sents=None):
        """
        similar to data.process_data()

        Parameters
        ----------
        pred_in_sents

        Returns
        -------

        """


        # train_fp = self.params_d["train_fp"]
        # dev_fp = self.params_d["dev_fp"]
        # test_fp = self.params_d["test_fp"]

        model_str = self.params_d["model_str"].replace("/", "_")
        cached_train_fp = f'{self.train_fp}.{model_str}.pkl'
        cached_dev_fp = f'{self.dev_fp}.{model_str}.pkl'
        cached_test_fp = f'{self.test_fp}.{model_str}.pkl'

        orig_sents = []
        if 'predict' in self.params_d["mode"]:
            # no caching used in predict mode
            if not pred_in_sents:  # predict
                # if self.params_d["in_fp"] :
                #     predict_fp = self.params_d["in_fp"]
                # else:
                #     predict_fp = self.params_d["predict_fp"]
                # will set predict_fp = PRED_IN_FP
                with open(PRED_IN_FP, "r") as f:
                    predict_lines = f.readlines()

                pred_in_sents = []
                for line in predict_lines:
                    # Normalize the quotes - similar to that in training data
                    line = line.replace('’', '\'')
                    line = line.replace('”', '\'\'')
                    line = line.replace('“', '\'\'')

                    # tokenized_line = line.split()

                    # Why use both nltk and spacy to word tokenize.
                    # get_ttt_datasets() uses nltk.word_tokenize()
                    # get_samples() uses spacy_model.pipe(sents...)

                    words = ' '.join(nltk.word_tokenize(line))
                    pred_in_sents.append(
                        words + UNUSED_TOKENS_STR)
                    pred_in_sents.append('\n')

            # openie 6 is wrong here. Uses wrong arguments for
            # process_data() which is get_samples() for us.
            # get_samples()
            # returns: examples, orig_sents
            predict_l_sample_d, orig_sents = \
                self.get_l_sample_d(PRED_IN_FP)
            #vocab = build_vocab(predict_l_sample_d)

            predict_dataset = SaxDataSet(predict_l_sample_d,
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
            if not os.path.exists(cached_train_fp) or\
                    self.params_d["build_cache"]:
                train_l_sample_d, _ = self.get_l_sample_d(self.train_fp)
                pickle.dump(train_l_sample_d, open(cached_train_fp, 'wb'))
            else:
                train_l_sample_d = pickle.load(open(cached_train_fp, 'rb'))

            if not os.path.exists(cached_dev_fp) or \
                    self.params_d["build_cache"]:
                dev_l_sample_d, _ = self.get_l_sample_d(self.dev_fp)
                pickle.dump(dev_l_sample_d, open(cached_dev_fp, 'wb'))
            else:
                dev_l_sample_d = pickle.load(open(cached_dev_fp, 'rb'))

            if not os.path.exists(cached_test_fp) or\
                    self.params_d["build_cache"]:
                test_l_sample_d, _ = self.get_l_sample_d(self.test_fp)
                pickle.dump(test_l_sample_d, open(cached_test_fp, 'wb'))
            else:
                test_l_sample_d = pickle.load(open(cached_test_fp, 'rb'))

            # vocab = self.build_vocab(
            #     train_l_sample_d + dev_l_sample_d + test_l_sample_d)

            train_dataset = SaxDataSet(train_l_sample_d,
                                    self.spacy_model,
                                    self.sent_pad_id)
            dev_dataset = SaxDataSet(dev_l_sample_d,
                                  self.spacy_model,
                                  self.sent_pad_id)
            test_dataset = SaxDataSet(test_l_sample_d,
                                   self.spacy_model,
                                   self.sent_pad_id)
            train_dataset.sort()  # to simulate bucket sort (along with pad_data)

        return train_dataset, dev_dataset, test_dataset # , vocab, orig_sents

    def get_ttt_dataloaders(self, type, pred_in_sents=None):
        """

        Parameters
        ----------
        type
        pred_in_sents

        Returns
        -------

        """

        train_dataset, val_dataset, test_dataset = \
            self.get_ttt_datasets(pred_in_sents)
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
