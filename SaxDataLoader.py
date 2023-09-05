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
from MInput import *

class SaxDataLoader:
    """
    Classes Example and Field from tt were used in the Openie6 code,
    but they are now deprecated, so they are not used Mappa Mundi. Here is
    link explaining a migration route ot of them.

    https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb#scrollTo=kBV-Wvlo07ye
    """

    def __init__(self, auto_tokenizer, pad_icode,
                 train_fp, val_fp, test_fp, use_spacy_model=True):

        self.params_d = PARAMS_D
        self.auto_tokenizer = auto_tokenizer
        self.pad_icode = pad_icode
        self.train_fp = train_fp
        self.val_fp = val_fp
        self.test_fp = test_fp
        self.use_spacy_model = use_spacy_model

        self.train_m_input = None
        self.val_m_input = None
        self.test_m_input = None
        self.predict_m_input = None

    def get_m_input(self, in_fp):
        m_input = MInput(TASK, self.auto_tokenizer, self.use_spacy_model)
        m_input.read_input_extags_file(in_fp)

        return m_input

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
        # val_fp = self.params_d["val_fp"]
        # test_fp = self.params_d["test_fp"]

        model_str = self.params_d["model_str"].replace("/", "_")
        cached_train_fp = f'{self.train_fp}.{model_str}.pkl'
        cached_val_fp = f'{self.val_fp}.{model_str}.pkl'
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
            predict_m_input = self.get_m_input(PRED_IN_FP)
            #vocab = build_vocab(predict_m_input)

            predict_dataset = SaxDataSet(predict_m_input,
                                      self.pad_icode,
                                      self.use_spacy_model)
            train_dataset, val_dataset, test_dataset = \
                predict_dataset, predict_dataset, predict_dataset
        else: # 'predict' not in self.params_d["mode"]
            if not os.path.exists(cached_train_fp) or\
                    self.params_d["build_cache"]:
                train_m_input = self.get_m_input(self.train_fp)
                pickle.dump(train_m_input, open(cached_train_fp, 'wb'))
            else:
                train_m_input = pickle.load(open(cached_train_fp, 'rb'))

            if not os.path.exists(cached_val_fp) or \
                    self.params_d["build_cache"]:
                val_m_input = self.get_m_input(self.val_fp)
                pickle.dump(val_m_input, open(cached_val_fp, 'wb'))
            else:
                val_m_input = pickle.load(open(cached_val_fp, 'rb'))

            if not os.path.exists(cached_test_fp) or\
                    self.params_d["build_cache"]:
                test_m_input = self.get_m_input(self.test_fp)
                pickle.dump(test_m_input, open(cached_test_fp, 'wb'))
            else:
                test_m_input = pickle.load(open(cached_test_fp, 'rb'))

            # vocab = self.build_vocab(
            #     train_m_input + val_m_input + test_m_input)

            train_dataset = SaxDataSet(train_m_input,
                                       self.pad_icode,
                                       self.use_spacy_model)
            val_dataset = SaxDataSet(val_m_input,
                                     self.pad_icode,
                                     self.use_spacy_model)
            test_dataset = SaxDataSet(test_m_input,
                                      self.pad_icode,
                                      self.use_spacy_model)
            train_dataset.sort()  # to simulate bucket sort (along with pad_data)

        return train_dataset, val_dataset, test_dataset # , vocab, orig_sents

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
