from sax_params_d import *
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
from sax_params_d import *


class SaxDataLoader:
    """
    Classes Example and Field from tt were used in the Openie6 code,
    but they are now deprecated, so they are not used Mappa Mundi. Here is
    link explaining a migration route ot of them.

    https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb#scrollTo=kBV-Wvlo07ye
    
    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    params_d: dict[str, Any]
    predict_fp: str
    test_fp: str
    train_fp: str
    tune_fp: str
    use_spacy_model: bool
    
    """

    def __init__(self,
                 auto_tokenizer,
                 train_fp, tune_fp, test_fp,
                 use_spacy_model=True):
        """

        Parameters
        ----------
        auto_tokenizer: AutoTokenizer
        train_fp: str
        tune_fp: str
        test_fp: str
        use_spacy_model: bool
        """

        self.params_d = PARAMS_D()
        self.auto_tokenizer = auto_tokenizer
        # full encoding is [101, 0, 102], 101=BOS_ILABEL, 102=EOS_ILABEL
        self.pad_ilabel = \
            auto_tokenizer.encode(auto_tokenizer.pad_token)[1]

        self.predict_fp = PRED_IN_FP
        self.train_fp = train_fp
        self.tune_fp = tune_fp
        self.test_fp = test_fp
        self.use_spacy_model = use_spacy_model

    def get_m_input(self, in_fp):
        """

        This is used to create an m_imput for in_fp=train_fp, tune_fp, test_fp.

        Parameters
        ----------
        in_fp: str

        Returns
        -------
        MInput

        """
        m_input = MInput(in_fp,
                         TASK,
                         self.auto_tokenizer,
                         self.use_spacy_model)

        return m_input

    def get_ttt_datasets(self, pred_in_sents=None):
        """
        similar to Openie6.data.process_data()

        Parameters
        ----------
        pred_in_sents: list[str]

        Returns
        -------
        SaxDataSet, SaxDataSet, SaxDataSet

        """

        # train_fp = self.params_d["train_fp"]
        # tune_fp = self.params_d["tune_fp"]
        # test_fp = self.params_d["test_fp"]

        model_str = self.params_d["model_str"].replace("/", "_")
        cached_train_fp = f'{self.train_fp}.{model_str}.pkl'
        cached_tune_fp = f'{self.tune_fp}.{model_str}.pkl'
        cached_test_fp = f'{self.test_fp}.{model_str}.pkl'

        orig_sents = []
        if 'predict' in MODE:
            # no caching used if predict in mode
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
                    line = use_ascii_quotes(line)
                    # tokenized_line = line.split()

                    # Why use both nltk and spacy to word tokenize.
                    # get_ttt_datasets() uses nltk.word_tokenize()
                    # get_samples() uses spacy_model.pipe(sents...)

                    words = ' '.join(nltk.word_tokenize(line))
                    pred_in_sents.append(
                        words + UNUSED_TOKENS_STR + "\n")

            # openie 6 is wrong here. Uses wrong arguments for
            # process_data() which is get_samples() for us.
            # get_samples()
            # returns: examples, orig_sents
            predict_m_input = self.get_m_input(self.predict_fp)
            # vocab = build_vocab(predict_m_input)

            predict_dataset = SaxDataSet(predict_m_input,
                                         self.pad_ilabel,
                                         self.use_spacy_model)

            train_dataset = predict_dataset
            tune_dataset = predict_dataset
            test_dataset = predict_dataset
        else:  # if 'predict' not in MODE, use caching
            if not os.path.exists(cached_train_fp) or \
                    self.params_d["build_cache"]:
                train_m_input = self.get_m_input(self.train_fp)
                pickle.dump(train_m_input, open(cached_train_fp, 'wb'))
            else:
                train_m_input = pickle.load(open(cached_train_fp, 'rb'))

            if not os.path.exists(cached_tune_fp) or \
                    self.params_d["build_cache"]:
                tune_m_input = self.get_m_input(self.tune_fp)
                pickle.dump(tune_m_input, open(cached_tune_fp, 'wb'))
            else:
                tune_m_input = pickle.load(open(cached_tune_fp, 'rb'))

            if not os.path.exists(cached_test_fp) or \
                    self.params_d["build_cache"]:
                test_m_input = self.get_m_input(self.test_fp)
                pickle.dump(test_m_input, open(cached_test_fp, 'wb'))
            else:
                test_m_input = pickle.load(open(cached_test_fp, 'rb'))

            # vocab = self.build_vocab(
            #     train_m_input + tune_m_input + test_m_input)

            train_dataset = SaxDataSet(train_m_input,
                                       self.pad_ilabel,
                                       self.use_spacy_model)

            tune_dataset = SaxDataSet(tune_m_input,
                                      self.pad_ilabel,
                                      self.use_spacy_model)

            test_dataset = SaxDataSet(test_m_input,
                                      self.pad_ilabel,
                                      self.use_spacy_model)

            # to simulate bucket sort (along with pad_data)
            # train_dataset.sort()
        return train_dataset, tune_dataset, test_dataset  # , vocab, orig_sents

    def get_ttt_dataloaders(self, ttt_kind, pred_in_sents=None):
        """

        Parameters
        ----------
        ttt_kind: str
        pred_in_sents: list[str]

        Returns
        -------
        DataLoader

        """

        train_dataset, tune_dataset, test_dataset = \
            self.get_ttt_datasets(pred_in_sents)
        # this method calls DataLoader

        if ttt_kind == "train":
            return DataLoader(train_dataset,
                              batch_size=self.params_d["batch_size"],
                              # collate_fn=None,
                              shuffle=True,
                              num_workers=1)
        elif ttt_kind == "tune":
            return DataLoader(tune_dataset,
                              batch_size=self.params_d["batch_size"],
                              # collate_fn=None,
                              num_workers=1)
        elif ttt_kind == "test":
            return DataLoader(test_dataset,
                              batch_size=self.params_d["batch_size"],
                              # collate_fn=None,
                              num_workers=1)
        else:
            assert False


if __name__ == "__main__":

    def main():
        print(PARAMS_D())
        auto = AutoTokenizer.from_pretrained(
            PARAMS_D()["model_str"],
            do_lower_case=True,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        use_spacy_model = True
        train_fp = "testing_files/extags_train.txt"
        tune_fp = "testing_files/extags_tune.txt"
        test_fp = "testing_files/extags_test.txt"

        dloader = SaxDataLoader(auto,
                                train_fp,
                                tune_fp,
                                test_fp,
                                use_spacy_model)
        train_dataset, tune_dataset, test_dataset = \
            dloader.get_ttt_datasets()

    # tried with
    # TASK, MODE = "ex", "predict"
    # TASK, MODE = "ex", "test"
    # in sax_params_d file
    main()
