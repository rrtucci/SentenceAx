from Params import *
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
from Params import *


class DataLoaderTool:
    """
    Classes Example and Field from tt were used in the Openie6 code,
    but they are now deprecated, so they are not used Mappa Mundi. Here is
    link explaining a migration route ot of them.

    https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb#scrollTo=kBV-Wvlo07ye
    
    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    pad_icode: int
    params: Params
    predict_fp: str
    test_fp: str
    train_fp: str
    tune_fp: str
    use_spacy_model: bool
    
    """

    def __init__(self,
                 params,
                 auto_tokenizer,
                 train_fp, tune_fp, test_fp,
                 use_spacy_model=True):
        """

        Parameters
        ----------
        params: Params
        auto_tokenizer: AutoTokenizer
        train_fp: str
        tune_fp: str
        test_fp: str
        use_spacy_model: bool
        """

        self.params = params
        self.auto_tokenizer = auto_tokenizer
        # full encoding is [101, 0, 102],
        # where 101=BOS_ICODE, 102=EOS_ICODE
        # print("nkjg", type(auto_tokenizer))
        self.pad_icode = \
            auto_tokenizer.encode(auto_tokenizer.pad_token)[1]

        self.train_fp = train_fp
        self.tune_fp = tune_fp
        self.test_fp = test_fp
        self.use_spacy_model = use_spacy_model

    def get_m_in(self, in_fp):
        """

        This is used to create an m_imput for in_fp=train_fp, tune_fp, test_fp.

        Parameters
        ----------
        in_fp: str

        Returns
        -------
        MInput

        """
        m_in = MInput(self.params.task,
                      in_fp,
                      self.auto_tokenizer,
                      self.use_spacy_model)

        return m_in

    def get_dataset_common(self):
        return

    def get_all_ttt_datasets(self):
        """
        similar to Openie6.data.process_data()

        Returns
        -------
        SaxDataSet, SaxDataSet, SaxDataSet

        """
        self.get_dataset_common()

        # train_fp = self.params.d["train_fp"]
        # tune_fp = self.params.d["tune_fp"]
        # test_fp = self.params.d["test_fp"]

        # if 'predict' not in params.mode, use caching
        assert self.train_fp, self.train_fp
        assert self.tune_fp, self.tune_fp
        assert self.test_fp, self.test_fp
        # model_str = self.params.d["model_str"].replace("/", "_")
        task = self.params.task
        cached_train_fp = TTT_CACHE_DIR + "/" + task + "_" + \
                          self.train_fp.split(".")[0] + ".pkl"
        cached_tune_fp = TTT_CACHE_DIR + "/" + task + "_" + \
                         self.tune_fp.split(".")[0] + ".pkl"
        cached_test_fp = TTT_CACHE_DIR + "/" + task + "_" + \
                         self.test_fp.split(".")[0] + ".pkl"

        if not os.path.exists(cached_train_fp) or \
                self.params.d["refresh_cache"]:
            train_m_in = self.get_m_in(self.train_fp)
            pickle.dump(train_m_in, open(cached_train_fp, 'wb'))
        else:
            train_m_in = pickle.load(open(cached_train_fp, 'rb'))

        if not os.path.exists(cached_tune_fp) or \
                self.params.d["refresh_cache"]:
            tune_m_in = self.get_m_in(self.tune_fp)
            pickle.dump(tune_m_in, open(cached_tune_fp, 'wb'))
        else:
            tune_m_in = pickle.load(open(cached_tune_fp, 'rb'))

        if not os.path.exists(cached_test_fp) or \
                self.params.d["refresh_cache"]:
            test_m_in = self.get_m_in(self.test_fp)
            pickle.dump(test_m_in, open(cached_test_fp, 'wb'))
        else:
            test_m_in = pickle.load(open(cached_test_fp, 'rb'))

        # vocab = self.build_vocab(
        #     train_m_in + tune_m_in + test_m_in)

        train_dataset = SaxDataSet(train_m_in,
                                   self.pad_icode,
                                   self.use_spacy_model)

        tune_dataset = SaxDataSet(tune_m_in,
                                  self.pad_icode,
                                  self.use_spacy_model)

        test_dataset = SaxDataSet(test_m_in,
                                  self.pad_icode,
                                  self.use_spacy_model)

        # to simulate bucket sort (along with pad_data)
        # train_dataset.sort()

        return train_dataset, tune_dataset, test_dataset  # , vocab, orig_sents

    def get_predict_dataset(self, predict_in_fp):
        """
        similar to Openie6.data.process_data()

        Parameters
        ----------
        predict_in_fp: str

        Returns
        -------
        SaxDataSet

        """
        self.get_dataset_common()
        # no caching used if predict in mode
        # if not pred_in_sents:  # predict
        #     # if self.params.d["in_fp"] :
        #     #     predict_fp = self.params.d["in_fp"]
        #     # else:
        #     #     predict_fp = self.params.d["predict_fp"]
        #     with open(self.predict_fp, "r") as f:
        #         predict_lines = f.readlines()
        #
        #     pred_in_sents = []
        #     for line in predict_lines:
        #         line = use_ascii_quotes(line)
        #         # tokenized_line = line.split()
        #
        #         # Why use both nltk and spacy to word tokenize.
        #         # get_all_ttt_datasets() uses nltk.word_tokenize()
        #         # get_samples() uses spacy_model.pipe(sents...)
        #
        #         words = ' '.join(nltk.word_tokenize(line))
        #         pred_in_sents.append(
        #             words + UNUSED_TOKENS_STR + "\n")
        #
        # # openie6 is wrong here. Uses wrong arguments for
        # process_data()
        # which is get_all_ttt_datasets() for us.
        # get_samples()
        # returns: examples, orig_sents

        assert predict_in_fp

        predict_m_in = self.get_m_in(predict_in_fp)
        # vocab = build_vocab(predict_m_in)

        predict_dataset = SaxDataSet(predict_m_in,
                                     self.pad_icode,
                                     self.use_spacy_model)

        return predict_dataset

    def get_one_ttt_dataloader(self, ttt):
        """
        # this method calls DataLoader

        Parameters
        ----------
        ttt: str

        Returns
        -------
        DataLoader

        """
        train_dataset, tune_dataset, test_dataset = \
            self.get_all_ttt_datasets()

        if ttt == "train":
            return DataLoader(train_dataset,
                              batch_size=self.params.d["batch_size"],
                              # collate_fn=None,
                              shuffle=True,
                              num_workers=1)
        elif ttt == "tune":
            return DataLoader(tune_dataset,
                              batch_size=self.params.d["batch_size"],
                              # collate_fn=None,
                              num_workers=1)
        elif ttt == "test":
            return DataLoader(test_dataset,
                              batch_size=self.params.d["batch_size"],
                              # collate_fn=None,
                              num_workers=1)
        else:
            assert False

    def get_predict_dataloader(self, predict_in_fp):
        """

        Parameters
        ----------
        predict_in_fp: str

        Returns
        -------
        DataLoader

        """
        return DataLoader(self.get_predict_dataset(predict_in_fp),
                          batch_size=self.params.d["batch_size"],
                          # collate_fn=None,
                          shuffle=True,
                          num_workers=1)


if __name__ == "__main__":
    def main(params_id):
        params = Params(params_id)
        auto = AutoTokenizer.from_pretrained(
            params.d["model_str"],
            do_lower_case=True,
            use_fast=True,
            data_dir=TTT_CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        use_spacy_model = True
        train_fp = "tests/extags_train.txt"
        tune_fp = "tests/extags_tune.txt"
        test_fp = "tests/extags_test.txt"

        dl_tool = DataLoaderTool(params,
                                 auto,
                                 train_fp,
                                 tune_fp,
                                 test_fp,
                                 use_spacy_model)
        train_dataset, tune_dataset, test_dataset = \
            dl_tool.get_all_ttt_datasets()

        print("len(train_dataset)=", len(train_dataset))
        print("len(tune_dataset)=", len(tune_dataset))
        print("len(test_dataset)=", len(test_dataset))


    # try with params_id=2, 3
    # 2: ex", "test"
    # 3: "ex", "predict"
    main(2)
    main(3)
