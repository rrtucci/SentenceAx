from Params import *
from AllenTool import *
from transformers import AutoTokenizer
import spacy
import torch
from torch.utils.data import DataLoader

import pickle
import os

# use of tt.data.Field, Field.build_vocab, tt.data.Example are deprecated
# and tt.data.Dataset signature has changed

import torchtext as tt

import nltk
from copy import deepcopy
from SaxDataSet import *
from MInput import *
from Params import *


class SaxDataLoaderTool:
    """
    data processing chain
    (optional allen_fp->)tags_in_fp->MInput->PaddedMInput->SaxDataSet
    ->SaxDataLoaderTool
    
    Classes Example and Field from tt were used in the Openie6 code,
    but they are now deprecated, so they are not used Mappa Mundi. Here is
    link explaining a migration route ot of them.

    https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb#scrollTo=kBV-Wvlo07ye
    
    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    pad_icode: int
    params: Params
    predict_dloader = []
    tags_test_fp: str
    tags_train_fp: str
    tags_tune_fp: str
    test_dloader: DataLoader
    train_dloader: DataLoader
    tune_dloader: DataLoader
    
    """

    def __init__(self,
                 params,
                 auto_tokenizer,
                 tags_train_fp, tags_tune_fp, tags_test_fp):
        """

        Parameters
        ----------
        params: Params
        auto_tokenizer: AutoTokenizer
        tags_train_fp: str
        tags_tune_fp: str
        tags_test_fp: str
        """

        self.params = params
        self.auto_tokenizer = auto_tokenizer
        # full encoding is [101, 0, 102],
        # where 101=BOS_ICODE, 102=EOS_ICODE
        # print("nkjg", type(auto_tokenizer))
        self.pad_icode = \
            auto_tokenizer.encode(auto_tokenizer.pad_token)[1]

        self.tags_train_fp = tags_train_fp
        self.tags_tune_fp = tags_tune_fp
        self.tags_test_fp = tags_test_fp

        self.train_dloader = []
        self.tune_dloader = []
        self.test_dloader = []
        self.predict_dloader = []

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

        # tags_train_fp = self.params.d["tags_train_fp"]
        # tags_tune_fp = self.params.d["tags_tune_fp"]
        # tags_test_fp = self.params.d["tags_test_fp"]

        # if 'predict' not in params.mode, use caching
        assert self.tags_train_fp, self.tags_train_fp
        assert self.tags_tune_fp, self.tags_tune_fp
        assert self.tags_test_fp, self.tags_test_fp
        # model_str = self.params.d["model_str"].replace("/", "_")
        task = self.params.task
        cached_train_m_in_fp = \
            CACHE_DIR + "/" + task + "_train_m_in_" + \
            self.tags_train_fp.replace("/", "_").split(".")[0] + ".pkl"
        cached_tune_m_in_fp = \
            CACHE_DIR + "/" + task + "_tune_m_in_" + \
            self.tags_tune_fp.replace("/", "_").split(".")[0] + ".pkl"
        cached_test_m_in_fp = \
            CACHE_DIR + "/" + task + "_test_m_in_" + \
            self.tags_test_fp.replace("/", "_").split(".")[0] + ".pkl"

        def find_m_in(cached_fp, tags_fp):
            if not os.path.exists(cached_fp) or \
                    self.params.d["refresh_cache"]:
                m_in = MInput(self.params, tags_fp, self.auto_tokenizer)
                pickle.dump(m_in, open(cached_fp, 'wb'))
            else:
                m_in = pickle.load(open(cached_fp, 'rb'))
            return m_in

        train_m_in = find_m_in(cached_train_m_in_fp, self.tags_train_fp)
        tune_m_in = find_m_in(cached_tune_m_in_fp, self.tags_tune_fp)
        test_m_in = find_m_in(cached_test_m_in_fp, self.tags_test_fp)

        # vocab = self.build_vocab(
        #     train_m_in + tune_m_in + test_m_in)

        train_dataset = SaxDataSet(train_m_in)

        tune_dataset = SaxDataSet(tune_m_in)

        test_dataset = SaxDataSet(test_m_in)

        # to simulate bucket sort (along with pad_data)
        # train_dataset.sort()


        return train_dataset, tune_dataset, test_dataset
        # , vocab, orig_sents

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

        predict_m_in = MInput(self.params,
                              predict_in_fp,
                              self.auto_tokenizer)
        # vocab = build_vocab(predict_m_in)

        predict_dataset = SaxDataSet(predict_m_in)

        return predict_dataset

    def set_all_ttt_dataloaders(self):
        """
        # this method calls DataLoader

        Returns
        -------
        None

        """
        train_dataset, tune_dataset, test_dataset = \
            self.get_all_ttt_datasets()

        self.train_dloader = \
            DataLoader(train_dataset,
                       batch_size=self.params.d["batch_size"],
                       # collate_fn=None,
                       shuffle=True,
                       num_workers=1)
        self.tune_dloader = \
            DataLoader(tune_dataset,
                       batch_size=self.params.d["batch_size"],
                       # collate_fn=None,
                       num_workers=1)
        self.test_dloader = \
            DataLoader(test_dataset,
                       batch_size=self.params.d["batch_size"],
                       # collate_fn=None,
                       num_workers=1)

    def set_predict_dataloader(self, predict_in_fp):
        """

        Parameters
        ----------
        predict_in_fp: str

        Returns
        -------
        None

        """
        self.predict_dloader = \
            DataLoader(self.get_predict_dataset(predict_in_fp),
                       batch_size=self.params.d["batch_size"],
                       # collate_fn=None,
                       shuffle=True,
                       num_workers=1)


if __name__ == "__main__":
    def main(params_id):
        params = Params(params_id)
        do_lower_case = ('uncased' in params.d["model_str"])
        auto = AutoTokenizer.from_pretrained(
            params.d["model_str"],
            do_lower_case=do_lower_case,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        tags_train_fp = "tests/extags_train.txt"
        tags_tune_fp = "tests/extags_tune.txt"
        tags_test_fp = "tests/extags_test.txt"

        dl_tool = SaxDataLoaderTool(params,
                                    auto,
                                    tags_train_fp,
                                    tags_tune_fp,
                                    tags_test_fp)
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
