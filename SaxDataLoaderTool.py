from Params import *
from AllenTool import *
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

import pickle
import os

import nltk
from copy import deepcopy
from SaxDataset import *
from MInput import *
from Params import *


class SaxDataLoaderTool:
    """
    This main purpose of this class is to create torch DataLoaders for ttt
    in ["train", "tune", "test"]. and for predicting.

    Dataset and DataLoader are located in torch.utils.data. Dataset stores a
    huge number of samples, and DataLoader wraps an iterable around the
    Dataset to enable access to batches of samples in a for loop.

    SaxDataset is a child of torch Dataset. SaxDataLoaderTool is not a child
    of DataLoader; instead, it creates multiple instances of DataLoader.
    That is why we call it SaxDataLoaderTool rather than just SaxDataLoader.

    data processing chain
    (optional allen_fp->)tags_in_fp->MInput->PaddedMInput->SaxDataset
    ->SaxDataLoaderTool

    Note from this chain that SaxDataLoaderTool has an instance of
    SaxDataset as input for each DataLoader instance it creates.

    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    pad_icode: int
        integer code for padding. This equals 0 for BERT
    params: Params
        parameters
    predict_dloader: DataLoader|None
        DataLoader for predicting.
    tags_test_fp: str
        file path for extags or cctags file used when ttt="test"
    tags_train_fp: str
        file path for extags or cctags file used when ttt="train"
    tags_tune_fp: str
        file path for extags or cctags file used when ttt="tune". (
        tune=validation)
    test_dloader: DataLoader|None
        DataLoader for ttt="test"
    train_dloader: DataLoader|None
        DataLoader for ttt="train"
    tune_dloader: DataLoader|None
        DataLoader for ttt="tune". (tune=validation)
    
    """

    def __init__(self,
                 params,
                 auto_tokenizer,
                 tags_train_fp, tags_tune_fp, tags_test_fp):
        """
        Constructor

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
        # print("nkjg", type(auto_tokenizer))
        self.pad_icode = \
            auto_tokenizer.encode(auto_tokenizer.pad_token)[1]

        self.tags_train_fp = tags_train_fp
        self.tags_tune_fp = tags_tune_fp
        self.tags_test_fp = tags_test_fp

        self.train_dloader = None
        self.tune_dloader = None
        self.test_dloader = None
        self.predict_dloader = None

    def get_all_ttt_datasets(self):
        """
        similar to Openie6.data.process_data()

        This method returns a triple of 3 SaxDatasets, one each for ttt in [
        "train", "tune", "test"].

        Take ttt="train" as an example. If self.params.d["refresh_cache"] =
        True or there is a file with the appropriate info previously stored
        in the `cache` folder, this method constructs the train dataset from
        that. Otherwise, this method reads the self.tags_train_fp file and
        constructs the dataset from that, and stores the results, for future
        use, as a pickle file in the `cache` folder.

        Returns
        -------
        SaxDataset, SaxDataset, SaxDataset

        """

        # tags_train_fp = self.params.d["tags_train_fp"]
        # tags_tune_fp = self.params.d["tags_tune_fp"]
        # tags_test_fp = self.params.d["tags_test_fp"]

        # if 'predict' not in params.action, use caching
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

        def find_m_in(cached_fp, tags_fp, ttt):
            if not os.path.exists(cached_fp) or \
                    self.params.d["refresh_cache"]:
                m_in = MInput(
                    self.params,
                    tags_fp,
                    self.auto_tokenizer,
                    omit_exless=get_omit_exless_flag(task, ttt))
                pickle.dump(m_in, open(cached_fp, 'wb'))
            else:
                m_in = pickle.load(open(cached_fp, 'rb'))
            return m_in

        train_m_in = find_m_in(cached_train_m_in_fp,
                               self.tags_train_fp,
                               "train")
        tune_m_in = find_m_in(cached_tune_m_in_fp,
                              self.tags_tune_fp,
                              "tune")
        test_m_in = find_m_in(cached_test_m_in_fp,
                              self.tags_test_fp,
                              "test")

        # vocab = self.build_vocab(
        #     train_m_in + tune_m_in + test_m_in)

        train_dataset = SaxDataset(train_m_in)

        tune_dataset = SaxDataset(tune_m_in)

        test_dataset = SaxDataset(test_m_in)

        # to simulate bucket sort (along with pad_data)
        # train_dataset.sort()

        return train_dataset, tune_dataset, test_dataset
        # , vocab, orig_sents

    def get_predict_dataset(self, predict_in_fp):
        """
        similar to Openie6.data.process_data()

        This method returns a dataset for predicting. It creates that
        dataset from the info it gleans by reading the file `predict_in_fp`.

        Parameters
        ----------
        predict_in_fp: str

        Returns
        -------
        SaxDataset

        """
        # no caching used if predict in action
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
        #         line = convert_to_ascii(line)
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
                              self.auto_tokenizer,
                              omit_exless=False)
        # vocab = build_vocab(predict_m_in)

        predict_dataset = SaxDataset(predict_m_in)

        return predict_dataset

    def set_all_ttt_dataloaders(self):
        """
        This method sets class attributes for 3 DataLoaders, one for each
        ttt in [ "train", "tune", "test"].

        The method does this by first calling get_all_ttt_dataset() to get 3
        Datasets. It then constructs the 3 DataLoaders from those 3 Datasets.

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
        This method sets the class attribute for the DataLoader for predicting.

        The method does this by first calling get_predict_dataset() to get a
        predict Dataset. It then constructs the DataLoader from that Dataset.


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
    def main(pid):
        params = Params(pid)
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
