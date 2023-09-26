import torch
import numpy as np
from torch.utils.data import Dataset
from SaxDataPadder import *


class SaxDataSet(Dataset):
    """

    Attributes
    ----------
    max_depth: int
    num_samples: int
    num_words: int
    num_xtypes: int
    x: torch.tensor
    y: torch.tensor

    """

    def __init__(self, m_input, pad_ilabel, use_spacy_model):
        """
        In Openie6, the `torchtext.data.Dataset` class is a normal class
        `Dataset(examples, fields)` is abstract class but in newer versions
        it is an abstract class.
        Ref:
        https://machinelearningmastery.com/using-dataset-classes-in-pytorch/

        abstract super class so don't need to call super().__init__(self)

        padded_data_d = {'ll_sentL_ilabel': padded_ll_sentL_ilabel,
                       'lll_label': padded_ll_label,
                       'l_word_start_locs': padded_l_word_start_locs,
                       'l_orig_sent': l_orig_sent}

        Parameters
        ----------
        m_input: MInput
        pad_ilabel: int
        use_spacy_model: bool
        """
        padder = SaxDataPadder(m_input, pad_ilabel, use_spacy_model)
        padded_data_d = padder.get_padded_data_d()

        self.num_samples, self.max_depth, self.num_words = \
            padded_data_d["lll_ilabel"].shape

        x = []
        num_xtypes = -1
        for sam in range(self.num_samples):
            sam_x = []
            for name, sub_x in padded_data_d.items():
                if name != "lll_ilabel":
                    num_xtypes += 1
                    sam_x.append(sub_x[sam])
            x.append(sam_x)
        self.num_xtypes = num_xtypes
        self.x = torch.tensor(x)

        y = padded_data_d["lll_ilabel"]
        self.y = torch.tensor(y)

    def __getitem__(self, sample_id):
        """

        Parameters
        ----------
        sample_id: int

        Returns
        -------
        torch.tensor, torch.tensor

        """
        return self.x[sample_id], self.y[sample_id]

    def __len__(self):
        """

        Returns
        -------
        int

        """
        return self.num_samples
