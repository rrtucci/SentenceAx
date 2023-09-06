import torch
import numpy as np
from torch.utils.data import Dataset
from SaxDataPadder import *


class SaxDataSet(Dataset):
    def __init__(self, m_input, pad_icode, use_spacy_model):
        """
        In Openie6, the `torchtext.data.Dataset` class is a normal class
        `Dataset(examples, fields)` is abstract class but in newer versions
        it is an abstract class.
        Ref:
        https://machinelearningmastery.com/using-dataset-classes-in-pytorch/
        """
        # abstract super class so don't need to call super().__init__(self)

        # padded_data_d = {'ll_sentL_icode': padded_ll_sentL_icode,
        #                'lll_label': padded_ll_label,
        #                'l_word_start_locs': padded_l_word_start_locs,
        #                'l_orig_sent': l_orig_sent}

        padder = SaxDataPadder(m_input, pad_icode, use_spacy_model)
        padded_data_d = padder.get_padded_data_d()

        self.num_samples, self.max_depth, self.num_words=\
            padded_data_d["lll_icode"].shape


        x = []
        num_xtypes = -1
        for sam in range(self.num_samples):
            sam_x = []
            for name, sub_x in padded_data_d.items():
                if name != "lll_icode":
                    num_xtypes +=1
                    sam_x.append(sub_x[sam])
            x.append(sam_x)
        self.num_xtypes = num_xtypes
        self.x = torch.tensor(x)

        y = padded_data_d["lll_icode"]
        self.y = torch.tensor(y)

        def __getitem__(self, sample_id):
            return self.x[sample_id], self.y[sample_id]

        def __len__(self):
            return self.num_samples
