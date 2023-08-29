import torch
import numpy as np
from torch.utils.data import Dataset
from DPadder import *


class SaxDSet(Dataset):
    def __init__(self, l_sample_d, spacy_model, sent_pad_id):
        """
        In Openie6, the `torchtext.data.Dataset` class is a normal class
        `Dataset(examples, fields)` is abstract class but in newer versions
        it is an abstract class.
        Ref:
        https://machinelearningmastery.com/using-dataset-classes-in-pytorch/
        """
        # abstract super class so don't need to call super().__init__(self)

        # padded_data = {'ll_sentL_id': padded_ll_sentL_id,
        #                'lll_label': padded_ll_label,
        #                'll_word_start': padded_ll_word_start,
        #                'l_orig_sent': l_orig_sent}

        padder = SaxDPadder(sent_pad_id, spacy_model)
        padded_data = padder.pad_data(l_sample_d)

        self.num_samples = len(padded_data["ll_sentL_id"])
        self.num_words =  len(padded_data["ll_sentL_id"][0])
        self.depth = len(padded_data["lll_ilabel"])


        x = []
        num_xtypes = -1
        for name, li in padded_data.items():
            if name != "lll_ilabel":
                num_xtypes +=1
                x.append(li)
        self.num_xtypes = num_xtypes
        x = np.array(x)
        self.x = torch.from_numpy(x)

        y = padded_data["lll_ilabel"]
        y = np.array(y)
        self.y = torch.from_numpy(y)

        def __getitem__(self, sample_id):
            return self.x[sample_id], self.y[sample_id]

        def __len__(self):
            return self.num_samples
