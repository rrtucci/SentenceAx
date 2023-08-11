import torch
import numpy as np
from torch.utils.data import Dataset
from DPadder import *


class DSet(Dataset):
    def __init__(self, example_ds, spacy_model, sent_pad_id):
        """
        In Openie6, the `torchtext.data.Dataset` class is a normal class
        `Dataset(examples, fields)` is abstract class but in newer versions
        it is an abstract class.
        Ref:
        https://machinelearningmastery.com/using-dataset-classes-in-pytorch/
        """
        # abstract super class so don't need to call super().__init__(self)

        # padded_data = {'l_sent_plus_ids': padded_l_sent_plus_ids,
        #                'll_ilabels': padded_ll_ilabels,
        #                'l_word_starts': padded_l_word_starts,
        #                'l_orig_sent': l_orig_sent}

        padder = DPadder(sent_pad_id, spacy_model)
        padded_data = padder.pad_data(example_ds)

        self.num_examples = len(padded_data["l_sent_plus_ids"])
        self.num_words =  len(padded_data["l_sent_plus_ids"][0])
        self.depth = len(padded_data["ll_ilabels"])


        x = []
        num_xtypes = -1
        for name, li in padded_data.items():
            if name != "ll_ilabels":
                num_xtypes +=1
                x.append(li)
        self.num_xtypes = num_xtypes
        x = np.array(x)
        self.x = torch.from_numpy(x)

        y = padded_data["ll_ilabels"]
        y = np.array(y)
        self.y = torch.from_numpy(y)

        def __getitem__(self, example_id):
            return self.x[example_id], self.y[example_id]

        def __len__(self):
            return self.num_examples
