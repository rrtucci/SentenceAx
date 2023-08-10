import torch
import numpy as np
from torch.utils.data import Dataset

class DSet(Dataset):
    def __init__(self, padded_data):
        """
        In Openie6, the `torchteyt.data.Dataset` class is a normal class
        `Dataset(examples, fields)` is abstract class but in newer versions
        it is an abstract class.
        Ref:
        https://machinelearningmastery.com/using-dataset-classes-in-pytorch/
        """
        # abstract super class so don't need to call super().__init__(self)

        # padded_data = {'l_sent_plus_ids': padded_l_sent_plus_ids,
        #                'll_ilabels': padded_ll_ilabels,
        #                'l_word_starts': padded_l_word_starts,
        #                'l_meta_data': l_meta_data}

        self.num_examples = len(padded_data["l_sent_plus_ids"])
        self.num_words =  len(padded_data["l_sent_plus_ids"][0])
        self.depth = len(padded_data["ll_ilabels"])

        x = padded_data["l_sent_plus_ids"]
        x = np.array(x)
        self.x = torch.from_numpy(x)

        y = []
        self.ytype_names = []
        ytype_id = -1
        for name, li in padded_data.items():
            if name != "l_sent_plus_ids":
                ytype_id += 1
                if name != "ll_ilabels":
                    self.ytype_names.append(name)
                    y.append(li)
                else:
                    for k, li in enumerate(padded_data["ll_ilabels"]):
                        self.ytype_names.append(name + "_" + str(k))
                        y.append(li)
        self.num_ytypes = ytype_id + 1
        y = np.array(y)
        np.transpose(y)
        self.y = torch.from_numpy(y)

        def __getitem__(self, example_id):
            return self.x[example_id], self.y[example_id]

        def __len__(self):
            return self.num_examples
