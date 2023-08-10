import torch
import numpy as np
from torch.utils.data import Dataset

class DSet(Dataset):
    def __init__(self, padded_data):
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
        #                'l_meta_data': l_meta_data}

        self.num_examples = len(padded_data["l_sent_plus_ids"])
        self.depth = len(padded_data["ll_ilabels"])

        self.y = padded_data["l_sent_plus_ids"]

        x = []
        self.feature_names = []
        feature_id = -1
        for name, li in padded_data.items():
            if name != "l_sent_plus_ids":
                feature_id += 1
                if name != "ll_ilabels":
                    self.feature_names.append(name)
                    x.append(li)
                else:
                    for k, li in enumerate(padded_data["ll_ilabels"]):
                        self.feature_names.append(name + "_" + str(k))
                        x.append(li)
        self.num_features = feature_id
        x = np.array(x)
        np.transpose(x)
        self.x = torch.from_numpy(x)

        def __getitem__(self, example_id):
            return self.x[example_id], self.y[example_id]

        def __len__(self):
            return self.num_examples
