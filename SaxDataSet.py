import torch
import numpy as np
from torch.utils.data import Dataset
from PaddedMInput import *


class SaxDataSet(Dataset):
    """
    data processing chain
    tags_in_fp->MInput->PaddedMInput->SaxDataSet->DataLoader
                                     ->Model->MOutput

    Attributes
    ----------
    l_orig_sent: list[str]
    num_depths: int
    num_samples: int
    num_words: int
    padded_m_in: PaddedMInput
    x: torch.Tensor
    xname_to_dim1: OrderedDict[str, int]
    y: torch.Tensor

    """

    def __init__(self, m_in):
        """
        In Openie6, the `torchtext.data.Dataset` class is a normal class
        `Dataset(examples, fields)` is abstract class but in newer versions
        it is an abstract class.
        Ref:
        https://machinelearningmastery.com/using-dataset-classes-in-pytorch/


        Parameters
        ----------
        m_in: MInput
        """
        super().__init__()
        self.padded_m_in = PaddedMInput(m_in)
        self.l_orig_sent = self.padded_m_in.l_orig_sent
        self.xname_to_dim1 = self.padded_m_in.xname_to_dim1

        self.num_samples, self.num_depths, self.num_words = \
            self.padded_m_in.lll_ilabel.shape

        x_d = self.padded_m_in.x_d
        xnames = x_d.keys()
        self.x = torch.cat([x_d[xname] for xname in xnames], dim=1)

        y_d = self.padded_m_in.y_d
        self.y = y_d["lll_ilabel"]

    @staticmethod
    def invert_cat(x, xname_to_dim1):
        """

        Parameters
        ----------
        x: torch.Tensor
        xname_to_dim1: OrderedDict[str, int]

        Returns
        -------
        OrderedDict[str, torch.Tensor]

        """
        dim1s = xname_to_dim1.values()
        endings = [0]
        dim1_sum = 0
        for dim1 in dim1s:
            dim1_sum += dim1
            endings.append(dim1_sum)
        xnames = xname_to_dim1.keys()
        xname_to_xtensor = OrderedDict()
        for i, xname in enumerate(xnames):
            xname_to_xtensor[xname] = x[:, endings[i]: endings[i + 1]]
        return xname_to_xtensor

    def __getitem__(self, isample):
        """

        Parameters
        ----------
        isample: int

        Returns
        -------
        torch.Tensor, torch.Tensor

        self.l_orig_sent[isample]   is what is called metadata.
        It is broken into batches along dim=0, just like self.x and self.y are
        """
        return self.x[isample], self.y[isample], self.l_orig_sent[isample]

    def __len__(self):
        """

        Returns
        -------
        int

        """
        return self.num_samples


if __name__ == "__main__":
    def main():
        task = "ex"
        in_fp = "tests/extags_test.txt"
        model_str = "bert-base-uncased"
        do_lower_case = ('uncased' in model_str)
        auto = AutoTokenizer.from_pretrained(
            model_str,
            do_lower_case=do_lower_case,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        use_spacy_model = True
        m_in = MInput(task,
                      in_fp,
                      auto,
                      use_spacy_model)
        # full encoding is [101, 0, 102], 101=BOS_ICODE, 102=EOS_ICODE
        pad_icode = auto.encode(auto.pad_token)[1]
        print("pad_token, pad_icode=", auto.pad_token, pad_icode)
        dset = SaxDataSet(m_in)
        print("xname_to_dim1=", dset.xname_to_dim1)
        print("x.shape, x.shape_product=",
              dset.x.shape, np.product(dset.x.shape))
        print("y.shape, y.shape_product=",
              dset.y.shape, np.product(dset.y.shape))
        xname_to_xtensor = SaxDataSet.invert_cat(dset.x, dset.xname_to_dim1)
        for xname in xname_to_xtensor.keys():
            assert xname_to_xtensor[xname].shape == \
                   dset.padded_m_in.x_d[xname].shape


    main()
