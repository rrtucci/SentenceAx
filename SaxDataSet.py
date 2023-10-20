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
    num_depths: int
    num_samples: int
    num_words: int
    padder: SaxDataPadder
    x: torch.Tensor
    xtypes: list[str]
    y: torch.Tensor

    """

    def __init__(self, m_in, pad_icode, use_spacy_model):
        """
        In Openie6, the `torchtext.data.Dataset` class is a normal class
        `Dataset(examples, fields)` is abstract class but in newer versions
        it is an abstract class.
        Ref:
        https://machinelearningmastery.com/using-dataset-classes-in-pytorch/

        abstract super class so don't need to call super().__init__()
        padded_data_d = {'ll_sentL_ilabel': padded_ll_sentL_ilabel,
                       'lll_label': padded_ll_label,
                       'l_wstart_locs': padded_l_wstart_locs,
                       'l_orig_sent': l_orig_sent}



        Parameters
        ----------
        m_in: MInput
        pad_icode: int
        use_spacy_model: bool
        """
        self.padded_m_in = PaddedMInput(m_in)

        self.num_samples, self.num_depths, self.num_words = \
            self.padded_m_in.lll_ilabel.shape

        self.xtypes = [name for name in data_d.keys() if
                       name != "lll_ilabel"]
        self.x = torch.cat([data_d[xtype] for xtype in self.xtypes], dim=1)

        self.y = data_d["lll_ilabel"]

    def __getitem__(self, sample_id):
        """

        Parameters
        ----------
        sample_id: int

        Returns
        -------
        torch.Tensor, torch.Tensor

        """
        return self.x[sample_id], self.y[sample_id]

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
        dset = SaxDataSet(m_in, pad_icode, use_spacy_model)
        dset.padder.print_padded_data_d_shapes()
        print("xtypes=", dset.xtypes)
        print("x.shape, x.shape_product=",
              dset.x.shape, np.product(dset.x.shape))
        print("y.shape, y.shape_product=",
              dset.y.shape, np.product(dset.y.shape))


    main()
