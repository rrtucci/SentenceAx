import torch
import numpy as np
from torch.utils.data import Dataset
from PaddedMInput import *

"""

import torchtext as tt 

Classes tt.data.Example and tt.data.Field are used in the Openie6 code, 
but they are now deprecated, so they are not used in SentenceAx.

Note also that Openie6 uses ttt.data.Dataset, which understands 
ttt.data.Field and ttt.data.Example. SentenceAx uses 
torch.utils.data.Dataset which has a different signature than the now 
deprecated ttt.data.Dataset

Refs:
https://colab.research.google.com/github/pytorch/text/blob/master/examples/legacy_tutorial/migration_tutorial.ipynb#scrollTo=kBV-Wvlo07ye
https://stackoverflow.com/questions/63539809/torchtext-0-7-shows-field-is-being-deprecated-what-is-the-alternative
https://machinelearningmastery.com/using-dataset-classes-in-pytorch/


"""


class SaxDataset(Dataset):
    """
    This class has torch's Dataset class as parent. Basically,
    all SaxDataset does is to override the basic methods of its parent class.
    
    Dataset and DataLoader are located in torch.utils.data. Dataset stores a
    huge number of samples, and DataLoader wraps an iterable around the
    Dataset to enable access to batches of samples in a for loop.


    data processing chain
    (optional allen_fp->)tags_in_fp->MInput->PaddedMInput->SaxDataset
    ->SaxDataLoaderTool
    
    Note from this chain that SaxDataset has an instance of PaddedMInput as
    input and its output goes into an instance of DataLoaderTool.


    Attributes
    ----------
    l_orig_sent: list[str]
        list (usually a batch) of original (i.e., before splitting) sentences
    num_depths: int
        number of extractions (first ex has depth 0, 2nd ex has depth 1, etc.).
        After padding, all samples have the same num_depths. (num_depths
        = 1-based max depth)
    num_samples: int
        number of samples, len(l_orig_sent), usually the size of a batch.
    num_words: int
        number of words in original sentence (osent), after padding!. Same for
        all osent after padding.
    padded_m_in: PaddedMInput
        padded model input
    x: torch.Tensor
        If x are the features of the dataset and y its classification,
        this is the x.
    xname_to_dim1: OrderedDict
        Each batch and feature is described by a tensor of shape (dim0,
        dim1) where dim0 is the batch size, and dim1 is the dim=1 size for
        that feature. This dictionary maps xname (i.e., the name of each
        feature) to its dim1.
    y: torch.Tensor
        If x are the features of the dataset and y its classification,
        this is the y.

    """

    def __init__(self, m_in):
        """
        Constructor

        Parameters
        ----------
        m_in: MInput
            unpadded model input data. This constructor pads the m_in data.
        """
        super().__init__()
        self.padded_m_in = PaddedMInput(m_in)
        self.l_orig_sent = self.padded_m_in.l_orig_sent

        self.num_samples, self.num_depths, self.num_words = \
            self.padded_m_in.lll_ilabel.shape

        x_d = self.padded_m_in.x_d
        xnames = x_d.keys()
        self.x = torch.cat([x_d[xname] for xname in xnames], dim=1)

        y_d = self.padded_m_in.y_d
        self.y = y_d["lll_ilabel"]

        self.xname_to_dim1 = OrderedDict(
            {xname: x_d[xname].shape[1] for xname in xnames})

    @staticmethod
    def invert_cat(x, xname_to_dim1):
        """
        cat=con-cat-enation

        In order to obtain self.x, we concatenate sub-tensors of shapes (
        batch_size, xname_to_dim1[xname]) for all xname in
        xname_to_dim1.keys(). This method does the inverse operation: it
        finds the sub-tensors from the full tensor self.x.

        The method returns a dictionary xname_to_xtensor that maps each
        xname to its sub-tensor.

        Parameters
        ----------
        x: torch.Tensor
        xname_to_dim1: OrderedDict

        Returns
        -------
        OrderedDict

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
        This method allows Model to access the x and y (and other "metadata"
        ) of each sample by a sample index `isample`. All this method does
        is to return self.x[isample], self.y[isample], self.l_orig_sent[
        isample], self.xname_to_dim1.

        self.l_orig_sent[isample] and self.xname_to_dim1 constitute what is
        called metadata. It is split into batches along dim=0, just like
        self.x and self.y are.

        Parameters
        ----------
        isample: int

        Returns
        -------
        torch.Tensor, torch.Tensor

        """
        return self.x[isample], self.y[isample], \
            self.l_orig_sent[isample], self.xname_to_dim1

    def __len__(self):
        """
        This method just returns the number of samples = len(l_orig_sent).
        This is normally the batch size.

        Returns
        -------
        int

        """
        return self.num_samples


if __name__ == "__main__":
    def main():
        params = Params(1)  # pid=1, task="ex", action="train_test"
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
        m_in = MInput(params,
                      in_fp,
                      auto)
        # full encoding is [101, 0, 102], 101=BOS_ICODE, 102=EOS_ICODE
        pad_icode = auto.encode(auto.pad_token)[1]
        print("pad_token, pad_icode=", auto.pad_token, pad_icode)
        dset = SaxDataset(m_in)
        print("xname_to_dim1=", dset.xname_to_dim1)
        print("x.shape, x.shape_product=",
              dset.x.shape, np.product(dset.x.shape))
        print("y.shape, y.shape_product=",
              dset.y.shape, np.product(dset.y.shape))
        print_tensor("y", dset.y)
        xname_to_xtensor = SaxDataset.invert_cat(dset.x, dset.xname_to_dim1)
        for xname in xname_to_xtensor.keys():
            assert xname_to_xtensor[xname].shape == \
                   dset.padded_m_in.x_d[xname].shape


    main()
