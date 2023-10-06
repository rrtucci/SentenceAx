import torch
from words_tags_ilabels_translation import *
from MInput import *


class MOutput:
    """

    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    in_cpu: bool
    ll_confi: list[list[float]] | torch.Tensor
    ll_osent_icode: list[list[int]] | torch.Tensor
    lll_ilabel: list[list[list[int]]] | torch.Tensor
    loss: float | torch.Tensor
    task: str
    """

    def __init__(self,
                 ll_osent_icode,
                 lll_ilabel,
                 ll_confi,
                 loss,
                 task,
                 auto_tokenizer):
        """
        The inputs to the constractor are torch.Tensor and in gpu. They get
        converted to lists and cpu before being stored as attributes of the
        class.

         stands for ten-sor

        Parameters
        ----------
        ll_osent_icode: list[list[int]] | torch.Tensor
        lll_ilabel: list[list[list[int]]] | torch.Tensor
        ll_confi: list[list[float]] | torch.Tensor
        loss: float | torch.Tensor
        task: str
        auto_tokenizer: AutoTokenizer
        
        """
        self.ll_osent_icode = ll_osent_icode
        self.lll_ilabel = lll_ilabel
        self.ll_confi = ll_confi
        self.loss = loss

        self.in_cpu = False
        self.task = task
        self.auto_tokenizer = auto_tokenizer

    def use_tensors(self, reverse=False):
        """
        
        Parameters
        ----------
        reverse: bool

        Returns
        -------
        None

        """
        li = [self.ll_osent_icode,
              self.lll_ilabel,
              self.ll_confi,
              self.loss]
        if not reverse:
            for x in li:
                x = torch.Tensor(x)
        else:
            for x in li:
                x = x.tolist()
            self.loss = float(self.loss[0])

    def move_to_cpu(self):
        """

        Returns
        -------
        None

        """
        assert not self.in_cpu
        self.in_cpu = True
        li = [self.ll_osent_icode,
              self.lll_ilabel,
              self.ll_confi,
              self.loss]
        for x in li:
            x = x.cpu()

        self.ll_confi = (self.ll_confi * 100).round() / 100

    def get_l_orig_sent(self):
        """
        
        Returns
        -------
        list[str]

        """
        l_orig_sent2 = \
            MInput.decode_ll_icode(self.ll_osent_icode_10.cpu(),
                                   self.auto_tokenizer)
        if self.task == "ex":
            return undoL(l_orig_sent2)
        else:
            return l_orig_sent2

    def get_lll_word(self):
        if self.task == "ex":
            translator = translate_ilabels_to_words_via_extags
        elif self.task == "cc":
            translator = translate_ilabels_to_words_via_cctags
        else:
            assert False
        l_orig_sentL = redoL(self.l_orig_sent)
        num_samples = len(self.lll_ilabel)
        num_depths = len(self.lll_ilabel[0])
        # sent_len = len(self.lll_ilabel[0][0])
        lll_word = []
        for sam in range(num_samples):
            ll_word = []
            for depth in range(num_depths):
                ll_word.append(
                    translator(self.lll_ilabel[sam][depth],
                               l_orig_sentL[sam]))
            lll_word.append(ll_word)
        return lll_word
