import torch
from words_tags_ilabels_translation import *
from MInput import *


class MOutput:
    """

    Attributes
    ----------
    in_cpu: bool
    l_orig_sent: list[str]
    ll_confi: list[list[float]]
    ll_confi_10: torch.Tensor
    ll_osent_icode_10: torch.Tensor
    lll_ilabel: list[list[list[int]]]
    lll_ilabel_10: torch.Tensor
    loss: float
    loss_10: torch.Tensor
    task: str
    true_lll_ilabel: list[list[list[int]]]
    true_lll_ilabel_10: torch.Tensor
    """

    def __init__(self,
                 ll_osent_icode_10,
                 lll_ilabel_10,
                 true_lll_ilabel_10,
                 ll_confi_10,
                 loss_10,
                 task,
                 auto_tokenizer):
        """
        The inputs to the constractor are torch.Tensor and in gpu. They get
        converted to lists and cpu before being stored as attributes of the
        class.

        _10 stands for ten-sor

        Parameters
        ----------
        ll_osent_icode_10: torch.Tensor
        lll_ilabel_10: torch.Tensor
        true_lll_ilabel_10: torch.Tensor
        ll_confi_10: torch.Tensor
        loss_10: torch.Tensor
        task: str
        auto_tokenizer: AutoTokenizer
        
        """
        self.ll_osent_icode_10 = ll_osent_icode_10
        self.lll_ilabel_10 = lll_ilabel_10
        self.true_lll_ilabel_10 = true_lll_ilabel_10
        self.ll_confi_10 = ll_confi_10
        self.loss_10 = loss_10

        self.l_orig_sent = None
        self.lll_ilabel = None
        self.true_lll_ilabel = None
        self.ll_confi = None
        self.loss = None

        self.in_cpu = False

        self.task = task
        self.auto_tokenizer = auto_tokenizer

    def move_to_cpu(self):
        """

        Returns
        -------
        None

        """
        self.in_cpu = True
        l_orig_sent2 = \
            MInput.decode_ll_icode(self.ll_osent_icode_10.cpu(),
                                   self.auto_tokenizer)
        if self.task == "ex":
            self.l_orig_sent = l_orig_sent2
        else:
            self.l_orig_sent = l_orig_sent2

        self.lll_ilabel = self.lll_ilabel_10.cpu().to_list()
        self.true_lll_ilabel = self.true_lll_ilabel_10.cpu().to_list()
        self.ll_confi = self.ll_confi_10.cpu().to_list()
        self.loss = float(self.loss_10.cpu())

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
