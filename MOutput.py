import torch
from words_tags_ilabels_translation import *
from MInput import *


class MOutput:
    """
    MOutput = Model Output
    This class stores the outputs of Model, usually for one batch.

    Attributes
    ----------
        l_orig_sent: list[str]
            list of original (before splitting) sentences, usually for a batch.
        ll_pred_confidence: torch.Tensor
            predicted confidence for each extraction of each batch sample
        lll_ilabel: torch.Tensor
            if x is the feature vector and y is the classification, this is
            the y. Use this variable to store the ground truth y. (This
            variable is only filled for supervised training, ttt="train")
        lll_pred_ilabel: torch.Tensor
            if x is the feature vector and y is the classification, this is
            the y. Use this variable to store the predicted y.
        loss: float
            batch loss
    """

    def __init__(self,
                 l_orig_sent,
                 lll_ilabel,
                 lll_pred_ilabel,
                 ll_pred_confidence,
                 loss):
        """
        Constructor


        Parameters
        ----------
        l_orig_sent: list[str]
        lll_ilabel: torch.Tensor
        lll_pred_ilabel: torch.Tensor
        ll_pred_confidence: torch.Tensor
        loss: float
        
        """
        self.l_orig_sent = l_orig_sent
        self.lll_ilabel = lll_ilabel
        self.lll_pred_ilabel = lll_pred_ilabel
        self.ll_pred_confidence = ll_pred_confidence
        self.loss = loss

    def move_to_cpu(self):
        """
        Moves data from gpu to cpu.

        Returns
        -------
        None

        """

        self.lll_ilabel = self.lll_ilabel.cpu()
        self.lll_pred_ilabel = self.lll_pred_ilabel.cpu()
        self.ll_pred_confidence = self.ll_pred_confidence.cpu()
        # self.loss = self.loss.cpu()

        self.ll_pred_confidence = \
            (self.ll_pred_confidence * 100).round() / 100

    # def get_l_orig_sent(self):
    #     """
    #
    #     Returns
    #     -------
    #     list[str]
    #
    #     """
    #     l_orig_sent2 = \
    #         MInput.decode_ll_icode(Li(self.ll_osent_icode),
    #                                self.auto_tokenizer)
    #     if self.task == "ex":
    #         return undoL(l_orig_sent2)
    #     else:
    #         return l_orig_sent2
    #
    # def get_lll_word(self):

    #     translator = translate_ilabels_to_words
    #     l_orig_sentL = redoL(self.get_l_orig_sent())
    #     num_samples = len(self.llll_pred_ilabel)
    #     num_depths = len(self.llll_pred_ilabel[0])
    #     # sent_len = len(self.lll_ilabel[0][0])
    #     lll_word = []
    #     for sam in range(num_samples):
    #         ll_word = []
    #         for depth in range(num_depths):
    #             ll_word.append(
    #                 translator(self.llll_pred_ilabel[sam][depth],
    #                            l_orig_sentL[sam]))
    #         lll_word.append(ll_word)
    #     return lll_word
