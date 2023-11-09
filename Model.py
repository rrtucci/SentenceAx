from SaxExtraction import *
from ExMetric import *
from CCMetric import *
from CCTree import *
from MOutput import *
from PaddedMInput import *
from SaxDataSet import *
from PickleList import *

import os
from copy import copy, deepcopy
from collections import OrderedDict
import logging
import regex as re
from pprint import pprint

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import lightning as L
from transformers import AdamW, AutoModel

# prevents printing of model weights, etc
logging.getLogger(
    'transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger(
    'transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

"""on_test_epoch_end() and on_validation_epoch_end() have only been 
available in `lightining` since version 2.0.1 (released Feb 2023)
https://github.com/Lightning-AI/lightning/releases
https://stackoverflow.com/questions/70790473/pytorch-lightning-epoch-end 
-validation-epoch-end.
In addition, note that `pytorch_lightning` has been superceeded by 
`lightning`. 'pytorch_lightning` is now deprecated"""
check_module_version("lightning", "2.0.1")


class Model(L.LightningModule):
    """
    import lightning.pytorch as pl
    import torch.nn as nn
    import torch.nn.functional as F
    class LitModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(28 * 28, 10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss_fun = F.cross_entropy(y_hat, y)
            return loss_fun

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

    This class has an abstract class as its parent. Uninherited methods
    start with `sax_`.


        batch_d={
        "lll_label": np.array of ints, shape:(batch_size, depth, labels_length)

        "meta_data": any

        "pos_locs": int

        "text": str

        "verb_bools": list[int],
            a list of 0, 1, 1 if word in text is a verb and 0 if not

        "verb_locs": list[int], locations of verbs  in text

        "word_starts":
    }

    BERT has a volume L * H * HL
    L = encoding length
    AH = number of attention heads
    HL = number of hidden layers

    BERTBASE (L=12, HL=768, AH=12, Total Parameters=110M)
    BERTLARGE (L=24, HL=1024, AH=16, Total Parameters=340M).


    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    base_model: BertModel
    cc_sent_to_words: dict[str, list[str]]
    con_to_weight: dict[str, float]
    dropout_fun: Dropout
    embedding: Embedding
    scores_epoch_end_d: dict[str, Any]
    _ex_sent_to_sent: dict[str, str]
    hidden_size: int
    ilabelling_layer: Linear
    init_name_to_param: dict[str, variable]
    iterative_transformer: ModuleList
    l_batch_m_out: list[MOutput]
    l_cc_pred_str: list[str]
    l_ex_pred_str: list[str]
    ll_cc_spanned_word: list[list[str]]
    lll_cc_spanned_loc: list[list[list[int]]]
    loss_fun: CrossEntropyLoss
    merge_layer: Linear
    metric: CCMetric | ExMetric
    model_name: str
    params: Params
    verbose_model: bool
    # some inherited attributes that won't be used
    # hparams (dictionary, Used by Openie6, not by us.
    #    We use the class Params instead.)
    # logger
    # trainer
    # on_gpu

    """

    def __init__(self,
                 params,
                 auto_tokenizer,
                 verbose_model,
                 model_name):
        """
        lightning/src/lightning/pytorch/core/module.py
        Parameters
        ----------
        params: Params
        auto_tokenizer: AutoTokenizer
        verbose_model: bool
        """
        super().__init__()
        self.params = params
        self.auto_tokenizer = auto_tokenizer
        self.init_name_to_param = None
        self.verbose_model = verbose_model
        self.model_name = model_name

        # return_dict=False avoids error message from Dropout
        self.base_model = AutoModel.from_pretrained(
            self.params.d["model_str"],
            cache_dir=CACHE_DIR,
            return_dict=False)
        self.hidden_size = self.base_model.config.hidden_size

        if self.params.d["iterative_layers"] > 0:
            num_layers = len(self.base_model.encoder.layer)
            num_encoder_layers = \
                num_layers - self.params.d["iterative_layers"]
            self.base_model.encoder.layer = \
                self.base_model.encoder.layer[0:num_encoder_layers]
            self.iterative_transformer = \
                self.base_model.encoder.layer[num_encoder_layers:num_layers]

        else:
            self.iterative_transformer = []

        self.dropout_fun = nn.Dropout(p=DROPOUT)  # 0.0

        """
        embedding = nn.Embedding(num_embeddings=10, embedding_dim=3)
        an embedding or encoding takes a tensor ll_x with shape (2, 4)
        to a tensor lll_x of shape (2, 4, 3)
        The elements of the tensor will be called icodes. The num of 
        possible icodes here is 10. This is also the vocab size.

        a = torch.LongTensor([[1, 2, 3, 9], [4, 3, 2, 0]]) # (2, 4)
        embedding(a) has shape (2, 4, 3)
        num_embeddings (int) – vocab size, num icodes
        embedding_dim (int) – 3 = len(lll[0][0])

        Embedding is a layer that takes a tensor of icodes to another tensor 
        of icodes with one more index. An encoder takes each word and 
        replaces it by an icode.

        output = nn.Linear(na, nb)(input)
        If input has shape (10, 20, na), then output has shape (10, 20, nb)

        """
        self.embedding = nn.Embedding(
            100,  # vocab size
            self.hidden_size)  # dim of embedding space
        self.merge_layer = nn.Linear(self.hidden_size,
                                     ILABELLING_DIM)  # 300
        self.ilabelling_layer = nn.Linear(ILABELLING_DIM,  # 300
                                          NUM_ILABELS)  # 6

        # ignore_index=-100 is the default, but including it
        # explicitly for clarity
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=-100)

        if self.params.task == "ex":
            self.metric = ExMetric()
        elif self.params.task == "cc":
            self.metric = CCMetric()

        self._ex_sent_to_sent = None  # property
        self.cc_sent_to_words = None

        self.scores_epoch_end_d = {}  # filled in test_epoch_end()

        # self.init_name_to_param=None #Openie6 has this as Model attribute but
        # not us

        if "multi_opt" not in self.params.d \
                or not self.params.d["multi_opt"]:
            constraint_str = ""
            con_weight_str = ""
        else:
            if "constraint_str" not in self.params.d or \
                    "con_weight_str" not in self.params.d:
                constraint_str = 'posm_hvc_hvr_hve'
                con_weight_str = '1_1_1_1'
            else:
                constraint_str = self.params.d["constraint_str"]
                con_weight_str = self.params.d["con_weight_str"]
        l_constraint = constraint_str.split('_')
        l_con_weight = con_weight_str.split('_')
        assert len(l_constraint) == len(l_con_weight)
        self.con_to_weight = {l_constraint[k]: float(l_con_weight[k])
                              for k in range(len(l_constraint)) if
                              l_constraint[k]}

        self.l_cc_pred_str = []  # all_predictions_conj
        self.ll_cc_spanned_word = []  # all_conjunct_words_conj
        self.lll_cc_spanned_loc = []  # all_sentence_indices_conj
        self.l_ex_pred_str = []  # all_predictions_oie

        self.l_batch_m_out = \
            PickleList(f"action_{model_name}_l_batch_m_out_dir")

    @property
    def ex_sent_to_sent(self):
        """

        Returns
        -------
        dict[str, str]

        """
        return self._ex_sent_to_sent

    @ex_sent_to_sent.setter
    def ex_sent_to_sent(self, value):
        """

        Parameters
        ----------
        value: dict[str, str]

        Returns
        -------
        None

        """
        self._ex_sent_to_sent = value
        if self.params.task == "ex":
            self.metric.sent_to_sent = value

    def configure_optimizers(self):
        """
        inherited method

        The optimizer can be Adam or AdamW. If there are multiple
        constraints `multi_con = True`, then an Adam or AdamW optimzer will
        be used for each constraint.

        Returns
        -------
        list[Adam|AdamW]

        """
        # self.named_parameters() is list[Tuple[str, Parameter]]
        all_pairs = list(self.named_parameters())

        # opt= optimizer
        # x = parameter
        # pair = (xname, x)

        def base_model_pairs():
            return [pair for pair in all_pairs if "base_model" in pair[0]]

        def non_base_model_pairs():
            return [pair for pair in all_pairs if "base_model" not in pair[0]]

        xnames = ["bias", "gamma", "beta"]

        def pair_in_xnames(pair):
            return any((pair[0] in xname) for xname in xnames)

        opt_param_d = [
            {"params": [pair[1] for pair in base_model_pairs() if
                        not pair_in_xnames(pair)],
             "weight_decay_rate": 0.01,
             'lr': self.params.d["lr"]},
            {"params": [pair[1] for pair in base_model_pairs() if
                        pair_in_xnames(pair)],
             "weight_decay_rate": 0.0,
             'lr': self.params.d["lr"]},
            {"params": [pair[1] for pair in non_base_model_pairs()],
             'lr': self.params.d["lr"]}
        ]

        if self.params.d["optimizer"] == 'adamW':
            optimizer = AdamW(opt_param_d, lr=1e-3)
        elif self.params.d["optimizer"] == 'adam':
            optimizer = Adam(opt_param_d, lr=1e-3)
        else:
            assert False

        if "multi_opt" in self.params.d:
            num_optimizers = len(self.con_to_weight)
            return [optimizer] * num_optimizers
        else:
            return [optimizer]

    def get_progress_bar_dict(self):
        """
        inherited method

        Additional items to be displayed in the progress bar.

        Openie6 uses tqdm for all progress bars, including this one.
        For this one, we use the one built into lightning.

        tqdm derives from the Arabic word taqaddum which can mean "progress"
        and is an abbreviation for "I love you so much" in Spanish (te
        quiero demasiado).

        Returns
        -------
        Dict[str, Union[int, str]]
            Dictionary with the items to be displayed in the progress bar.


        """
        # # get avg_training_loss
        # running_train_loss = self.trainer.running_loss.mean()
        # avg_training_loss = running_train_loss.cpu().item() if \
        #     running_train_loss else float('NaN')
        # # get `best` as float
        # if type(self.trainer.checkpoint_callback.kth_value) \
        #         not in [int, float]:
        #     best = self.trainer.checkpoint_callback.kth_value.item()
        # else:
        #     best = self.trainer.checkpoint_callback.kth_value
        #
        # tqdm_d = OrderedDict()
        # tqdm_d['loss_fun'] = '{:.3f}'.format(avg_training_loss)
        # tqdm_d['best'] = best
        # return tqdm_d

        # # Get the losses
        # losses = self.log_dict.pop('val_loss', None)
        # val_losses = losses if losses is not None else self.log_dict.pop(
        #     'val_main_loss', None)

        # Get the progress bar
        progress_bar_d = super().get_progress_bar_dict()

        # # Add the losses to the progress bar
        # progress_bar_d['loss'] = self.log_dict['loss']
        # progress_bar_d['epoch_acc'] = self.log_dict['epoch_acc']
        return progress_bar_d

    @staticmethod
    def sax_get_batch_in_dicts(batch):
        """

        Parameters
        ----------
        batch: tuple[torch.Tensor, torch.Tensor, list[str]]

        Returns
        -------
        OrderedDict, dict[str, torch.Tensor], dict[str, list[str]]

        """
        x, y, l_orig_sent, xname_to_l_dim1 = batch
        y_d = {"lll_ilabel": y}
        meta_d = {"l_orig_sent": l_orig_sent}
        xname_to_dim1 = OrderedDict(
            {xname: int(l_dim1[0]) for xname, l_dim1 in
             xname_to_l_dim1.items()})
        x_d = SaxDataSet.invert_cat(x, xname_to_dim1)
        return x_d, y_d, meta_d

    def sax_get_llll_word_score(self, x_d, y_d, ttt):
        """
        used inside self.forward()
        
        Parameters
        ----------
        x_d: OrderedDict
        y_d: dict[str, torch.Tensor]
        ttt: str

        Returns
        -------
        list[torch.Tensor]

        """
        # lll_label is similar to openie6 labels
        # first (outer) list over batch/sample of events
        # second list over extractions
        # third (inner) list over number of labels in a line
        # after padding and adding the 3 unused tokens
        batch_size, num_depths, num_words = y_d["lll_ilabel"].shape
        # sometimes num_depths will exceed max
        if ttt != 'train':
            num_depths = get_num_depths(self.params.task)

        # `loss_fun` is not used in this function anymore
        # loss_fun, lstm_loss = 0, 0

        # batch_text = " ".join(redoL(meta_d["l_orig_sent"]))
        # base_model_input = \
        #     torch.Tensor(self.auto_tokenizer.encode(batch_text))

        lll_hidden_state, _ = self.base_model(x_d["ll_osent_icode"])

        lll_word_score = Ten([0])  # this statement is unecessary
        llll_word_score = []  # similar to all_depth_scores
        depth = 0
        while True:
            for layer in self.iterative_transformer:
                # layer(lll_hidden_state)[0] returns a copy
                # of the tensor lll_hidden_state after transforming it
                # in some way
                # [0] chooses first component
                lll_hidden_state = layer(lll_hidden_state)[0]

            lll_hidden_state = self.dropout_fun(lll_hidden_state)
            # a chaptgpt generated explanation of this transformation
            # is given in misc/hidden_states_transformation2.txt
            lll_loc = x_d["ll_osent_wstart_loc"].unsqueeze(2). \
                repeat(1, 1, lll_hidden_state.shape[2])
            lll_word_hidden_state = torch.gather(
                input=lll_hidden_state,
                dim=1,
                index=lll_loc)

            if depth != 0:
                ll_greedy_ilabel = torch.argmax(lll_word_score, dim=-1)
                # not an integer code/embedding
                lll_pred_code = self.embedding(ll_greedy_ilabel)
                lll_word_hidden_state += lll_pred_code

            lll_word_hidden_state = self.merge_layer(lll_word_hidden_state)
            lll_word_score = self.ilabelling_layer(lll_word_hidden_state)
            llll_word_score.append(lll_word_score)

            depth += 1
            if depth >= num_depths:
                break

            if ttt != 'train':
                ll_pred_ilabel = torch.max(lll_word_score, dim=2)[1]
                valid_extraction = False
                for l_pred_ilabel in ll_pred_ilabel:
                    if is_valid_label_list(
                            l_pred_ilabel, self.params.task, "ilabels"):
                        valid_extraction = True
                        break
                if not valid_extraction:
                    break
        return llll_word_score

    def sax_increment_loss(self,
                           loss,
                           x_d,
                           llll_word_score):
        """
        used inside self.forward()

        Parameters
        ----------
        loss: float
        llll_word_score: list[torch.Tensor]
        x_d: OrderedDict

        Returns
        -------
        float

        """
        batch_size, _, _ = llll_word_score[0].shape
        if self.con_to_weight:
            # dim=1 is depth. This cats along depth dimension
            llll_word_scoreT = torch.cat(
                [lll.unsqueeze(1) for lll in llll_word_score], dim=1)
            llll_word_scoreT = torch.softmax(llll_word_scoreT, dim=-1)

            con_loss = Model.sax_constrained_loss(
                x_d,
                llll_word_scoreT,
                self.con_to_weight) / batch_size
            loss = con_loss

        if "wreg" in self.params.d:
            weight_diff = 0
            name_to_param = dict(self.named_parameters())
            for name in self.init_name_to_param:
                weight_diff += torch.norm(name_to_param[name]
                                          - self.init_name_to_param[name])
            loss += self.params.d["wreg"] * weight_diff
        return loss

    def sax_get_con_to_l_loss(self,
                              x_d,
                              llll_word_score,
                              lll_pred_ilabel0):
        """
        used inside self.forward()
        This method is never used. Never checked

        self.con_to_l_loss similar to self.self._constD in Openie6

        Parameters
        ----------
        x_d: OrderedDict
        llll_word_score: list[torch.Tensor]
        lll_pred_ilabel0: torch.Tensor

        Returns
        -------
        dict[str, list[float]]

        """
        con_to_l_loss = {}
        # this calculates llll_word_score
        if self.constraint_str and \
                'predict' not in self.params.d["action"] and \
                self.params.d["batch_size"] != 1:
            # reshape llll_word_score
            llll_word_scoreT = torch.cat([lll.unsqueeze(1) for
                                          lll in llll_word_score], dim=1)
            # this fills tensor with 0's
            llll_word_scoreT.fill_(0)

            # for checking test set
            # lll_ilabel = copy(lll_pred_ilabel)
            # ll_ilabel[lll_ilabel == -100] = 0
            lll_ilabel = copy(lll_pred_ilabel0)

            llll_ilabel = lll_ilabel.unsqueeze(-1)
            number_depths = llll_ilabel.shape[1]
            llll_word_scoreT = llll_word_scoreT[:, :number_depths, :, :]
            llll_word_scoreT.scatter_(
                dim=3,
                index=llll_ilabel.long(),
                src=1)

            # this uses llll_word_score that was calculated previously
            # to calculate con_to_l_loss
            for constraint, con_weight in self.con_to_weight.items():
                con_loss = Model.sax_constrained_loss(
                    x_d,
                    llll_word_scoreT,
                    {constraint: con_weight})
                if constraint not in con_to_l_loss:
                    con_to_l_loss[constraint] = []
                con_to_l_loss[constraint].append(con_loss)
        return con_to_l_loss

    @staticmethod
    def sax_constrained_loss(x_d,
                             llll_word_scoreT,
                             con_to_weight):
        """
        similar to Openie6.model.constrained_loss()
        used inside self.forward()

        Parameters
        ----------
        x_d: OrderedDict
        llll_word_scoreT: torch.Tensor
        con_to_weight: dict[str, float]

        Returns
        -------
        float
            hinge_loss

        """
        batch_size, num_depths, num_words, icode_dim = llll_word_scoreT.shape
        hinge_loss = 0
        llll_index = x_d["ll_osent_verb_loc"].unsqueeze(1).unsqueeze(3). \
            repeat(1, num_depths, 1, icode_dim)
        llll_verb_confi = torch.gather(
            input=llll_word_scoreT,
            dim=2,
            index=llll_index)
        lll_verb_rel_confi = llll_verb_confi[:, :, :, 2]
        # (batch_size, depth, num_words)
        lll_bool = (x_d["ll_osent_verb_loc"] != 0).unsqueeze(1).float()

        lll_verb_rel_confi = lll_verb_rel_confi * lll_bool
        # every head-verb must be included in a relation
        if 'hvc' in con_to_weight:
            ll_column_loss = \
                torch.abs(1 - torch.sum(lll_verb_rel_confi, dim=1))
            ll_column_loss = \
                ll_column_loss[x_d["ll_osent_verb_loc"] != 0]
            hinge_loss += con_to_weight['hvc'] * ll_column_loss.sum()

        # extractions must have at least k-relations with
        # a head verb in them
        if 'hvr' in con_to_weight:
            l_a = x_d["ll_osent_verb_bool"].sum(dim=1).float()
            l_b = torch.max(lll_verb_rel_confi, dim=2)[0].sum(dim=1)
            row_rel_loss = F.relu(l_a - l_b)
            hinge_loss += con_to_weight['hvr'] * row_rel_loss.sum()

        # one relation cannot contain more than one head verb
        if 'hve' in con_to_weight:
            ll_ex_loss = F.relu(torch.sum(lll_verb_rel_confi, dim=2) - 1)
            hinge_loss += con_to_weight['hve'] * ll_ex_loss.sum()

        if 'posm' in con_to_weight:
            llll_index = x_d["ll_osent_pos_loc"]. \
                unsqueeze(1).unsqueeze(3).repeat(1, num_depths, 1, icode_dim)
            llll_pred_confi = torch.gather(
                input=llll_word_scoreT,
                dim=2,
                index=llll_index)
            lll_pos_not_none_confi = \
                torch.max(llll_pred_confi[:, :, :, 1:], dim=-1)[0]
            ll_column_loss = \
                (1 - torch.max(lll_pos_not_none_confi, dim=1)[0]) * \
                (x_d["ll_osent_pos_loc"] != 0).float()
            hinge_loss += con_to_weight['posm'] * ll_column_loss.sum()

        return hinge_loss

    def forward(self, batch, batch_idx, ttt):
        """
        inherited method
        signature of parent method:  def forward(self, *args, **kwargs)

        wreg = weight regulator (default =0)
        loss_fun = loss_fun + wreg*weight_diff

        The following methods invoke forward() once:
        training_step(), validation_step(), test_step()

        Parameter
        ----------
        batch: tuple[torch.Tensor, torch.Tensor, list[str]]
        batch_idx: int
        ttt: str

        Returns
        -------
        MOutput
            batch_m_out

        """
        x_d, y_d, meta_d = Model.sax_get_batch_in_dicts(batch)
        # print_tensor("y_d['lll_ilabel']", y_d['lll_ilabel'])
        # print_list("y_d['lll_ilabel'][0][0]", y_d['lll_ilabel'][0][0])
        if "wreg" in self.params.d:
            self.init_name_to_param = deepcopy(
                dict(self.named_parameters()))

        # lll_label is similar to openie6 labels
        # first (outer) list over batch/sample of events
        # second list over extractions
        # third (inner) list over number of labels in a line
        # after padding and adding the 3 unused tokens
        # batch_size, num_depths, num_words = y_d["lll_ilabel"].shape

        # `loss_fun` is not used in this function anymore
        # loss_fun, lstm_loss = 0, 0

        # batch_text = " ".join(redoL(meta_d["l_orig_sent"]))
        # base_model_input = \
        #     torch.Tensor(self.auto_tokenizer.encode(batch_text))

        llll_word_score = self.sax_get_llll_word_score(x_d, y_d, ttt)

        # print_tensor("lll_word_score", lll_word_score)
        # print("vvbg", "len(llll_word_score)", len(llll_word_score))
        # print_tensor("llll_word_score[0]", llll_word_score[0])
        loss = 0
        llll_pred_ilabel = []  # = all_depth_predictions
        # lll_pred_ilabel0 = all_depth_predictions after cat dim=1
        lll_pred_confi = []  # = all_depth_confidences
        # ll_pred_confi0 = all_depth_confidences after cat dim=1
        batch_size, num_words, xxx = llll_word_score[0].shape
        #xxx = 6, the number of different ilabels (classes)
        # y_d["lll_ilabel"] = \
        #     y_d["lll_ilabel"].long()
        for depth, lll_word_score in enumerate(llll_word_score):
            if ttt == 'train':
                # here -1 will be the depth
                ll_loss_input = \
                    lll_word_score.reshape(batch_size * num_words, -1)
                # print_tensor("lll_word_score", lll_word_score)
                # print_tensor("ll_loss_input", ll_loss_input)

                # ll_loss_input.shape = (batch_size * num_words, num_classes=6)
                # l_loss_target.shape = (batch_size * num_words, )
                # l_loss_target[i] \in range(6)
                # loss is scalar

                l_loss_target = \
                    y_d["lll_ilabel"][:, depth, :].reshape(-1)
                loss += self.loss_fun(ll_loss_input,
                                      l_loss_target)

                # print("loss shape", loss.shape)
                # print_tensor("l_loss_target", l_loss_target)
                # print("loss", loss)
            else:  # ttt != "train
                lll_soft_word_score = \
                    torch.log_softmax(lll_word_score, dim=2)
                ll_max_log_prob, ll_pred_ilabel = \
                    torch.max(lll_soft_word_score, dim=2)
                # print_tensor("ll_max_log_prob", ll_max_log_prob)
                # print_tensor("ll_pred_ilabel", ll_pred_ilabel)
                # remember: lll_ilabel was similar to labels
                # first (outer) list over batch events
                # second list over extractions
                # third (inner) list over number of ilabels in a line
                # print("ttt, action", ttt, self.params.action)
                # print_tensor("lll_ilabel", y_d["lll_ilabel"])
                ll_nonpad_bool = \
                    (y_d["lll_ilabel"][:, 0, :] != -100).float()
                # print("dfrt", {name: x_d[name].shape for name in x_d.keys()})
                # print_tensor("ll_nonpad_bool", ll_nonpad_bool)
                # print_tensor("(ll_pred_ilabel != 0)",
                #              (ll_pred_ilabel != 0).float())
                # * is element-wise multiplication of tensors

                ll_nonpad_bool = ll_nonpad_bool[:, :ll_pred_ilabel.shape[1]]
                ll_nonpad_bool = \
                    (ll_pred_ilabel != 0).float() * ll_nonpad_bool
                ll_norm_log_prob = \
                    (ll_max_log_prob * ll_nonpad_bool) \
                    / (1 + ll_nonpad_bool.sum(dim=0))
                l_confi = torch.exp(
                    torch.sum(ll_norm_log_prob, dim=1))

                # this unsqueezes depth dim=1
                llll_pred_ilabel.append(ll_pred_ilabel.unsqueeze(1))
                lll_pred_confi.append(l_confi.unsqueeze(1))
        # } on of for depth, lll_word_score
        if ttt == 'train':
            loss = self.sax_increment_loss(
                loss,
                x_d,
                llll_word_score)
        # if A and B are of shape (3, 4):
        # torch.cat([A, B], dim=0) will be of shape (6, 4)
        # torch.stack([A, B], dim=0) will be of shape (2, 3, 4)

        # llll_pred_ilabel: list[tensor]
        # lll_pred_confi: list[tensor]
        if ttt != "train":
            lll_pred_ilabel0 = torch.cat(llll_pred_ilabel, dim=1)
            ll_pred_confi0 = torch.cat(lll_pred_confi, dim=1)
        else:
            lll_pred_ilabel0 = Ten([0])
            ll_pred_confi0 = Ten([0])

        # never used
        # self.con_to_l_loss = self.sax_get_con_to_l_loss(
        #     x_d,
        #     llll_word_scoreT,
        #     lll_pred_ilabel0)

        batch_m_out = MOutput(meta_d["l_orig_sent"],
                              y_d["lll_ilabel"],
                              lll_pred_ilabel0,
                              ll_pred_confi0,
                              loss)
        return batch_m_out

    def sax_ttt_step(self, batch, batch_idx, ttt):
        """

        Parameters
        ----------
        batch: tuple[torch.Tensor, torch.Tensor, list[str]]
        batch_idx: int
        ttt: str

        Returns
        -------
        float
            loss

        """
        if self.verbose_model:
            if ttt == "train":
                str0 = "training_step"
            elif ttt == "tune":
                str0 = "validation_step"
            elif ttt == "test":
                str0 = "test_step"
            else:
                assert False
            if VERBOSE:
                print("Entering Model." + str0 +
                      " method, batch_idx=" + str(batch_idx))

        batch_m_out = self.forward(batch, batch_idx, ttt)

        if ttt not in ["train", "resume"]:
            # only collect batch_m_out if going to score it.
            # only ttt not in ["train", "resume"] are scored
            self.l_batch_m_out.append(batch_m_out)

        if ttt == "tune":
            self.sax_write_batch_sents_out(batch_idx, batch_m_out)

        loss = batch_m_out.loss

        if ttt == "train":
            self.log('train_step_loss', loss,
                     prog_bar=True,
                     logger=True,
                     on_step=True)

        return loss

    def training_step(self, batch, batch_idx):
        """
        inherited method

        Parameters
        ----------
        batch: tuple[torch.Tensor, torch.Tensor, list[str]]
        batch_idx: int

        Returns
        -------
        dict[str, Any]
            step_end_d

        """
        return self.sax_ttt_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        inherited method

        called by `validation_step()`

        Parameters
        ----------
        batch: tuple[torch.Tensor, torch.Tensor, list[str]]
        batch_idx: int

        Returns
        -------
        dict[str, Any]
            step_end_d

        """
        return self.sax_ttt_step(batch, batch_idx, "tune")

    def test_step(self, batch, batch_idx):
        """
        inherited method
        test_step() and validation_step() are identical. They invoke
        forward() once. The following methods invoke forward() once:
        training_step(), validation_step(), test_step()

        Parameters
        ----------
        batch: tuple[torch.Tensor, torch.Tensor, list[str]]
        batch_idx: int

        Returns
        -------
        dict[str, Any]
            step_end_d

        """
        return self.sax_ttt_step(batch, batch_idx, "test")

    def sax_get_scores_at_epoch_end(self, ttt):
        """
        similar to Openie6.model.evaluation_end()
        
        used inside self.sax_on_ttt_epoch_end()
        
        note that both `mode` and self.params.d["action"] are used

        `outputs` similar to `l_batch_m_out`

        Parameters
        ----------
        ttt: str
            either "train", "tune", "test", "pred"

        Returns
        -------
        dict[str, Any]
            scores_epoch_end_d

        """
        # if self.params.action == 'test':
        #     for batch_m_out in self.l_batch_m_out:
        #         batch_m_out.move_to_cpu()

        if 'predict' in self.params.action:
            score_d = self.metric.get_zero_score_d()
        else:
            for k, batch_m_out in enumerate(self.l_batch_m_out):
                if VERBOSE: print("batch id", k)
                if self.params.task == "cc":
                    self.metric(
                        batch_m_out.l_orig_sent,  # meta data
                        batch_m_out.lll_pred_ilabel,  # predictions
                        batch_m_out.lll_ilabel)  # ground truth
                elif self.params.task == "ex":
                    self.metric(
                        batch_m_out.l_orig_sent,  # meta data
                        batch_m_out.lll_pred_ilabel,  # predictions
                        batch_m_out.ll_pred_confi)  # scores
            score_d = self.metric.get_score_d(ttt)

        if self.params.task == "cc":
            epoch_acc = score_d["F1_exact"]
        elif self.params.task == "ex":
            epoch_acc = score_d["F1"]
        else:
            assert False
        scores_epoch_end_d = OrderedDict(score_d)
        scores_epoch_end_d["epoch_acc"] = epoch_acc

        print('\nScores at end of epoch ' +
              str(self.trainer.current_epoch) + ":")
        print(scores_epoch_end_d)
        # For computing the constraint violations
        # if hasattr(self, 'con_to_l_loss') and \
        # self.params.d["constraint_str"] != '':
        #     for key in self.con_to_l_loss:
        #         self.con_to_l_loss[key] = sum(self.con_to_l_loss[key]).item()
        #     print('\nViolations: ', self.con_to_l_loss)
        #     self.con_to_l_loss = dict()
        return scores_epoch_end_d

    def sax_on_ttt_epoch_end(self, ttt):
        """

        Parameters
        ----------
        ttt: str

        Returns
        -------
        dict[str, Any]
            epoch_end_d

        """
        if self.verbose_model:
            if ttt == "train":
                assert False
            elif ttt == "tune":
                str0 = "on_validation_epoch_end"
            elif ttt == "test":
                str0 = "on_test_epoch_end"
            else:
                assert False
            if VERBOSE: print("Entering Model." + str0 + " method")

        self.scores_epoch_end_d = \
            self.sax_get_scores_at_epoch_end(ttt)
        # epoch_end_d = {"log": scores_epoch_end_d,
        #                "epoch_acc": scores_epoch_end_d["epoch_acc"]}
        # if ttt == "test":
        #     epoch_end_d["progress_bar"] = self.scores_epoch_end_d

        epoch_acc = self.scores_epoch_end_d["epoch_acc"]
        self.log("epoch_acc", epoch_acc,
                 prog_bar=True,
                 logger=True,
                 on_epoch=True)

        self.l_batch_m_out.restart()
        # self.l_batch_m_out.clear()  # free memory

        return epoch_acc

    def on_validation_epoch_end(self):
        """

        Returns
        -------
        dict[str, Any]
            epoch_end_d

        """
        return self.sax_on_ttt_epoch_end("tune")

    def on_test_epoch_end(self):
        """
        inherited method

        Returns
        -------
        dict[str, Any]
            epoch_end_d

        """
        return self.sax_on_ttt_epoch_end("test")

    def train_dataloader(self):
        """
        inherited abstract method

        Returns
        -------
        None

        """
        return

    def val_dataloader(self):
        """
        inherited abstract method

        Returns
        -------
        None

        """
        return

    def sax_write_if_task_ex(self, batch_idx, batch_m_out):
        """

        called by `sax_write_batch_sents_out()`

        Parameters
        ----------
        batch_idx: int
        batch_m_out: MOutput

        Returns
        -------
        None

        """
        lll_ilabel = batch_m_out.lll_pred_ilabel
        ll_confi = batch_m_out.ll_pred_confi
        num_samples, num_depths, _ = lll_ilabel.shape
        l_orig_sent = batch_m_out.l_orig_sent

        osent_to_l_pred_ex = {}
        for sample_id, orig_sent in enumerate(l_orig_sent):
            orig_sentL = redoL(orig_sent)
            add_key_to_target_d(key=orig_sent,
                                fix_d=self.ex_sent_to_sent,
                                target_d=osent_to_l_pred_ex)
            for depth in range(num_depths):
                num_words = len(get_words(orig_sentL))
                ex_ilabels = lll_ilabel[sample_id][depth][:num_words]
                if sum(ex_ilabels) == 0:  # extractions completed
                    break
                ex = SaxExtraction.get_ex_from_ilabels(
                    ex_ilabels, orig_sentL, ll_confi[sample_id][depth])
                if ex.arg1 and ex.rel:
                    add_key_value_pair_to_target_d(
                        key=orig_sent,
                        value=ex,
                        fix_d=self.ex_sent_to_sent,
                        target_d=osent_to_l_pred_ex)
        l_pred_str = []  # similar to `all_pred`
        l_pred_allen_str = []  # similar to `all_pred_allen_nlp`
        for osent, l_pred_ex in osent_to_l_pred_ex.items():
            orig_sentL = redoL(osent)
            str0 = ""
            for pred_ex in l_pred_ex:
                str0 += pred_ex.get_simple_sent() + '\n'
            l_pred_str.append(str0.strip("/n"))
            allen_str = ""
            for pred_ex in l_pred_ex:
                arg1 = pred_ex.arg1
                rel = pred_ex.rel
                arg2 = pred_ex.arg2
                allen_str += f"{orig_sentL}\t"
                allen_str += f"<arg1> {arg1} </arg1>"
                allen_str += f"<rel> {rel} </rel>"
                allen_str += f"<arg2> {arg2} </arg2>\t"
                allen_str += f"{pred_ex.confi}\n"
            l_pred_allen_str.append(allen_str.strip("/n"))

        fmode = "w" if batch_idx == 0 else "a"
        out_fp = VAL_OUT_DIR + "/ex_out_.txt"
        with open(out_fp, "a") as pred_f:
            pred_f.write('\n'.join(l_pred_str) + '\n')
        if self.params.d["write_allen_file"]:
            al_out_fp = VAL_OUT_DIR + "/ex_out_allen.txt"
            with open(al_out_fp, fmode) as allen_f:
                allen_f.write('\n'.join(l_pred_allen_str) + '\n')

        self.l_ex_pred_str = l_pred_str

    def sax_write_if_task_cc(self, batch_idx, batch_m_out):
        """

        called by `sax_write_batch_sents_out()`

        Parameters
        ----------
        batch_idx: int
        batch_m_out: MOutput

        Returns
        -------
        None

        """

        # correct = True
        total_num_ccsents1 = 0
        total_num_ccsents2 = 0
        lll_ilabel = batch_m_out.lll_ilabel
        num_samples, num_depths, _ = lll_ilabel.shape
        # true_lll_ilabel = self.true_batch_m_out.lll_label
        l_orig_sent = batch_m_out.l_orig_sent
        l_cc_pred_str = []
        ll_cc_spanned_word = []
        lll_cc_spanned_loc = []
        l_pred_str = []
        ll_spanned_word = []
        lll_spanned_loc = []
        for sam, orig_sent in enumerate(l_orig_sent):
            ll_ilabel = []
            for depth in range(num_depths):
                num_words = len(get_words(orig_sent))
                l_ilabel = lll_ilabel[sam][depth][:num_words].tolist()
                ll_ilabel.append(l_ilabel)

            pred_str = orig_sent + '\n'
            # split_sentences, conj_words, sentence_indices_list = \
            #       data.coords_to_sentences(pred_coords, words)
            # this function returns
            # return word_sentences, conj_words, sentences
            # openie6.data.coords_to_sentences()
            # is similar to
            # CCTree.set_ccsents()
            # split_sentences, conj_words, sentence_indices_list
            # is similar to
            #  ccsents, l_spanned_word, ll_spanned_loc

            tree = CCTree(orig_sent, ll_ilabel)
            ccsents = tree.ccsents  # split_sentences
            spanned_words = \
                tree.l_spanned_word  # conj_words
            ll_spanned_loc = \
                tree.ll_spanned_loc  # sentence_indices_list
            ll_spanned_word.append(spanned_words)
            lll_spanned_loc.append(ll_spanned_loc)
            total_num_ccsents1 += len(ccsents)
            total_num_ccsents2 += 1 if len(ccsents) > 0 else 0
            pred_str += '\n'.join(ccsents) + '\n'

            l_pred_str.append(pred_str)
        # list1 + list2 is the same as list1.extend(list2)
        ll_cc_spanned_word += ll_spanned_word
        l_cc_pred_str += l_pred_str
        lll_cc_spanned_loc += lll_spanned_loc

        fmode = "w" if batch_idx == 0 else "a"
        out_fp = VAL_OUT_DIR + "/cc_out.txt"
        with open(out_fp, fmode) as pred_f:
            pred_f.write('\n'.join(l_pred_str) + '\n')

        self.l_cc_pred_str = l_cc_pred_str
        self.ll_cc_spanned_word = ll_cc_spanned_word
        self.lll_cc_spanned_loc = lll_cc_spanned_loc

    def sax_write_batch_sents_out(self, batch_idx, batch_m_out):
        """
        similar to Openie6.model.write_to_file()

        called by self.validation_step()

        Parameters
        ----------
        batch_idx: int
        batch_m_out: MOutput

        Returns
        -------
        None

        """
        # batch_m_out = self.l_batch_m_out[batch_idx]
        # batch_m_out.move_to_cpu()

        if self.params.task == "ex":
            self.sax_write_if_task_ex(batch_idx, batch_m_out)
        elif self.params.task == "cc":
            self.sax_write_if_task_cc(batch_idx, batch_m_out)
        else:
            assert False
