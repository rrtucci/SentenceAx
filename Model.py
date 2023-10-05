from SaxExtraction import *
from ExMetric import *
from CCMetric import *
from CCTree import *
from MOutput import *

import os
from copy import copy, deepcopy
from collections import OrderedDict
import logging
import regex as re

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from transformers import AdamW, AutoModel

# prevents printing of model weights, etc
logging.getLogger(
    'transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger(
    'transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


class Model(pl.LightningModule):
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

        def training_step(self, batch, batch_id):
            x, y = batch
            y_hat = self(x)
            loss_fun = F.cross_entropy(y_hat, y)
            return loss_fun

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

    This class has an abstract class as its parent. Uninherited methods
    start with an underscore.

    
        batch_d={
        "lll_label": np.array of ints, shape:(batch_size, depth, labels_length)
    
        "meta_data": any
    
        "pos_locs": int
    
        "text": str
    
        "verb_bools": list[int], a list of 0, 1, 1 if word in text is a verb and 0 if not
    
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
    base_model: AutoModel
    batch_m_out: MOutput
    dropout_fun: nn.Dropout
    eval_out_d: dict[str, Any]
    hidden_size: int
    icodes: list[int]
    illabelling_layer: self.base_model.encoder.layer
    iterative_transformer: self.base_model.encoder.layer
    ll_wstart_loc: listlist[[int]]
    loss_fun: nn.CrossEntropyLoss
    merge_layer: nn.Linear
    metric: CCMetric | ExMetric
    name_to_param: dict[str, Any]
    params: Params
    pos_locs: list[int]
    pos_bools: list[int]
    true_batch_m_out: MOutput
    verb_locs: list[int]
    verb_bools: list[int]


    """

    def __init__(self, params, auto_tokenizer):
        """
        lightning/src/lightning/pytorch/core/module.py
        Parameters
        ----------
        params: Params
        auto_tokenizer: AutoTokenizer
        """
        super().__init__(self)
        self.params = params
        self.init_name_to_param = None
        self.auto_tokenizer = auto_tokenizer

        self.base_model = AutoModel.from_pretrained(
            self.params.d["model_str"], cache_dir=CACHE_DIR)
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
        possible icodes here is 10.

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
        self.embedding = nn.Embedding(NUM_ICODES,  # 100
                                      self.hidden_size)
        self.merge_layer = nn.Linear(self.hidden_size,
                                     ILABELLING_DIM)  # 300
        self.compress_layer = nn.Linear(ILABELLING_DIM,  # 300
                                        NUM_ILABELS)  # 6

        self.loss_fun = nn.CrossEntropyLoss()

        if self.params.task == "ex":
            self.metric = ExMetric(self.params.d)
        elif self.params.task == "cc":
            self.metric = CCMetric()

        self.pos_bools = None
        self.ll_pos_loc = None
        self.ll_verb_bool = None
        self.ll_verb_loc = None
        self.ll_wstart_loc = None

        self.l_batch_m_out = None
        self.eval_out_d = {}  # filled in test_epoch_end()

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

        tqdm derives from the Arabic word taqaddum which can mean "progress"
        and is an abbreviation for "I love you so much" in Spanish (te
        quiero demasiado).

        Returns
        -------
        dict[str, Any]
            tqdm_d

        """
        # get avg_training_loss
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if \
            running_train_loss else float('NaN')
        # get `best` as float
        if type(self.trainer.checkpoint_callback.kth_value) != float:
            best = self.trainer.checkpoint_callback.kth_value.item()
        else:
            best = self.trainer.checkpoint_callback.kth_value

        tqdm_d = OrderedDict()
        tqdm_d['loss_fun'] = '{:.3f}'.format(avg_training_loss)
        tqdm_d['best'] = best
        return tqdm_d

    def forward(self, batch_id=-1, ttt='train'):
        """
        inherited method
        signature of parent method:  def forward(self, *args, **kwargs)
        
        wreg = weight regulator (default =0)
        loss_fun = loss_fun + wreg*weight_diff
        
        The following methods invoke forward() once:
        training_step(), validation_step(), test_step()
        
        Parameters
        ----------
        mode: str
        batch_id: int

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            lll_ilabel, lll_confi, batch_loss

        """
        if "wreg" in self.params.d:
            self.init_name_to_param = deepcopy(
                dict(self.named_parameters()))

        # lll_label is similar to openie6 labels
        # first (outer) list over batch/sample of events
        # second list over extractions
        # third (inner) list over number of labels in a line
        # after padding and adding the 3 unused tokens
        batch_size, num_depths, num_words = \
            self.true_batch_m_out.lll_ilabel.shape

        # `loss_fun` is not used in this function anymore
        # loss_fun, lstm_loss = 0, 0
        llll_word_confi = []

        lll_hidden_state, _ = self.base_model()

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
            #
            lll_loc = self.ll_wstart_loc.unsqueeze(2). \
                repeat(1, 1, lll_hidden_state.shape[2])
            lll_word_hidden_state = torch.gather(
                input=lll_hidden_state,
                dim=1,
                index=lll_loc)

            if depth != 0:
                lll_pred_ilabel = torch.argmax(llll_word_confi, dim=-1)
                lll_pred_code = self.embedding(lll_pred_ilabel)
                lll_word_hidden_state += lll_pred_code

            lll_word_hidden_state = self.merge_layer(lll_word_hidden_state)
            lll_word_confi = self.compress_layer(lll_word_hidden_state)
            llll_word_confi.append(lll_word_confi)

            depth += 1
            if depth >= num_depths:
                break
            if self.params.d["mode"] != 'train':
                ll_prob_ilabel = torch.max(lll_word_confi, dim=2)[1]
                valid_extraction = False
                assert self.params.d["task"] == "ex"
                for l_prob_ilabel in ll_prob_ilabel:
                    # 'ARG1': 1, 'REL': 2
                    if 1 in l_prob_ilabel and 2 in l_prob_ilabel:
                        valid_extraction = True
                        break
                if not valid_extraction:
                    break
        # outsource everything after do loop to a new function
        return self._calc_forward_output(ttt,
                                         llll_word_confi)

    def _calc_forward_output(self,
                             ttt,
                             llll_word_confi):
        """
        not inherited method. used in forward() method

        Parameters
        ----------
        ttt
        llll_word_confi

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            llll_pred_ilabel, lll_confi, batch_loss


        """
        batch_loss = 0
        llll_pred_ilabel = []  # all_depth_predictions
        lll_confi = []  # all_depth_confidences  
        batch_size, num_words, _ = llll_word_confi.shape
        self.true_batch_m_out.llll_pred_ilabel = \
            self.true_batch_m_out.llll_pred_ilabel.long()
        for depth, lll_word_confi in enumerate(llll_word_confi):
            if ttt == 'train':
                input = lll_word_confi.reshape(batch_size * num_words, -1)
                target = self.true_batch_m_out. \
                             llll_pred_ilabel[:, depth, :].reshape(-1)
                batch_loss += self.loss_fun(input, target)
            else:
                lll_soft_word_confi = torch.log_softmax(llll_word_confi, dim=2)
                ll_max_log_prob, lll_pred_ilabel = \
                    torch.max(lll_soft_word_confi, dim=2)
                # remember: lll_label was similar to labels
                # first (outer) list over batch events
                # second list over extractions
                # third (inner) list over number of labels in a line
                ll_pred_bool = \
                    (self.true_batch_m_out.llll_pred_ilabel[:, 0,
                     :] != -100).float()

                # * is element-wise multiplication of tensors
                ll_pred_bool = \
                    (lll_pred_ilabel != 0).float() * ll_pred_bool
                ll_norm_log_prob = \
                    (ll_max_log_prob * ll_pred_bool) \
                    / (1 + ll_pred_bool.sum(dim=0))
                l_confi = torch.exp(
                    torch.sum(ll_norm_log_prob, dim=1))

                # this unsqueezes depth dim=1
                llll_pred_ilabel.append(lll_pred_ilabel.unsqueeze(1))
                lll_confi.append(l_confi.unsqueeze(1))

        if ttt == 'train':
            if self.constraint_str:
                # dim=1 is depth. This cats along depth dimension
                llll_word_confi = torch.cat([ll.unsqueeze(1) for
                                             ll in llll_word_confi], dim=1)
                llll_word_confi = torch.softmax(llll_word_confi, dim=-1)

                con_loss = self._constrained_loss(
                    llll_word_confi,
                    self.con_to_weight) / batch_size
                batch_loss = con_loss

            if "wreg" in self.params.d["wreg"]:
                weight_diff = 0
                name_to_param = dict(self.named_parameters())
                for name in self.init_name_to_param:
                    weight_diff += torch.norm(name_to_param[name]
                                              - self.init_name_to_param[name])
                batch_loss += self.params.d["wreg"] * weight_diff
        else:  # not training
            # if A and B are of shape (3, 4):
            # torch.cat([A, B], dim=0) will be of shape (6, 4)
            # torch.stack([A, B], dim=0) will be of shape (2, 3, 4)
            llll_pred_ilabel = torch.cat(llll_pred_ilabel, dim=1)
            lll_confi = torch.cat(lll_confi, dim=1)

            if self.constraint_str and \
                    'predict' not in self.params.d["mode"] and \
                    self.params.d["batch_size"] != 1:
                llll_word_confi = torch.cat([ll.unsqueeze(1) for
                                             ll in llll_word_confi], dim=1)
                # this fills tensor with 0's
                llll_word_confi.fill_(0)

                # for checking test set
                # lll_ilabel = copy(llll_pred_ilabel)
                # ll_ilabel[lll_ilabel == -100] = 0
                lll_ilabel = copy(llll_pred_ilabel)

                llll_ilabel = lll_ilabel.unsqueeze(-1)
                number_depths = llll_ilabel.shape[1]
                llll_word_confi = llll_word_confi[:, :number_depths, :, :]
                llll_word_confi.scatter_(
                    dim=3,
                    index=llll_ilabel.long(),
                    src=1)

                for constraint, con_weight in self.con_to_weight.items():
                    con_loss = self._constrained_loss(llll_word_confi,
                                                      {constraint: con_weight})
                    if constraint not in self.con_to_l_loss:
                        self.con_to_l_loss[constraint] = []
                    self.con_to_l_loss[constraint].append(con_loss)

        return llll_pred_ilabel, lll_confi, batch_loss

    def _constrained_loss(self,
                          llll_word_confi,
                          con_to_weight):
        """
        similar to Openie6.model.constrained_loss()
        not inherited method
        called by forward()

        Parameters
        ----------
        llll_word_confi: torch.Tensor
        con_to_weight: dict[str, float]

        Returns
        -------
        float
            hinge_loss

        """
        batch_size, num_depths, num_words, icode_dim = llll_word_confi.shape
        hinge_loss = 0
        llll_index = self.ll_verb_loc.unsqueeze(1).unsqueeze(3). \
            repeat(1, num_depths, 1, icode_dim)
        llll_verb_confi = torch.gather(
            input=llll_word_confi,
            dim=2,
            index=llll_index)
        lll_verb_rel_confi = llll_verb_confi[:, :, :, 2]
        # (batch_size, depth, num_words)
        lll_bool = (self.ll_verb_loc != 0).unsqueeze(1).float()

        lll_verb_rel_confi = lll_verb_rel_confi * lll_bool
        # every head-verb must be included in a relation
        if 'hvc' in con_to_weight:
            ll_column_loss = \
                torch.abs(1 - torch.sum(lll_verb_rel_confi, dim=1))
            ll_column_loss = ll_column_loss[self.ll_verb_loc != 0]
            hinge_loss += con_to_weight['hvc'] * ll_column_loss.sum()

        # extractions must have at least k-relations with
        # a head verb in them
        if 'hvr' in con_to_weight:
            l_a = self.ll_verb_bool.sum(dim=1).float()
            l_b = torch.max(lll_verb_rel_confi, dim=2)[0].sum(dim=1)
            row_rel_loss = F.relu(l_a - l_b)
            hinge_loss += con_to_weight['hvr'] * row_rel_loss.sum()

        # one relation cannot contain more than one head verb
        if 'hve' in con_to_weight:
            ll_ex_loss = F.relu(torch.sum(lll_verb_rel_confi, dim=2) - 1)
            hinge_loss += con_to_weight['hve'] * ll_ex_loss.sum()

        if 'posm' in con_to_weight:
            llll_index = self.ll_pos_loc.unsqueeze(1).unsqueeze(3). \
                repeat(1, num_depths, 1, icode_dim)
            llll_confi = torch.gather(
                input=llll_word_confi,
                dim=2,
                index=llll_index)
            lll_pos_not_none_confi = \
                torch.max(llll_confi[:, :, :, 1:], dim=-1)[0]
            ll_column_loss = \
                (1 - torch.max(lll_pos_not_none_confi, dim=1)[0]) * \
                (self.ll_pos_loc != 0).float()
            hinge_loss += con_to_weight['posm'] * ll_column_loss.sum()

        return hinge_loss

    def training_step(self, batch_id):
        """
        inherited method

        Parameters
        ----------
        batch_id: int
        use_all_con: bool

        Returns
        -------
        float
            batch_loss

        """

        _, _, batch_loss = self.forward(batch_id=batch_id,
                                        ttt='train')

        return batch_loss

    def validation_step(self, batch_id):
        """
        inherited method

        Parameters
        ----------
        batch_id: int

        Returns
        -------
        dict[str, Any]
            tune_out_d

        """
        lll_ilabel, lll_confi, loss = self.forward(
            batch_id,
            "tune")

        tune_out_d = {"lll_ilabel": lll_ilabel,
                      "lll_confi": lll_confi,
                      "ground_truth": self.true_batch_m_out.lll_ilabel,
                      "l_orig_sent": self.batch_m_out.ll_osent_icode}
        tune_out_d = OrderedDict(tune_out_d)

        if self.params.d["mode"] != 'test':
            self._write_output(batch_id)

        return tune_out_d

    def test_step(self, batch_id):
        """
        inherited method
        test_step() and validation_step() are identical. They invoke
        forward() once. The following methods invoke forward() once:
        training_step(), validation_step(), test_step()

        Parameters
        ----------
        batch_id

        Returns
        -------
        dict[str, Any]
            tune_out_d

        """
        return self.validation_step(batch_id)

    def _eval_metrics_at_epoch_end(self, ttt):
        """
        similar to Openie6.model.evaluation_end()
        not inherited method, used in *_epoch_end methods
        note that both `mode` and self.params.d["mode"] are used

        `outputs` similar to `self.l_batch_m_out`

        Parameters
        ----------
        ttt: str
            either "train", "tune", "test"

        Returns
        -------
        dict[str, Any]
            eval_out_d

        """
        eval_out_d = None
        if self.params.d["mode"] == 'test':
            for batch_m_out in self.l_batch_m_out:
                lll_ilabel = batch_m_out.lll_ilabel.cpu()
                lll_confi = batch_m_out.lll_confi.cpu()
                l_orig_sent = batch_m_out.l_orig_sent.cpu()
                true_lll_ilabel = batch_m_out.true_lll_ilabel.cpu()
                lll_confi = (lll_confi * 100).round() / 100

        if self.params.task == "cc":
            if 'predict' in self.params.mode:
                metrics_d = {'P_exact': 0, 'R_exact': 0, 'F1_exact': 0}
            else:
                for batch_m_out in l_batch_m_out
                    batch_m_out.l_orig_sent = \
                        MInput.decode_ll_icode(ll_osent_icode,
                                               self.auto_tokenizer)
                    self.metric(batch_m_out.l_orig_sent,
                                batch_m_out.lll_ilabel,
                                batch_m_out.true_lll_ilabel)
                metrics_d = self.metric.get_score_d(do_reset=True)

            val_acc = metrics_d["F1_exact"]
            # val_auc = 0
            eval_out_d = {"eval_f1": val_acc,
                          "eval_p": metrics_d["P_exact"],
                          "eval_r": metrics_d["R_exact"]}

        elif self.params.task == "ex":
            if 'predict' in self.params.d["mode"]:
                metrics_d = {'carb_f1': 0,
                             'carb_auc': 0,
                             'carb_last_f1': 0}
            else:
                for batch_m_out in self.l_batch_m_out:
                    l_orig_sent = MInput.decode_ll_icode(ll_osent_icode,
                                               self.auto_tokenizer)
                    self.metric(batch_m_out.l_orig_sent,
                                batch_m_out.lll_ilabel,
                                batch_m_out.true_lll_ilabel)
                metrics_d = self.metric.get_score_d(do_reset=True)

            eval_out_d = {"eval_f1": metrics_d["carb_f1"],
                          "eval_auc": metrics_d["carb_auc"],
                          "eval_last_f1": metrics_d["carb_last_f1"]}

        print('\nResults:\n' + str(eval_out_d))
        # For computing the constraint violations
        # if hasattr(self, 'con_to_l_loss') and \
        # self.params.d["constraint_str"] != '':
        #     for key in self.con_to_l_loss:
        #         self.con_to_l_loss[key] = sum(self.con_to_l_loss[key]).item()
        #     print('\nViolations: ', self.con_to_l_loss)
        #     self.con_to_l_loss = dict()
        return eval_out_d

    def validation_epoch_end(self):
        """
        inherited method

        Returns
        -------
        dict[str, Any]
            val_ee_out_d

        """
        eval_out_d = \
            self._eval_metrics_at_epoch_end("tune")
        val_ee_out_d = {}
        if eval_out_d:
            val_ee_out_d = {"log": eval_out_d,
                            "eval_acc": eval_out_d["eval_f1"]}

        return val_ee_out_d

    def test_epoch_end(self):
        """
        inherited method

        Returns
        -------
        dict[str, Any]
            test_ee_out_d

        """
        self.eval_out_d = \
            self._eval_metrics_at_epoch_end('test')
        test_ee_out_d = {"log": self.eval_out_d,
                         "progress_bar": self.eval_out_d,
                         "test_acc": self.eval_out_d["eval_f1"]}
        # self.results = d_eval_results # never used!

        return test_ee_out_d

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

    def _write_if_task_ex(self, batch_id):
        """

        Parameters
        ----------
        batch_id: int

        Returns
        -------
        None

        """
        fix_d = self.metric.fix_d

        lll_ilabel = self.batch_m_out.lll_ilabel
        lll_confi = self.batch_m_out.lll_confi
        num_samples, num_depths, _ = lll_ilabel.shape
        l_orig_sent = self.batch_m_out.ll_osent_icode

        osent_to_l_pred_ex = {}
        for sample_id, orig_sent in enumerate(l_orig_sent):
            orig_sentL = redoL(orig_sent)
            if fix_d:
                orig_sent0 = fix_d[orig_sent]
                if orig_sent0 not in osent_to_l_pred_ex:
                    osent_to_l_pred_ex[orig_sent0] = []
            else:
                if orig_sent not in osent_to_l_pred_ex:
                    osent_to_l_pred_ex[orig_sent] = []
            for depth in range(num_depths):
                num_words = len(get_words(orig_sentL))
                ex_ilabels = lll_ilabel[sample_id][depth][:num_words]
                if sum(ex_ilabels) == 0:  # extractions completed
                    break
                ex = SaxExtraction.get_ex_from_ilabels(
                    ex_ilabels, orig_sentL, lll_confi[sample_id][depth])
                if ex.arg1 and ex.rel:
                    if fix_d:
                        orig_sent0 = fix_d[orig_sent]
                        if ex.is_not_in(
                                osent_to_l_pred_ex[orig_sent0]):
                            osent_to_l_pred_ex[orig_sent0]. \
                                append(ex)
                    else:  # no fix_d
                        if ex.is_not_in(
                                osent_to_l_pred_ex[orig_sent]):
                            osent_to_l_pred_ex[orig_sent].append(ex)
        l_pred_str = []
        l_pred_allen_str = []
        for sample_id, l_pred_ex in enumerate(osent_to_l_pred_ex):
            orig_sentL = redoL(l_orig_sent[sample_id])
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

        fmode = "w" if batch_id == 0 else "a"
        fpath = self.params.task + ".txt"
        with open(fpath, fmode) as pred_f:
            pred_f.write('\n'.join(l_pred_str) + '\n')
        if "write_allennlp" in self.params.d:
            fpath = PREDICTIONS_DIR + "/allen.txt"
            with open(fpath, fmode) as allen_f:
                allen_f.write('\n'.join(l_pred_allen_str) + '\n')

    def _write_if_task_cc(self, batch_id):
        """

        Parameters
        ----------
        batch_id: int

        Returns
        -------
        None

        """
        fix_d = self.metric.fix_d

        sample_id = 0
        correct = True
        total_num_ccsents1 = 0
        total_num_ccsents2 = 0
        lll_ilabel = self.batch_m_out.lll_ilabel
        num_samples, num_depths, _ = lll_ilabel.shape
        # true_lll_ilabel = self.true_batch_m_out.lll_label
        l_orig_sent = [self.batch_m_out.ll_osent_icode for
                       k in range(num_samples)]
        l_pred_str = []
        ll_spanned_word = []
        ll_spanned_loc = []
        for id in range(len(l_orig_sent)):
            sample_id += 1
            orig_sent = l_orig_sent[id]
            ll_ilabel = []
            l_orig_sent = []
            for depth in range(num_depths):
                num_words = len(get_words(orig_sent))
                l_ilabel = lll_ilabel[id][depth][:num_words].tolist()
                ll_ilabel.append(l_ilabel)
            tree = CCTree(orig_sent, ll_ilabel)

            pred_str = orig_sent + '\n'
            ex_sents, spanned_words, l_spanned_locs = tree.ccsents
            ll_spanned_word.append(spanned_words)
            ll_spanned_loc.append(l_spanned_locs)
            total_num_ccsents1 += len(ex_sents)
            total_num_ccsents2 += 1 if len(ex_sents) > 0 else 0
            pred_str += '\n'.join(ex_sents) + '\n'

            l_pred_str.append(pred_str)
        # list1 + list2 is the same as list1.extend(list2)
        self.cc_ll_spanned_word += ll_spanned_word
        self.cc_l_pred_str += l_pred_str
        self.cc_ll_spanned_loc += ll_spanned_loc

        fmode = "w" if batch_id == 0 else "a"
        fpath = self.params.task + ".txt"
        with open(fpath, fmode) as pred_f:
            pred_f.write('\n'.join(l_pred_str) + '\n')

    def _write_output(self, batch_id):
        """
        similar to Openie6.model.write_to_file()

        Parameters
        ----------
        batch_id: int

        Returns
        -------
        None

        """
        num_samples = len(self.batch_m_out.lll_ilabel)
        l_orig_sent = MInput.decode_ll_icode(ll_osent_icode,
                               self.auto_tokenizer)
        if self.params.task == "ex":
            self._write_if_task_ex(batch_id)
        elif self.params.task == "cc":
            self._write_if_task_cc(batch_id)
        else:
            assert False
