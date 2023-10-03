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
    
        "verb_mask": list[int], a list of 0, 1, 1 if word in text is a verb and 0 if not
    
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
    pos_mask: list[int]
    true_batch_m_out: MOutput
    verb_locs: list[int]
    verb_mask: list[int]


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
        an embedding or encoding is the vector image for an int or 
        tensor of ints. Hence, an embedding or encoding is a map: 
        A->A^n where A = range(len(A))
        Will call i\in A an icode.

        a = torch.LongTensor([[1, 2, 3, 9], [4, 3, 2, 0]]) # (2, 4)
        embedding(a) has shape (2, 4, 3)
        num_embeddings (int) – len(A)=vocab size=num_icodes
        embedding_dim (int) – n=new_inner_dim
            
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

        self.pos_mask = None
        self.pos_locs = None
        self.verb_mask = None
        self.verb_locs = None
        self.ll_wstart_loc = None

        self.batch_m_out = None
        self.true_batch_m_out = None
        self.eval_out_d = {}  # filled in test_epoch_end()

        # self.init_name_to_param=None #Openie6 has this as Model attribute but
        # not us

    def configure_optimizers(self):
        """
        inherited method

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
            assert "constraint_str" in self.params.d
            num_optimizers = \
                len(self.params.d["constraint_str"].split('_'))
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

    def forward(self, batch_id=-1, ttt='train',
                constraint_str=None, con_weight_str=None):
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
        constraint_str: str
        con_weight_str: str

        Returns
        -------
        torch.tensor, torch.tensor, torch.tensor
            lll_ilabel, ll_confi, batch_loss

        """
        if "wreg" in self.params.d:
            self.init_name_to_param = deepcopy(
                dict(self.named_parameters()))

        # lll_label is similar to openie6 labels
        # first (outer) list over batch/sample of events
        # second list over extractions
        # third (inner) list over number of labels in a line
        # after padding and adding the 3 unused tokens
        batch_size, num_depths, ilabels_length = \
            self.true_batch_m_out.lll_ilabel.shape

        # `loss_fun` is not used in this function anymore
        # loss_fun, lstm_loss = 0, 0
        lll_word_confi = []

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
                ll_pred_ilabel = torch.argmax(lll_word_confi, dim=-1)
                ll_pred_icode = self.embedding(ll_pred_ilabel)
                lll_word_hidden_state += ll_pred_icode

            lll_word_hidden_state = self.merge_layer(lll_word_hidden_state)
            ll_word_confi = self.compress_layer(lll_word_hidden_state)
            lll_word_confi.append(ll_word_confi)

            depth += 1
            if depth >= num_depths:
                break
            if self.params.d["mode"] != 'train':
                ll_prob_ilabel = torch.max(lll_word_confi, dim=2)[1]
                valid_extraction = False
                for l_prob_ilabel in ll_prob_ilabel:
                    # 'ARG1': 1, 'REL': 2
                    if 1 in l_prob_ilabel and 2 in l_prob_ilabel:
                        valid_extraction = True
                        break
                if not valid_extraction:
                    break
        # outsource everything after do loop to a new function
        return self._calc_forward_output(ttt,
                                         lll_word_confi,
                                         constraint_str,
                                         con_weight_str)

    def _calc_forward_output(self,
                             ttt,
                             lll_word_confi,
                             constraint_str,
                             con_weight_str):
        """
        not inherited method. used in forward() method

        Parameters
        ----------
        ttt
        lll_word_confi
        constraint_str
        con_weight_str

        Returns
        -------
        torch.tensor, torch.tensor, torch.tensor
            lll_pred_ilabel, ll_confi, batch_loss


        """
        batch_loss = 0
        lll_pred_ilabel = []  # all_depth_predictions
        ll_confi = []  # all_depth_confidences  
        batch_size, num_words, _ = lll_word_confi.shape
        self.true_batch_m_out.lll_pred_ilabel = \
            self.true_batch_m_out.lll_pred_ilabel.long()
        for depth, ll_word_confi in enumerate(lll_word_confi):
            if ttt == 'train':
                input = ll_word_confi.reshape(batch_size * num_words, -1)
                target = self.true_batch_m_out. \
                             lll_pred_ilabel[:, depth, :].reshape(-1)
                batch_loss += self.loss_fun(input, target)
            else:
                lll_soft_word_confi = torch.log_softmax(lll_word_confi, dim=2)
                ll_max_log_prob, ll_pred_ilabel = \
                    torch.max(lll_soft_word_confi, dim=2)
                # remember: lll_label was similar to labels
                # first (outer) list over batch events
                # second list over extractions
                # third (inner) list over number of labels in a line
                ll_ilabel_mask = \
                    (self.true_batch_m_out.lll_pred_ilabel[:, 0,
                     :] != -100).float()

                # * is element-wise multiplication of tensors
                ll_ilabel_mask = \
                    (ll_pred_ilabel != 0).float() * ll_ilabel_mask
                ll_norm_log_prob = \
                    (ll_max_log_prob * ll_ilabel_mask) \
                    / (1 + ll_ilabel_mask.sum(dim=0))
                l_confi = torch.exp(
                    torch.sum(ll_norm_log_prob, dim=1))

                # this unsqueezes depth dim=1
                lll_pred_ilabel.append(ll_pred_ilabel.unsqueeze(1))
                ll_confi.append(l_confi.unsqueeze(1))

        if ttt == 'train':
            if constraint_str:
                # dim=1 is depth. This cats along depth dimension
                lll_word_confi = torch.cat([ll.unsqueeze(1) for
                                            ll in lll_word_confi], dim=1)
                lll_word_confi = torch.softmax(lll_word_confi, dim=-1)

                con_loss = self._constrained_loss(
                    lll_word_confi,
                    constraint_str,
                    con_weight_str) / batch_size
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
            lll_pred_ilabel = torch.cat(lll_pred_ilabel, dim=1)
            ll_confi = torch.cat(ll_confi, dim=1)

            if constraint_str and \
                    'predict' not in self.params.d["mode"] and \
                    self.params.d["batch_size"] != 1:
                lll_word_confi = torch.cat([ll.unsqueeze(1) for
                                            ll in lll_word_confi], dim=1)
                # this fills tensor with 0's
                lll_word_confi.fill_(0)

                # for checking test set
                # lll_ilabel = copy(lll_pred_ilabel)
                # ll_ilabel[lll_ilabel == -100] = 0
                ilabels = copy(lll_pred_ilabel)

                ilabels = ilabels.unsqueeze(-1)
                ilabels_depth = ilabels.shape[1]
                lll_word_confi = lll_word_confi[:, :ilabels_depth, :, :]
                lll_word_confi.scatter_(3, ilabels.long(), 1)

                constraint_str = 'posm_hvc_hvr_hve'
                con_weight_str = '1_1_1_1'
                l_constraint = constraint_str.split('_')
                l_con_weight = con_weight_str.split('_')
                if len(l_con_weight) == 1:
                    l_con_weight = [con_weight_str] * len(l_constraint)

                for constraint, con_weight in \
                        zip(l_constraint, l_con_weight):
                    con_loss = self._constrained_loss(lll_word_confi,
                                                      constraint,
                                                      con_weight)
                    if constraint not in self.con_to_l_loss:
                        self.con_to_l_loss[constraint] = []
                    self.con_to_l_loss[constraint].append(con_loss)

        return lll_pred_ilabel, ll_confi, batch_loss

    def _constrained_loss(self, lll_word_confi,
                          constraint_str, con_weight_str):
        """
        similar to Openie6.model.constrained_loss()
        not inherited method
        called by forward()

        Parameters
        ----------
        lll_word_confi: torch.tensor
        constraint_str: str
        con_weight_str: str

        Returns
        -------
        float
            hinge_loss

        """
        batch_size, depth, num_words, num_ilabels = lll_word_confi.shape
        hinge_loss = 0
        xx = self.verb_locs.unsqueeze(1).unsqueeze(3). \
            repeat(1, depth, 1, num_ilabels)
        verb_confis = torch.gather(lll_word_confi, 2, xx)
        verb_rel_confis = verb_confis[:, :, :, 2]
        # (batch_size, depth, num_words)
        verb_rel_confis = verb_rel_confis * (self.verb_locs != 0). \
            unsqueeze(1).float()

        # every head-verb must be included in a relation
        if 'hvc' in constraint_str:
            column_loss = torch.abs(1 - torch.sum(verb_rel_confis, dim=1))
            column_loss = column_loss[self.verb_locs != 0]
            hinge_loss += con_weight_str * column_loss.sum()

        # extractions must have at least k-relations with
        # a head verb in them
        if 'hvr' in constraint_str:
            row_rel_loss = F.relu(self.verb_mask.sum(dim=1).float() -
                                  torch.max(verb_rel_confis, dim=2)[0].sum(
                                      dim=1))
            hinge_loss += con_weight_str * row_rel_loss.sum()

        # one relation cannot contain more than one head verb
        if 'hve' in constraint_str:
            ex_loss = F.relu(torch.sum(verb_rel_confis, dim=2) - 1)
            hinge_loss += con_weight_str * ex_loss.sum()

        if 'posm' in constraint_str:
            xx = self.pos_locs.unsqueeze(1).unsqueeze(3). \
                repeat(1, depth, 1, num_ilabels)
            pos_confis = torch.gather(lll_word_confi, 2, xx)
            pos_nnone_confis = \
                torch.max(pos_confis[:, :, :, 1:], dim=-1)[0]
            column_loss = (1 - torch.max(pos_nnone_confis, dim=1)[0]) * \
                          (self.pos_locs != 0).float()
            hinge_loss += con_weight_str * column_loss.sum()

        return hinge_loss

    def training_step(self, batch_id, optimizer_id=-1):
        """
        inherited method

        Parameters
        ----------
        batch_id: int
        optimizer_id: int

        Returns
        -------
        float
            batch_loss

        """
        if "multi_opt" in self.params.d:
            assert "constraint_str" in self.params.d
            constraint_str = self.params.d["constraint_str"].split('_')[
                optimizer_id]
            con_weight_str = float(
                self.params.d["con_weight_str"].split('_')[optimizer_id])
        else:
            constraint_str = self.params.d["constraint_str"]
            con_weight_str = self.params.d["con_weight_str"]

        lll_ilabel, ll_confi, batch_loss = \
            self.forward(mode='train',
                         batch_id=batch_id,
                         constraint_str=constraint_str,
                         con_weight_str=con_weight_str)

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
        lll_ilabel, ll_confi, loss = self.forward(
            mode=self.params.mode,
            constraint_str=self.params.d["constraint_str"],
            con_weight_str=self.params.d["con_weight_str"])

        tune_out_d = {"lll_ilabel": lll_ilabel,
                      "ll_confi": ll_confi,
                      "ground_truth": self.true_batch_m_out.lll_ilabel,
                      "meta_data": self.batch_m_out.meta_data}
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
            lll_ilabel = self.batch_m_out.lll_ilabel
            ll_confi = self.batch_m_out.ll_confi
            l_orig_sent = self.batch_m_out.l_orig_sent
            true_lll_ilabel = self.true_batch_m_out.lll_ilabel
            ll_confi = (ll_confi * 100).round() / 100

        if self.params.task == "cc":
            if 'predict' in self.params.mode:
                metrics_d = {'P_exact': 0, 'R_exact': 0, 'F1_exact': 0}
            else:
                num_samples = len(lll_ilabel)
                for k in enumerate(range(num_samples)):
                    if type(l_orig_sent[k][0]) != str:
                        l_orig_sent[k] = [self.auto_tokenizer.decode[m] for
                                          m in l_orig_sent]
                    self.metric(lll_ilabel[k],
                                true_lll_ilabel[k],
                                l_orig_sent[k])
                metrics_d = self.metric.get_score_d(do_reset=True)

            val_acc = metrics_d["F1_exact"]
            eval_out_d = {"eval_f1": val_acc,
                          "eval_p": metrics_d["P_exact"],
                          "eval_r": metrics_d["R_exact"]}

        elif self.params.task == "ex":
            if 'predict' in self.params.d["mode"]:
                metrics_d = {'carb_f1': 0,
                             'carb_auc': 0,
                             'carb_last_f1': 0}
            else:
                num_samples = len(self.batch_m_out.lll_ilabel)
                for k in range(num_samples):
                    if type(l_orig_sent[k][0]) != str:
                        l_orig_sent[k] = [self.auto_tokenizer.decode[m]
                                          for m in
                                          l_orig_sent[k]]
                    self.metric(l_orig_sent[k],
                                lll_ilabel[k],
                                true_lll_ilabel[k])
                metrics_d = self.metric.get_score_d(do_reset=True)

            eval_out_d = {"eval_f1": metrics_d["carb_f1"],
                          "eval_auc": metrics_d["carb_auc"],
                          "eval_last_f1": metrics_d["carb_last_f1"]}

        print('\nResults: ' + str(eval_out_d))
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
        pass

    def tune_dataloader(self):
        """
        inherited abstract method

        Returns
        -------
        None

        """
        pass

    def _write_if_task_ex(self):
        """

        Returns
        -------
        list[str], list[str]
            l_pred_str, l_pred_allen_str

        """
        fix_d = self.metric.fix_d

        lll_ilabel = self.batch_m_out.lll_ilabel
        ll_confi = self.batch_m_out.ll_confi
        num_samples, num_depths, _ = lll_ilabel.shape

        l_orig_sentL = [self.batch_m_out.l_orig_sent[k]
                        + UNUSED_TOKENS_STR for
                        k in range(num_samples)]

        orig_sent_to_pred_l_ex = {}
        for sample_id, orig_sentL in enumerate(l_orig_sentL):
            orig_sent = undoL(orig_sentL)
            if fix_d:
                orig_sent0 = fix_d[orig_sent]
                if orig_sent0 not in orig_sent_to_pred_l_ex:
                    orig_sent_to_pred_l_ex[orig_sent0] = []
            else:
                if orig_sent not in orig_sent_to_pred_l_ex:
                    orig_sent_to_pred_l_ex[orig_sent] = []
            for depth in range(num_depths):
                num_words = len(get_words(orig_sentL))
                ex_ilabels = lll_ilabel[sample_id][depth][:num_words]
                if sum(ex_ilabels) == 0:  # extractions completed
                    break
                ex = SaxExtraction.get_ex_from_ilabels(
                    ex_ilabels, orig_sentL, ll_confi[sample_id][depth].item())
                if ex.arg1 and ex.rel:
                    if fix_d:
                        orig_sent0 = fix_d[orig_sent]
                        if ex.is_not_in(
                                orig_sent_to_pred_l_ex[orig_sent0]):
                            orig_sent_to_pred_l_ex[orig_sent0]. \
                                append(ex)
                    else:  # no fix_d
                        if ex.is_not_in(
                                orig_sent_to_pred_l_ex[orig_sent]):
                            orig_sent_to_pred_l_ex[orig_sent].append(ex)
        l_pred_str = []
        l_pred_allen_str = []
        for sample_id, pred_ex_sent in enumerate(orig_sent_to_pred_l_ex):
            pred_l_ex = orig_sent_to_pred_l_ex[pred_ex_sent]
            orig_sentL = l_orig_sentL[sample_id]
            str0 = f'{pred_ex_sent}\n'
            for pred_ex in pred_l_ex:
                str0 += pred_ex.get_simple_sent() + '\n'
            l_pred_str.append(str0.strip("/n"))
            allen_str = ""
            for pred_ex in pred_l_ex:
                arg1 = pred_ex.arg1
                rel = pred_ex.rel
                arg2 = pred_ex.arg2
                allen_str += f"{orig_sentL}\t"
                allen_str += f"<arg1> {arg1} </arg1>"
                allen_str += f"<rel> {rel} </rel>"
                allen_str += f"<arg2> {arg2} </arg2>\t"
                allen_str += f"{pred_ex.confi}\n"
            l_pred_allen_str.append(allen_str.strip("/n"))
        return l_pred_str, l_pred_allen_str

    def _write_if_task_cc(self):
        """

        Returns
        -------
        list[str]
            l_pred_str

        """
        fix_d = self.metric.fix_d

        sample_id = 0
        correct = True
        total_num_ccsents1 = 0
        total_num_ccsents2 = 0
        lll_ilabel = self.batch_m_out.lll_ilabel
        num_samples, num_depths, _ = lll_ilabel.shape
        # true_lll_ilabel = self.true_batch_m_out.lll_label
        l_orig_sent = [self.batch_m_out.l_orig_sent for
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

        return l_pred_str

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
        l_orig_sent = self.batch_m_out.l_orig_sent
        for k in range(num_samples):
            l_orig_sent = [self.auto_tokenizer.decode[m] for m
                           in l_orig_sent[k]]
        if self.params.task == "ex":
            l_pred_str, l_pred_allen_str = \
                self._write_if_task_ex()
        elif self.params.task == "cc":
            l_pred_str = self._write_if_task_cc()
        else:
            assert False
        fpath = self.params.task + ".txt"
        if batch_id == 0:
            fmode = 'w'
        else:
            fmode = 'a'
        with open(fpath, fmode) as pred_f:
            pred_f.write('\n'.join(l_pred_str) + '\n')
        if self.params.d.task == "ex" and \
                "write_allennlp" in self.params.d:
            fpath = PREDICTIONS_DIR + "/allen.txt"
            if batch_id == 0:
                fmode = "w"
            else:
                fmode = "a"
            with open(fpath, fmode) as allen_f:
                allen_f.write('\n'.join(l_pred_allen_str) + '\n')
