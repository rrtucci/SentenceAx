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



    """

    def __init__(self, params_d, auto_tokenizer):
        """
        lightning/src/lightning/pytorch/core/module.py
        Parameters
        ----------
        params_d
        auto_tokenizer
        """
        super().__init__(self)
        self.params_d = params_d
        self.auto_tokenizer = auto_tokenizer

        self.base_model = AutoModel.from_pretrained(
            self.params_d["model_str"], cache_dir=CACHE_DIR)
        self.hidden_size = self.base_model.config.hidden_size

        if self.params_d["iterative_layers"] > 0:
            num_layers = len(self.base_model.encoder.layer)
            num_encoder_layers = \
                num_layers - self.params_d["iterative_layers"]
            self.base_model.encoder.layer = \
                self.base_model.encoder.layer[0:num_encoder_layers]
            self.iterative_transformer = \
                self.base_model.encoder.layer[num_encoder_layers:num_layers]

        else:
            self.iterative_transformer = []


        self.dropout_fun = nn.Dropout(p=DROPOUT)  # 0.0

        """
        nn.Embedding(num_embeddings, embedding_size)
        num_embeddings (int) – size of the dictionary of embeddings
        embedding_dim (int) – the size of each embedding vector
            
        """
        self.ilabel_embeddings = nn.Embedding(NUM_EMBEDDINGS,  # 100
                                             self.hidden_size)
        self.merge_layer = nn.Linear(self.hidden_size,
                                     ILABELING_LEN)  # 300
        self.ilabelling_layer = nn.Linear(ILABELING_LEN,  # 300
                                          NUM_ILABELS)  # 6

        self.loss_fun = nn.CrossEntropyLoss()

        if TASK == "ex":
            self.metric = ExMetric(self.params_d)
        elif TASK == "cc":
            self.metric = CCMetric()

        self.pos_mask = None
        self.pos_locs = None
        self.verb_mask = None
        self.verb_locs = None
        self.l_word_start_loc = None

        self.batch_out = MOutput(TASK)
        self.true_batch_out = MOutput(TASK)
        self.eval_out_d = {}  # filled in test_epoch_end()

        # self.init_params_d=None #Openie6 has this as Model attribute but
        # not us

    def configure_optimizers(self):
        """
        inherited method

        Returns
        -------

        """
        # self.named_parameters() is Iterator[Tuple[str, Parameter]]
        all_param_pairs = list(self.named_parameters())
        # opt= optimizer
        # p = parameter
        opt_p_names = ["bias", "gamma", "beta"]

        def p_in_overlap_of_lists(p, li1, li2):
            return any(p in li1 for p in li2)

        def p_not_in_overlap_of_lists(p, li1, li2):
            return not any(p in li1 for p in li2)

        def cond0(pair):
            p_names, p = pair
            return p_not_in_overlap_of_lists(p, p_names, opt_p_names) and \
                'base_model' in p_names

        def cond1(pair):
            p_names, p = pair
            return p_in_overlap_of_lists(p, p_names, opt_p_names) and \
                'base_model' in p_names

        def cond2(pair):
            p_names, p = pair
            return 'base_model' not in p_names

        opt_param_d = [
            {"params": [pair[1] for pair in all_param_pairs if cond0(pair)],
             "weight_decay_rate": 0.01,
             'lr': self.params_d["lr"]},
            {"params": [pair[1] for pair in all_param_pairs if cond1(pair)],
             "weight_decay_rate": 0.0,
             'lr': self.params_d["lr"]},
            {"params": [pair[1] for pair in all_param_pairs if cond2(pair)],
             'lr': self.params_d["lr"]}
        ]

        if self.params_d["optimizer"] == 'adamW':
            optimizer = AdamW(opt_param_d, lr=1e-3)
        elif self.params_d["optimizer"] == 'adam':
            optimizer = Adam(opt_param_d, lr=1e-3)
        else:
            assert False

        if self.params_d["multi_opt"] and \
                self.params_d["constraints_str"]:
            num_optimizers = len(self.params_d["constraints_str"].split('_'))
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

        """
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if \
            running_train_loss else float('NaN')
        if type(self.trainer.checkpoint_callback.kth_value) != type(0.0):
            best = self.trainer.checkpoint_callback.kth_value.item()
        else:
            best = self.trainer.checkpoint_callback.kth_value

        tqdm_d = OrderedDict()
        tqdm_d['loss_fun'] = '{:.3f}'.format(avg_training_loss)
        tqdm_d['best'] = best
        return tqdm_d

    def forward(self, batch_id=-1, mode='train', 
                constraints_str=None, cweights_str=None):
        """
        inherited method
        signature of parent method:  def forward(self, *args, **kwargs)
        
        wreg = weight regulator (default =0)
        loss_fun = loss_fun + wreg*weight_diff
        
        The following methods invoke forward() once:
        training_step(), validation_step(), test_step()
        
        Parameters
        ----------
        mode
        batch_id
        constraints_str
        cweights_str

        Returns
        -------

        """
        if self.params_d["wreg"] and \
                not hasattr(self, 'init_params_d'):
            init_params_d = deepcopy(
                dict(self.named_parameters()))

        # lll_label is similar to openie6 labels
        # first (outer) list over batch/sample of events
        # second list over extractions
        # third (inner) list over number of labels in a line
        # after padding and adding the 3 unused tokens
        batch_size, max_depth, ilabels_length = \
            self.true_batch_out.lll_ilabel.shape

        # `loss_fun` is not used in this function anymore
        # loss_fun, lstm_loss = 0, 0
        ll_word_score = []
        word_scores = []

        hidden_states, _ = self.base_model()

        d = 0
        while True:
            for layer in self.iterative_transformer:
                # layer(hidden_states)[0] returns a copy
                # of the tensor hidden_states after transforming it
                # in some way
                #[0] chooses first component
                hidden_states = layer(hidden_states)[0]

            hidden_states = self.dropout_fun(hidden_states)
            # a chaptgpt generated explanation of this transformation
            # is given in misc/hidden_states_transformation2.txt
            #
            xx = self.l_word_start_loc.unsqueeze(2). \
                repeat(1, 1, hidden_states.shape[2])
            word_hidden_states = torch.gather(hidden_states, 1, xx)

            if d != 0:
                greedy_ilabels = torch.argmax(word_scores, dim=-1)
                ilabel_embeddings = self.ilabel_embeddings(greedy_ilabels)
                word_hidden_states += ilabel_embeddings

            word_hidden_states = self.merge_layer(word_hidden_states)
            word_scores = self.ilabelling_layer(word_hidden_states)
            ll_word_score.append(word_scores)

            d += 1
            if d >= max_depth:
                break
            if self.params_d["mode"] != 'train':
                predictions = torch.max(word_scores, dim=2)[1]
                valid_ext = False
                for p in predictions:
                    if 1 in p and 2 in p:
                        valid_ext = True
                        break
                if not valid_ext:
                    break
        # outsource everything after do loop to a new function
        return self._calc_forward_output(
            init_params_d,
            mode,
            ll_word_score,
            constraints_str, 
            cweights_str)

    def _calc_forward_output(
            self,
            init_params_d,
            mode,
            ll_word_score,
            constraints_str, 
            cweights_str):
        """
        not inherited method. used in forward() method

        Parameters
        ----------
        mode
        ll_word_score
        constraints_str
        cweights_str

        Returns
        -------

        """

        word_scores = ll_word_score[-1]
        batch_loss = 0
        lll_ilabel = []
        ll_score = []
        batch_size, num_words, _ = word_scores.shape
        self.true_batch_out.lll_ilabel = self.true_batch_out.lll_ilabel.long()
        for d, word_scores in enumerate(ll_word_score):
            if mode == 'train':
                batch_loss += self.loss_fun(
                    word_scores.reshape(batch_size * num_words, -1),
                    self.true_batch_out.lll_ilabel[:, d, :].reshape(-1))
            else:
                word_log_probs = torch.log_softmax(word_scores, dim=2)
                max_log_probs, predictions = \
                    torch.max(word_log_probs, dim=2)
                # remember: lll_label was similar to labels
                # first (outer) list over batch events
                # second list over extractions
                # third (inner) list over number of labels in a line
                padding_ilabels = (
                        self.true_batch_out.lll_ilabel[:, 0, :] != -100).float()

                sro_lll_ilabel = \
                    (predictions != 0).float() * padding_ilabels
                log_probs_norm_ext_len = \
                    (max_log_probs * sro_lll_ilabel) \
                    / (sro_lll_ilabel.sum(dim=0) + 1)
                scores = torch.exp(
                    torch.sum(log_probs_norm_ext_len, dim=1))

                lll_ilabel.append(predictions.unsqueeze(1))
                ll_score.append(scores.unsqueeze(1))

        if mode == 'train':
            if constraints_str:
                ll_word_score = torch.cat([ws.unsqueeze(1) for
                                           ws in ll_word_score], dim=1)
                ll_word_score = torch.softmax(ll_word_score, dim=-1)

                const_loss = self._constrained_loss(
                    ll_word_score,
                    constraints_str, cweights_str) / batch_size
                batch_loss = const_loss

            if self.params_d["wreg"] != 0:
                weight_diff = 0
                current_parameters = dict(self.named_parameters())
                for name in init_params_d:
                    weight_diff += torch.norm(current_parameters[name]
                                              - init_params_d[name])
                batch_loss = batch_loss + self.params_d["wreg"] * weight_diff
        else:  # not train
            # if A and B are of shape (3, 4):
            # torch.cat([A, B], dim=0) will be of shape (6, 4)
            # torch.stack([A, B], dim=0) will be of shape (2, 3, 4)
            lll_ilabel = torch.cat(lll_ilabel, dim=1)
            ll_score = torch.cat(ll_score, dim=1)


            if constraints_str and \
                    'predict' not in self.params_d["mode"] and \
                    self.params_d["batch_size"] != 1:
                ll_word_score = torch.cat([d.unsqueeze(1) for
                                           d in ll_word_score], dim=1)
                ll_word_score.fill_(0)

                # for checking test set
                # labels = copy(self.lll_label)
                # labels[labels == -100] = 0
                ilabels = copy(lll_ilabel)

                ilabels = ilabels.unsqueeze(-1)
                ilabels_depth = ilabels.shape[1]
                ll_word_score = ll_word_score[:, :ilabels_depth, :, :]
                ll_word_score.scatter_(3, ilabels.long(), 1)

                constraints_str = 'posm_hvc_hvr_hve'
                cweights_str = '1_1_1_1'
                l_constraint_str = constraints_str.split('_')
                l_cweight_str = cweights_str.split('_')
                if len(l_constraint_str) != len(l_cweight_str):
                    l_cweight_str = [cweights_str] * len(l_constraint_str)

                for constraint_str, weight_str in \
                        zip(l_constraint_str, l_cweight_str):
                    const_loss = self._constrained_loss(self.pos_locs,
                                                        ll_word_score,
                                                        constraint_str,
                                                        weight_str)
                    if constraint_str not in self.constraints_str_d:
                        self.constraints_str_d[constraint_str] = []
                    self.constraints_str_d[constraint_str].append(const_loss)

        return lll_ilabel, ll_score, batch_loss

    def _constrained_loss(self, ll_word_score,
                          constraints_str, cweights_str):
        """
        similar to Openie6.model.constrained_loss()
        not inherited method
        called by forward()

        Parameters
        ----------
        ll_word_score
        constraints_str
        cweights_str

        Returns
        -------

        """
        batch_size, depth, num_words, num_ilabels = ll_word_score.shape
        hinge_loss = 0
        xx = self.verb_locs.unsqueeze(1).unsqueeze(3). \
            repeat(1, depth, 1, num_ilabels)
        verb_scores = torch.gather(ll_word_score, 2, xx)
        verb_rel_scores = verb_scores[:, :, :, 2]
        # (batch_size, depth, num_words)
        verb_rel_scores = verb_rel_scores * (self.verb_locs != 0). \
            unsqueeze(1).float()

        # every head-verb must be included in a relation
        if 'hvc' in constraints_str:
            column_loss = torch.abs(1 - torch.sum(verb_rel_scores, dim=1))
            column_loss = column_loss[self.verb_locs != 0]
            hinge_loss += cweights_str * column_loss.sum()

        # extractions must have at least k-relations with
        # a head verb in them
        if 'hvr' in constraints_str:
            row_rel_loss = F.relu(self.verb_mask.sum(dim=1).float() -
                                  torch.max(verb_rel_scores, dim=2)[0].sum(
                                      dim=1))
            hinge_loss += cweights_str * row_rel_loss.sum()

        # one relation cannot contain more than one head verb
        if 'hve' in constraints_str:
            ex_loss = F.relu(torch.sum(verb_rel_scores, dim=2) - 1)
            hinge_loss += cweights_str * ex_loss.sum()

        if 'posm' in constraints_str:
            xx = self.pos_locs.unsqueeze(1).unsqueeze(3). \
                repeat(1, depth, 1, num_ilabels)
            pos_scores = torch.gather(ll_word_score, 2, xx)
            pos_nnone_scores = \
                torch.max(pos_scores[:, :, :, 1:], dim=-1)[0]
            column_loss = (1 - torch.max(pos_nnone_scores, dim=1)[0]) * \
                          (self.pos_locs != 0).float()
            hinge_loss += cweights_str * column_loss.sum()
            
        return hinge_loss


    def training_step(self, batch_id, optimizer_id=-1):
        """
        inherited method

        Parameters
        ----------
        batch_id
        optimizer_id

        Returns
        -------

        """
        if self.params_d["multi_opt"]:
            constraints_str = self.params_d["constraints_str"].split('_')[
                optimizer_id]
            cweights_str = float(
                self.params_d["cweights_str"].split('_')[optimizer_id])
        else:
            constraints_str = self.params_d["constraints_str"]
            cweights_str = float(self.params_d["cweights_str"])

        lll_ilabel, ll_score, batch_loss = self.forward(mode='train',
                                batch_id=batch_id,
                                constraints_str=constraints_str,
                                cweights_str=cweights_str)

        return batch_loss

    def validation_step(self, batch_id):
        """
        inherited method

        Parameters
        ----------
        batch_id

        Returns
        -------

        """
        lll_ilabel, ll_score, loss= self.forward(
            mode='val',
            constraints_str=self.params_d["constraints_str"],
            cweights_str=self.params_d["cweights_str"])

        val_out_d = {"lll_ilabel": lll_ilabel,
                     "ll_score": ll_score,
                     "ground_truth": self.true_batch_out.lll_ilabel,
                     "meta_data": self.batch_out.meta_data}
        val_out_d = OrderedDict(val_out_d)

        if self.params_d["mode"] != 'test':

            self._write_output(val_out_d, batch_id, self.params_d["task"])

        return val_out_d

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

        """
        return self.validation_step(batch_id)

    def _eval_metrics_at_epoch_end(self,
                                   mode):
        """
        similar to Openie6.model.evaluation_end()
        not inherited method, used in *_epoch_end methods
        note that both `mode` and self.params_d["mode"] are used

        Parameters
        ----------
        output
        mode

        Returns
        -------

        """
        eval_out_d = None
        if self.params_d["mode"] == 'test':
            self.batch_out.to_cpu()
            self.true_batch_out.to_cpu()
            lll_ilabel = self.batch_out.lll_ilabel
            ll_score = self.batch_out.ll_score
            l_orig_sent = self.batch_out.l_orig_sent
            true_lll_ilabel = self.true_batch_out.lll_ilabel
            ll_score = (ll_score * 100).round() / 100

        if self.params_d["task"] == "cc":
            if 'predict' in self.params_d["mode"]:
                metrics_d = {'P_exact': 0, 'R_exact': 0, 'F1_exact': 0}
            else:
                num_samples = len(lll_ilabel)
                for k in enumerate(num_samples):
                    if type(l_orig_sent[k][0]) != str:
                        l_orig_sent[k] = [self.auto_tokenizer.decode[m] for
                                          m in l_orig_sent]
                    self.metric(lll_ilabel[k],
                                true_lll_ilabel[k],
                                l_orig_sent[k])
                metrics_d = self.metric.get_metric_values(
                    reset=True, mode=mode)

            val_acc = metrics_d["F1_exact"]
            eval_out_d = {"eval_f1": val_acc,
                              "eval_p": metrics_d["P_exact"],
                              "eval_r": metrics_d["R_exact"]}

        elif self.params_d["task"] == "ex":
            if 'predict' in self.params_d["mode"]:
                metrics_d = {'carb_f1': 0, 'carb_auc': 0, 'carb_lastf1': 0}
            else:
                num_samples = len(self.batch_out.lll_ilabel)
                for k in range(num_samples):
                    if type(l_orig_sent[k][0]) != str:
                        l_orig_sent[k] = [self.auto_tokenizer.decode[m]
                                                 for m in
                                                 l_orig_sent[k]]
                    self.metric(lll_ilabel[k],
                                true_lll_ilabel[k],
                                l_orig_sent[k])
                metrics_d = self.metric.get_metric_values(
                    reset=True, mode=mode)

            eval_out_d = {"eval_f1": metrics_d["carb_f1"],
                              "eval_auc": metrics_d["carb_auc"],
                              "eval_lastf1": metrics_d["carb_lastf1"]}

        print('\nResults: ' + str(eval_out_d))
        # For computing the constraint violations
        # if hasattr(self, 'constraints_str_d') and \
        # self.params_d["constraints_str"] != '':
        #     for key in self.constraints_str_d:
        #         self.constraints_str_d[key] = sum(self.constraints_str_d[key]).item()
        #     print('\nViolations: ', self.constraints_str_d)
        #     self.constraints_str_d = dict()
        return eval_out_d

    def validation_epoch_end(self):
        """
        inherited method

        Parameters
        ----------
        output

        Returns
        -------

        """
        eval_out_d = \
            self._eval_metrics_at_epoch_end('dev')
        val_ee_out_d = {}
        if eval_out_d :
            val_ee_out_d = {"log": eval_out_d,
                        "eval_acc": eval_out_d["eval_f1"]}

        return val_ee_out_d

    def test_epoch_end(self):
        """
        inherited method

        Parameters
        ----------
        output

        Returns
        -------

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

        """
        return None

    def val_dataloader(self):
        """
        inherited abstract method

        Returns
        -------

        """
        return None

    def _get_extraction(self, ex_ilabels, orig_sentL, score):
        """
        similar to Openie6.model.process_extraction()

        ILABEL_TO_EXTAG={
            0: 'NONE',
            1: 'ARG1',
            2: 'REL',
            3: 'ARG2',
            4: 'ARG2',
            5: 'NONE'
        }


        Parameters
        ----------
        ex_labels:
        orig_sentL
        score

        Returns
        -------

        """
        ex_ilabels = ex_ilabels.to_list()  # change from torch tensor to list

        l_rel = []
        l_arg1 = []
        l_arg2 = []
        # l_loc_time=[]
        # l_args = []
        rel_case = 0
        for i, word in enumerate(get_words(orig_sentL)):
            if '[unused' in word:
                if ex_ilabels[i] == 2:  # REL
                    rel_case = int(
                        re.search('\[unused(.*)\]', word).group(1)
                    )  # this returns either 1, 2 or 3
                continue
            if ex_ilabels[i] == 0:  # NONE
                pass
            elif ex_ilabels[i] == 1:  # ARG1
                l_arg1.append(word)
            elif ex_ilabels[i] == 2:  # REL
                l_rel.append(word)
            elif ex_ilabels[i] == 3:  # ARG2
                l_arg2.append(word)
            elif ex_ilabels[i] == 4:  # ARG2
                # l_loc_time.append(word)
                l_arg2.append(word)
            else:
                assert False

        rel = ' '.join(l_rel).strip()
        if rel_case == 1:
            rel = 'is ' + rel
        elif rel_case == 2:
            rel = 'is ' + rel + ' of'
        elif rel_case == 3:
            rel = 'is ' + rel + ' from'

        arg1 = ' '.join(l_arg1).strip()
        arg2 = ' '.join(l_arg2).strip()

        # args = ' '.join(l_args).strip()
        # loc_time = ' '.join(l_loc_time).strip()
        # if not self.params_d["no_lt"]: # no_lt = no loc time
        #     arg2 = (arg2 + ' ' + loc_time + ' ' + args).strip()

        extraction = SaxExtraction(orig_sentL,
                                   arg1,
                                   rel,
                                   arg2,
                                   score=score)

        return extraction

    def _write_if_task_ex(self):
        fix_d = self.metric.fix_d

        lll_ilabel = self.batch_out.lll_ilabel
        ll_score = self.batch_out.ll_score
        num_samples, max_depth, _= lll_ilabel.shape

        l_orig_sentL = [self.batch_out.l_orig_sent[k]
                        + UNUSED_TOKENS_STR for
                        k in range(num_samples)]

        orig_sent_to_pred_l_ex = {}
        for sample_id, orig_sentL in enumerate(l_orig_sentL):
            orig_sent = unL(orig_sentL)
            if fix_d:
                orig_sent0 = fix_d[orig_sent]
                if orig_sent0 not in orig_sent_to_pred_l_ex:
                    orig_sent_to_pred_l_ex[orig_sent0] = []
            else:
                if orig_sent not in orig_sent_to_pred_l_ex:
                    orig_sent_to_pred_l_ex[orig_sent] = []
            for depth in range(max_depth):
                num_words = len(get_words(orig_sentL))
                ex_ilabels = lll_ilabel[sample_id][depth][:num_words]
                if sum(ex_ilabels) == 0:  # extractions completed
                    break
                ex = self._get_extraction(
                    ex_ilabels, orig_sentL, ll_score[sample_id][depth].item())
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
                allen_str += f"{pred_ex.score}\n"
            l_pred_allen_str.append(allen_str.strip("/n"))
        return l_pred_str, l_pred_allen_str

    def _write_if_task_cc(self):
        fix_d = self.metric.fix_d

        sample_id = 0
        correct = True
        total_num_cc_sents1 = 0
        total_num_cc_sents2 = 0
        lll_ilabel = self.batch_out.lll_ilabel
        num_samples, max_depth, _ = lll_ilabel.shape
        # true_lll_ilabel = self.true_batch_out.lll_label
        l_orig_sent = [self.batch_out.l_orig_sent for
                        k in range(num_samples)]
        l_pred_str = []
        ll_spanned_word = []
        ll_spanned_loc = []
        for id in range(len(l_orig_sent)):
            sample_id += 1
            orig_sent = l_orig_sent[id]
            ll_ilabel = []
            l_orig_sent = []
            for depth in range(max_depth):
                num_words = len(get_words(orig_sent))
                l_ilabel = lll_ilabel[id][depth][:num_words].tolist()
                ll_ilabel.append(l_ilabel)
            tree = CCTree(orig_sent, ll_ilabel)

            pred_str = orig_sent + '\n'
            ex_sents, spanned_words, l_spanned_locs = tree.cc_sents
            ll_spanned_word.append(spanned_words)
            ll_spanned_loc.append(l_spanned_locs)
            total_num_cc_sents1 += len(ex_sents)
            total_num_cc_sents2 += 1 if len(ex_sents) > 0 else 0
            pred_str += '\n'.join(ex_sents) + '\n'

            l_pred_str.append(pred_str)
        # list1 + list2 is the same as list1.extend(list2)
        self.cc_ll_spanned_word+= ll_spanned_word
        self.cc_l_pred_str += l_pred_str
        self.cc_ll_spanned_loc += ll_spanned_loc

        return l_pred_str

    def _write_output(self, batch_id, task):
        """
        similar to Openie6.model.write_to_file()

        Parameters
        ----------
        sample
        batch_id
        task

        Returns
        -------

        """
        self.batch_out.to_cpu()
        self.true_batch_out.lll_ilabel.to_cpu()
        num_samples = len(self.batch_out.lll_ilabel)
        l_orig_sent = self.batch_out.l_orig_sent
        for k in range(num_samples):
            l_orig_sent = [self.auto_tokenizer.decode[m] for m
                                 in l_orig_sent[k]]
        if task == "ex":
            l_pred_str, l_pred_allen_str = \
                self._write_if_task_ex()
        elif task == "cc":
            l_pred_str = self._write_if_task_cc()
        else:
            assert False
        fpath = TASK + ".txt"
        if batch_id == 0:
            fmode= 'w'
        else:
            fmode = 'a'
        with open(fpath, fmode) as pred_f:
            pred_f.write('\n'.join(l_pred_str) + '\n')
        if task == "ex" and self.params_d["write_allennlp"]:
            fpath = PREDICTIONS_DIR + "/allen.txt"
            if batch_id == 0:
                fmode = "w"
            else:
                fmode = "a"
            with open(fpath, fmode)  as allen_f:
                allen_f.write('\n'.join(l_pred_allen_str) + '\n')
