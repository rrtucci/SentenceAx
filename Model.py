from SAXExtraction import *
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

import threading
from threading import Thread

sem = threading.Semaphore()

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
            loss = F.cross_entropy(y_hat, y)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

    output_d = {
        "meta_data":
        "ground_truth":
        "loss":
        "predictions":
        "scores":
    }

    This class has an abstract class as its parent. Uninherited methods
    start with an underscore.

    """

    def __init__(self, params_d, auto_tokenizer):
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

        self.dropout = nn.Dropout(p=DROPOUT)  # 0.0

        """
        nn.Embedding(num_embeddings, embedding_size)
        num_embeddings (int) – size of the dictionary of embeddings
        embedding_dim (int) – the size of each embedding vector
            
        """
        self.label_embeddings = nn.Embedding(NUM_EMBEDDINGS,  # 100
                                             self.hidden_size)
        self.merge_layer = nn.Linear(self.hidden_size,
                                     LABELLING_DIM)  # 300
        self.labelling_layer = nn.Linear(LABELLING_DIM,  # 300
                                         NUM_LABELS)  # 6

        self.loss = nn.CrossEntropyLoss()

        self.metric = ExMetric(self.params_d) \
            if self.params_d["task"] == "ex" else CCMetric()
        
        self.output = MOutput()



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
            running_train_loss is not None else float('NaN')
        if type(self.trainer.checkpoint_callback.kth_value) != type(0.0):
            best = self.trainer.checkpoint_callback.kth_value.item()
        else:
            best = self.trainer.checkpoint_callback.kth_value
        tqdm = {'loss': '{:.3f}'.format(avg_training_loss), 'best': best}
        return tqdm

    def forward(self, batch_d, mode='train', batch_id=-1,
                constraints_str=None, cweights_str=None):
        """
        inherited method
        signature of parent method:  def forward(self, *args, **kwargs)
        
        wreg = weight regulator (default =0)
        loss = loss + wreg*weight_diff
        
        Parameters
        ----------
        batch_d
        mode
        batch_id
        constraints_str
        cweights_str

        Returns
        -------

        """
        if self.params_d["wreg"] and \
                not hasattr(self, 'init_params_d'):
            self.init_params_d = deepcopy(
                dict(self.named_parameters()))

        # lll_label is similar to (in openie6, labels)
        # first (outer) list over batch/sample of events
        # second list over extractions
        # third (inner) list over number of labels in a line
        # after padding and adding the 3 unused tokens
        batch_size, depth, labels_length = batch_d["lll_label"].shape
        if mode != 'train':
            depth = MAX_DEPTH

        # `loss` is not used in this function anymore
        # loss, lstm_loss = 0, 0
        hidden_states, _ = self.base_model(batch_d["text"])
        output_d = {}
        l_word_scores = []
        word_scores = []

        d = 0
        while True:
            for layer in self.iterative_transformer:
                hidden_states = layer(hidden_states)[0]

            hidden_states = self.dropout(hidden_states)
            bat = batch_d["word_starts"].unsqueeze(2). \
                repeat(1, 1, hidden_states.shape[2])
            word_hidden_states = torch.gather(hidden_states, 1, bat)

            if d != 0:
                greedy_labels = torch.argmax(word_scores, dim=-1)
                label_embeddings = self.label_embeddings(greedy_labels)
                word_hidden_states += label_embeddings

            word_hidden_states = self.merge_layer(word_hidden_states)
            word_scores = self.labelling_layer(word_hidden_states)
            l_word_scores.append(word_scores)

            d += 1
            if d >= depth:
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
        return self._fill_output_d_loss(
            mode, output_d,
            batch_d, l_word_scores,
            constraints_str, cweights_str)

    def _fill_output_d_loss(
            self, mode, output_d,
            batch_d, l_word_scores,
            constraints_str, cweights_str):
        """
        not inherited method. used in forward() method

        Parameters
        ----------
        mode
        output_d
        batch_d
        l_word_scores
        constraints_str
        cweights_str

        Returns
        -------

        """

        word_scores = l_word_scores[-1]
        loss = 0
        l_predictions = []
        ll_score = []
        batch_size, num_words, _ = word_scores.shape
        batch_d["lll_label"] = batch_d["lll_label"].long()
        for d, word_scores in enumerate(l_word_scores):
            if mode == 'train':
                loss += self.loss(
                    word_scores.reshape(batch_size * num_words, -1),
                    batch_d["lll_label"][:, d, :].reshape(-1))
            else:
                word_log_probs = torch.log_softmax(word_scores, dim=2)
                max_log_probs, predictions = \
                    torch.max(word_log_probs, dim=2)
                # remember: lll_label was similar to labels
                # first (outer) list over batch events
                # second list over extractions
                # third (inner) list over number of labels in a line
                padding_labels = (
                        batch_d["lll_label"][:, 0, :] != -100).float()

                sro_label_predictions = \
                    (predictions != 0).float() * padding_labels
                log_probs_norm_ext_len = \
                    (max_log_probs * sro_label_predictions) \
                    / (sro_label_predictions.sum(dim=0) + 1)
                confidences = torch.exp(
                    torch.sum(log_probs_norm_ext_len, dim=1))

                l_predictions.append(predictions.unsqueeze(1))
                ll_score.append(confidences.unsqueeze(1))

        if mode == 'train':
            if constraints_str:
                l_word_scores = torch.cat([ws.unsqueeze(1) for
                                           ws in l_word_scores], dim=1)
                l_word_scores = torch.softmax(l_word_scores, dim=-1)

                const_loss = self._constrained_loss(
                    l_word_scores, batch_d,
                    constraints_str, cweights_str) / batch_size
                loss = const_loss

            if self.params_d["wreg"] != 0:
                weight_diff = 0
                current_parameters = dict(self.named_parameters())
                for name in self.init_params_d:
                    weight_diff += torch.norm(current_parameters[name]
                                              - self.init_params_d[name])
                loss = loss + self.params_d["wreg"] * weight_diff
        else:  # not train
            # if A and B are of shape (3, 4):
            # torch.cat([A, B], dim=0) will be of shape (6, 4)
            # torch.stack([A, B], dim=0) will be of shape (2, 3, 4)
            l_predictions = torch.cat(l_predictions, dim=1)
            ll_score = torch.cat(ll_score, dim=1)

            self.output.lll_prediction = l_predictions
            self.output.ll_score = ll_score

            if constraints_str and \
                    'predict' not in self.params_d["mode"] and \
                    self.params_d["batch_size"] != 1:
                l_word_scores = torch.cat([d.unsqueeze(1) for
                                           d in l_word_scores], dim=1)
                l_word_scores.fill_(0)

                # for checking test set
                # labels = copy(batch_d["lll_label"])
                # labels[labels == -100] = 0
                labels = copy(l_predictions)

                labels = labels.unsqueeze(-1)
                labels_depth = labels.shape[1]
                l_word_scores = l_word_scores[:, :labels_depth, :, :]
                l_word_scores.scatter_(3, labels.long(), 1)

                constraints_str = 'posm_hvc_hvr_hve'
                cweights_str = '1_1_1_1'
                l_constraint = constraints_str.split('_')
                l_cweight = cweights_str.split('_')
                if len(l_constraint) != len(l_cweight):
                    l_cweight = [cweights_str] * len(l_constraint)

                for constraint, weight in \
                        zip(l_constraint, l_cweight):
                    const_loss = self._constrained_loss(l_word_scores,
                                                        batch_d, constraint,
                                                        float(weight))
                    if constraint not in self.constraints_str_d:
                        self.constraints_str_d[constraint] = []
                    self.constraints_str_d[constraint].append(const_loss)

        self.output.loss = loss
        return output_d

    def _constrained_loss(self, l_word_scores, batch_d,
                          constraints_str, cweights_str):
        """
        similar to model.constrained_loss()
        not inherited method

        Parameters
        ----------
        l_word_scores
        batch_d
        constraints_str
        cweights_str

        Returns
        -------

        """
        batch_size, depth, num_words, num_labels = l_word_scores.shape
        hinge_loss = 0
        bat = batch_d["verb_index"].unsqueeze(1).unsqueeze(3). \
            repeat(1, depth, 1, num_labels)
        verb_scores = torch.gather(l_word_scores, 2, bat)
        verb_rel_scores = verb_scores[:, :, :, 2]
        # (batch_size, depth, num_words)
        verb_rel_scores = verb_rel_scores * (batch_d["verb_index"] != 0). \
            unsqueeze(1).float()

        # every head-verb must be included in a relation
        if 'hvc' in constraints_str:
            column_loss = torch.abs(1 - torch.sum(verb_rel_scores, dim=1))
            column_loss = column_loss[batch_d["verb_index"] != 0]
            hinge_loss += cweights_str * column_loss.sum()

        # extractions must have at least k-relations with
        # a head verb in them
        if 'hvr' in constraints_str:
            row_rel_loss = F.relu(batch_d["verb"].sum(dim=1).float() -
                                  torch.max(verb_rel_scores, dim=2)[0].sum(
                                      dim=1))
            hinge_loss += cweights_str * row_rel_loss.sum()

        # one relation cannot contain more than one head verb
        if 'hve' in constraints_str:
            ex_loss = F.relu(torch.sum(verb_rel_scores, dim=2) - 1)
            hinge_loss += cweights_str * ex_loss.sum()

        if 'posm' in constraints_str:
            bat = batch_d["pos_index"].unsqueeze(1).unsqueeze(3). \
                repeat(1, depth, 1, num_labels)
            pos_scores = torch.gather(l_word_scores, 2, bat)
            pos_nnone_scores = \
                torch.max(pos_scores[:, :, :, 1:], dim=-1)[0]
            column_loss = (1 - torch.max(pos_nnone_scores, dim=1)[0]) * \
                          (batch_d["pos_index"] != 0).float()
            hinge_loss += cweights_str * column_loss.sum()

        return hinge_loss

    def training_step(self, batch_d, batch_id, optimizer_id=-1):
        """
        inherited method

        Parameters
        ----------
        batch_d
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

        output_d = self.forward(batch_d, mode='train',
                                batch_id=batch_id,
                                constraints_str=constraints_str,
                                cweights_str=cweights_str)
        tqdm_d = {"train_loss": self.output.loss}
        output0_d = OrderedDict({"loss": self.output.loss, "log": tqdm_d})

        return output0_d

    def validation_step(self, batch_d, batch_id):
        """
        inherited method

        Parameters
        ----------
        batch_d
        batch_id

        Returns
        -------

        """
        output_d = self.forward(
            batch_d,
            mode='val',
            constraints_str=self.params_d["constraints_str"],
            cweights_str=self.params_d["cweights_str"])

        output0_d = {"l_predictions": self.output.lll_prediction,
                     "ll_score": self.output.ll_score,
                     "ground_truth": batch_d["lll_label"],
                     "meta_data": batch_d["meta_data"]}
        output0_d = OrderedDict(output0_d)

        if self.params_d["mode"] != 'test':
            if self.params_d["write_async"]:
                t = Thread(target=self._write_output,
                           args=(output0_d, batch_id, self.params_d["task"]))
                t.start()
            else:
                self._write_output(output0_d, batch_id, self.params_d["task"])

        return output0_d

    def test_step(self, batch_d, batch_id):
        """
        inherited method

        Parameters
        ----------
        batch_d
        batch_id

        Returns
        -------

        """
        return self.validation_step(batch_d, batch_id)

    def _eval_metrics_at_epoch_end(self,
                                   l_output_d,
                                   mode):
        """
        similar to model.evaluation_end()
        not inherited method, used in *_epoch_end methods
        note that both `mode` and self.params_d["mode"] are used

        Parameters
        ----------
        l_output_d
        mode

        Returns
        -------

        """
        eval_results_d = None
        if self.params_d["mode"] == 'test':
            for output_index, output_d in enumerate(l_output_d):
                self.output.lll_prediction = self.output.lll_prediction.cpu()
                self.output.ll_score = self.output.ll_score.cpu()
                self.output.ll_score = \
                    (self.output.ll_score * 100).round() / 100
                self.output.ground_truth = self.output.ground_truth.cpu()
                self.output.meta_data = self.output.meta_data.cpu()
        if self.params_d["task"] == "cc":
            if 'predict' in self.params_d["mode"]:
                metrics_d = {'P_exact': 0, 'R_exact': 0, 'F1_exact': 0}
            else:
                for output_d in l_output_d:
                    if type(self.output.meta_data[0]) != str:
                        self.output.meta_data = [self.auto_tokenizer.decode[m]
                                                 for m in
                                                 self.output.meta_data]
                    self.metric(self.output.lll_prediction,
                                self.output.ground_truth,
                                meta_data=self.output.meta_data)
                metrics_d = self.metric.get_metric_values(reset=True, mode=mode)

            val_acc = metrics_d["F1_exact"]
            eval_results_d = {"eval_f1": val_acc,
                              "eval_p": metrics_d["P_exact"],
                              "eval_r": metrics_d["R_exact"]}

        elif self.params_d["task"] == "ex":
            if 'predict' in self.params_d["mode"]:
                metrics_d = {'carb_f1': 0, 'carb_auc': 0, 'carb_lastf1': 0}
            else:
                for output_d in l_output_d:
                    if type(self.output.meta_data[0]) != str:
                        self.output.meta_data = [self.auto_tokenizer.decode[m]
                                                 for m in
                                                 self.output.meta_data]
                    self.metric(self.output.lll_prediction,
                                self.output.meta_data,
                                self.output.ll_score)
                metrics_d = self.metric.get_metric_values(reset=True, mode=mode)

            eval_results_d = {"eval_f1": metrics_d["carb_f1"],
                              "eval_auc": metrics_d["carb_auc"],
                              "eval_lastf1": metrics_d["carb_lastf1"]}

        print('\nResults: ' + str(eval_results_d))
        # For computing the constraint violations
        # if hasattr(self, 'constraints_str_d') and \
        # self.params_d["constraints_str"] != '':
        #     for key in self.constraints_str_d:
        #         self.constraints_str_d[key] = sum(self.constraints_str_d[key]).item()
        #     print('\nViolations: ', self.constraints_str_d)
        #     self.constraints_str_d = dict()
        return eval_results_d

    def validation_epoch_end(self, l_output_d):
        """
        inherited method

        Parameters
        ----------
        l_output_d

        Returns
        -------

        """
        eval_results_d = \
            self._eval_metrics_at_epoch_end(l_output_d, 'dev')
        result_d = {}
        if eval_results_d :
            result_d = {"log": eval_results_d,
                        "eval_acc": eval_results_d["eval_f1"]}

        return result_d

    def test_epoch_end(self, l_output_d):
        """
        inherited method

        Parameters
        ----------
        l_output_d

        Returns
        -------

        """
        eval_results_d = \
            self._eval_metrics_at_epoch_end(l_output_d, 'test')
        # self.l_output_d = l_output_d # never used
        results_d = {"log": eval_results_d,
                     "progress_bar": eval_results_d,
                     "test_acc": eval_results_d["eval_f1"]}
        # self.results = d_eval_results # never used!
        if self.params_d["write_async"]:
            while not sem.acquire(blocking=True):
                pass
            sem.release()
        return results_d

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

    def _get_extraction(self, ex_labels, orig_sentL, score):
        """
        similar to model.process_extraction()

        LABEL_TO_EXTAG={
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
        ex_labels = ex_labels.to_list()  # change from torch tensor to list

        l_rel = []
        l_arg1 = []
        l_arg2 = []
        # l_loc_time=[]
        # l_args = []
        rel_case = 0
        for i, word in enumerate(get_words(orig_sentL)):
            if '[unused' in word:
                if ex_labels[i] == 2:  # REL
                    rel_case = int(
                        re.search('\[unused(.*)\]', word).group(1)
                    )  # this returns either 1, 2 or 3
                continue
            if ex_labels[i] == 0:  # NONE
                pass
            elif ex_labels[i] == 1:  # ARG1
                l_arg1.append(word)
            elif ex_labels[i] == 2:  # REL
                l_rel.append(word)
            elif ex_labels[i] == 3:  # ARG2
                l_arg2.append(word)
            elif ex_labels[i] == 4:  # ARG2
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

        extraction = SAXExtraction(orig_sentL,
                                   arg1,
                                   rel,
                                   arg2,
                                   confidence=score)

        return extraction

    def _write_if_task_ex(self, output_d):
        fix_d = self.metric.fix_d

        lll_prediction = self.output.lll_prediction
        l_orig_sentL = self.output.meta_data
        ll_score = self.output.ll_score
        num_sents, ex_depth, max_sent_len = \
            lll_prediction.shape
        assert num_sents == len(l_orig_sentL)
        orig_sent_to_pred_l_ex = {}
        for sample_id, orig_sentL in enumerate(l_orig_sentL):
            orig_sent = orig_sentL.split('[unused1]')[0].strip()
            if fix_d:
                orig_sent0 = fix_d[orig_sent]
                if orig_sent0 not in orig_sent_to_pred_l_ex:
                    orig_sent_to_pred_l_ex[orig_sent0] = []
            else:
                if orig_sent not in orig_sent_to_pred_l_ex:
                    orig_sent_to_pred_l_ex[orig_sent] = []
            for depth in range(ex_depth):
                num_words = len(get_words(orig_sentL))
                ex_labels = lll_prediction[sample_id][depth][:num_words]
                if sum(ex_labels) == 0:  # extractions completed
                    break
                ex = self._get_extraction(
                    ex_labels, orig_sentL, ll_score[sample_id][depth].item())
                if ex.arg1_pair[0] and ex.rel_pair[0]:
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
                str0 += pred_ex.get_str() + '\n'
            l_pred_str.append(str0.strip("/n"))
            allen_str = ""
            for pred_ex in pred_l_ex:
                arg1 = pred_ex.arg1_pair[0]
                rel = pred_ex.rel_pair[0]
                arg2 = pred_ex.arg2_pair[0]
                allen_str += f"{orig_sentL}\t"
                allen_str += f"<arg1> {arg1} </arg1>"
                allen_str += f"<rel> {rel} </rel>"
                allen_str += f"<arg2> {arg2} </arg2>\t"
                allen_str += f"{pred_ex.confidence}\n"
            l_pred_allen_str.append(allen_str.strip("/n"))
        return l_pred_str, l_pred_allen_str

    def _write_if_task_cc(self, output_d):
        fix_d = self.metric.fix_d

        sample_id = 0
        correct = True
        total_num_ex_sents1 = 0
        total_num_ex_sents2 = 0
        lll_prediction = self.output.lll_prediction
        # thruth = self.output.ground_truth"]
        l_orig_sentL = self.output.meta_data
        total_depth = lll_prediction.shape[1]
        l_pred_str = []
        l_spanned_words = []
        ll_spanned_loc = []
        for id in range(len(l_orig_sentL)):
            sample_id += 1
            orig_sentL = l_orig_sentL[id]
            ll_label = []
            l_orig_sent = []
            for depth in range(total_depth):
                num_words = len(get_words(orig_sentL))
                l_label = lll_prediction[id][depth][:num_words].tolist()
                ll_label.append(l_label)
            orig_sent = orig_sentL.split("[used1]")[0]
            tree = CCTree(orig_sent, ll_label)
            tree.set_ccnodes()
            pred_ccnodes = tree.ccnodes

            pred_str = orig_sentL + '\n'
            ex_sents, spanned_words, l_spanned_locs = tree.get_ex_sents()
            l_spanned_words.append(spanned_words)
            ll_spanned_loc.append(l_spanned_locs)
            total_num_ex_sents1 += len(ex_sents)
            total_num_ex_sents2 += 1 if len(ex_sents) > 0 else 0
            pred_str += '\n'.join(ex_sents) + '\n'

            l_pred_str.append(pred_str)
        # list1 + list2 is the same as list1.extend(list2)
        self.cc_l_spanned_words+= l_spanned_words
        self.cc_l_pred_str += l_pred_str
        self.cc_ll_spanned_loc += ll_spanned_loc

        return l_pred_str

    def _write_output(self, output_d, batch_id, task):
        """
        similar to model.write_to_file()

        Parameters
        ----------
        output_d
        batch_id
        task

        Returns
        -------

        """
        if self.params_d["write_async"]:
            while not sem.acquire(blocking=True):
                # print("No Semaphore available")
                pass
            # print('Got semaphore')
        self.output.lll_prediction = self.output.lll_prediction.cpu()
        self.output.ll_score = self.output.ll_score.cpu()
        self.output.ground_truth = self.output.ground_truth.cpu()
        self.output.meta_data = self.output.meta_data.cpu()
        # note, right hand side depends on self.output.meta_data
        self.output.meta_data = [self.auto_tokenizer.decode[m] for m
                                 in self.output.meta_data]
        if task == "ex":
            l_pred_str, l_pred_allen_str = self._write_if_task_ex(output_d)
        elif task == "cc":
            l_pred_str = self._write_if_task_cc(output_d)
        else:
            assert False
        fpath = TASK + ".txt"
        if batch_id == 0:
            fmode= 'w'
        else:
            fmode = 'a'
        pred_f = open(fpath, fmode)
        pred_f.write('\n'.join(l_pred_str) + '\n')
        pred_f.close()
        if task == "ex" and self.params_d["write_allennlp"]:
            fpath = PREDICTIONS_DIR + "/allen.txt"
            if batch_id == 0:
                fmode = "w"
            else:
                fmode = "a"
            allen_f = open(fpath, fmode)
            allen_f.write('\n'.join(l_pred_allen_str) + '\n')
            allen_f.close()
        if self.params_d["write_async"]:
            sem.release()
