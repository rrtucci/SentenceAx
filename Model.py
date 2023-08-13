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
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
"""
from Extraction_sax import *
from ExMetric import *
from CCMetric import *
from CCTree import *

import os
import copy
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

    def __init__(self, auto_tokenizer):
        super(Model, self).__init__()
        self.params_d = PARAMS_D
        self.auto_tokenizer = auto_tokenizer

        self.base_model = AutoModel.from_pretrained(
            self.params_d["model_str"], cache_dir=CACHE_DIR)
        self._hidden_size = self.base_model.config.hidden_size

        if self.params_d["iterative_layers"] != 0:
            num_layers = len(self.base_model.encoder.layer)
            mid = num_layers - self.params_d["iterative_layers"]
            self.base_model.encoder.layer = \
                self.base_model.encoder.layer[0:mid]
            self.iterative_transformer = \
                self.base_model.encoder.layer[mid:num_layers]

        else:
            self.iterative_transformer = []

        self.num_labels = NUM_LABELS
        self.labelling_dim = self.params_d["labelling_dim"]
        self.dropout = nn.Dropout(p=self.params_d["dropout"])

        """
        nn.Embedding(num_embeddings, embedding_size)
        num_embeddings (int) – size of the dictionary of embeddings
        embedding_dim (int) – the size of each embedding vector
            
        """
        self.label_embeddings = nn.Embedding(100,
                                             self._hidden_size)
        self.merge_layer = nn.Linear(self._hidden_size,
                                     self.labelling_dim)
        self.labelling_layer = nn.Linear(self.labelling_dim,
                                         self.num_labels)

        self.loss = nn.CrossEntropyLoss()

        self.metric = ExMetric(self.params_d) \
            if self.params_d["task"] == "ex" else CCMetric()

        self.constraints_d = dict()

        self.all_cc_predictions = []
        self.all_cc_sent_locs = []
        self.all_cc_words = []
        self.all_ex_predictions = []

    def configure_optimizers(self):
        # self.named_parameters() is Iterator[Tuple[str, Parameter]]
        all_params = list(self.named_parameters())
        # opt= optimizer
        # p = parameter
        opt_p_names = ["bias", "gamma", "beta"]

        def p_in_overlap_of_lists(p, li1, li2):
            return any(p in li1 for p in li2)

        def p_not_in_overlap_of_lists(p, li1, li2):
            return not any(p in li1 for p in li2)

        opt_params = [
            {"params": [p for p_names, p in all_params if
                        p_not_in_overlap_of_lists(p, p_names, opt_p_names) and
                        'base_model' in p_names],
             "weight_decay_rate": 0.01,
             'lr': self.params_d["lr"]},
            {"params": [p for p_names, p in all_params if
                        p_in_overlap_of_lists(p, p_names, opt_p_names) and
                        'base_model' in p_names],
             "weight_decay_rate": 0.0,
             'lr': self.params_d["lr"]},
            {"params": [p for p_names, p in all_params if
                        'base_model' not in p_names],
             'lr': self.params_d["lr"]}
        ]
        if self.params_d["optimizer"] == 'adamW':
            optimizer = AdamW(opt_params, lr=1e-3)
        elif self.params_d["optimizer"] == 'adam':
            optimizer = Adam(opt_params, lr=1e-3)
        else:
            assert False

        if self.params_d["multi_opt"] and \
                self.params_d["constraints"] != None:
            num_optimizers = len(self.params_d["constraints"].split('_'))
            return [optimizer] * num_optimizers
        else:
            return [optimizer]

    def forward(self, batch_d, mode='train', batch_idx=-1,
                constraints=None, cweights=None):
        # signature of parent method:  def forward(self, *args, **kwargs):
        if self.params_d["wreg"] != 0 and \
                not hasattr(self, 'init_params_d'):
            self.init_params_d = copy.deepcopy(
                dict(self.named_parameters()))

        # remember: labels = li_li_li_label
        # first (outer) list over batch events
        # second list over extractions
        # third (inner) list over number of ilabels in a line
        batch_size, depth, labels_length = batch_d["labels"].shape
        if mode != 'train':
            depth = MAX_DEPTH

        hidden_states, _ = self.base_model(batch_d["text"])
        output_d = dict()
        all_depth_scores = []

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
                word_hidden_states = word_hidden_states + label_embeddings

            word_hidden_states = self.merge_layer(word_hidden_states)
            word_scores = self.labelling_layer(word_hidden_states)
            all_depth_scores.append(word_scores)

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
        return self.fill_forward_output_d(
            batch_d, mode,
            constraints, cweights,
            all_depth_scores, word_scores, output_d)

    def fill_forward_output_d(
            self, batch_d, mode,
            constraints, cweights,
            all_depth_scores, word_scores, output_d):

        loss, lstm_loss = 0, 0
        all_depth_predictions, all_depth_confidences = [], []
        batch_size, num_words, _ = word_scores.shape
        batch_d["labels"] = batch_d["labels"].long()
        for d, word_scores in enumerate(all_depth_scores):
            if mode == 'train':
                # batch_labels_d = batch_d["labels"][:, d, :]
                # mask = torch.ones(batch_d["word_starts"].shape).int(). \
                #     type_as(hidden_states)
                loss += self.loss(
                    word_scores.reshape(batch_size * num_words, -1),
                    batch_d["labels"][:, d, :].reshape(-1))
            else:
                word_log_probs = torch.log_softmax(word_scores, dim=2)
                max_log_probs, predictions = \
                    torch.max(word_log_probs, dim=2)
                # remember: labels = li_li_li_label
                # first (outer) list over batch events
                # second list over extractions
                # third (inner) list over number of ilabels in a line
                padding_labels = (batch_d["labels"][:, 0, :] != -100).float()

                sro_label_predictions = \
                    (predictions != 0).float() * padding_labels
                log_probs_norm_ext_len = \
                    (max_log_probs * sro_label_predictions) \
                    / (sro_label_predictions.sum(dim=0) + 1)
                confidences = torch.exp(
                    torch.sum(log_probs_norm_ext_len, dim=1))

                all_depth_predictions.append(predictions.unsqueeze(1))
                all_depth_confidences.append(confidences.unsqueeze(1))

        if mode == 'train':
            if constraints != '':
                all_depth_scores = torch.cat([d.unsqueeze(1) for
                                              d in all_depth_scores], dim=1)
                all_depth_scores = torch.softmax(all_depth_scores, dim=-1)

                const_loss = self.constrained_loss(
                    all_depth_scores, batch_d,
                    constraints, cweights) / batch_size
                loss = const_loss

            if self.params_d["wreg"] != 0:
                weight_diff = 0
                current_parameters = dict(self.named_parameters())
                for name in self.init_params_d:
                    weight_diff += torch.norm(current_parameters[name]
                                              - self.init_params_d[name])
                loss = loss + self.params_d["wreg"] * weight_diff
        else:  # not train

            all_depth_predictions = torch.cat(all_depth_predictions, dim=1)
            all_depth_confidences = torch.cat(all_depth_confidences, dim=1)

            output_d['predictions'] = all_depth_predictions
            output_d['scores'] = all_depth_confidences

            if constraints != '' and \
                    'predict' not in self.params_d["mode"] and \
                    self.params_d["batch_size"] != 1:
                all_depth_scores = torch.cat([d.unsqueeze(1) for
                                              d in all_depth_scores], dim=1)
                all_depth_scores.fill_(0)

                # for checking test set
                # labels = copy.copy(batch_d["labels"])
                # labels[labels == -100] = 0
                labels = copy.copy(all_depth_predictions)

                labels = labels.unsqueeze(-1)
                labels_depth = labels.shape[1]
                all_depth_scores = all_depth_scores[:, :labels_depth, :, :]
                all_depth_scores.scatter_(3, labels.long(), 1)

                constraints, cweights = 'posm_hvc_hvr_hve', '1_1_1_1'
                constraints_list, cweights_list = \
                    constraints.split('_'), cweights.split('_')
                if len(constraints_list) != len(cweights_list):
                    cweights_list = [cweights] * len(constraints_list)

                for constraint, weight in \
                        zip(constraints_list, cweights_list):
                    const_loss = self.constrained_loss(all_depth_scores,
                                                       batch_d, constraint,
                                                       float(weight))
                    if constraint not in self.constraints_d:
                        self.constraints_d[constraint] = []
                    self.constraints_d[constraint].append(const_loss)

        output_d['loss'] = loss
        return output_d

    def constrained_loss(self, all_depth_scores, batch_d,
                         constraints, cweights):
        batch_size, depth, num_words, labels = all_depth_scores.shape
        hinge_loss = 0
        bat = batch_d["verb_index"].unsqueeze(1).unsqueeze(3). \
            repeat(1, depth, 1, labels)
        verb_scores = torch.gather(all_depth_scores, 2, bat)
        verb_rel_scores = verb_scores[:, :, :, 2]
        # (batch_size, depth, num_words)
        verb_rel_scores = verb_rel_scores * (batch_d["verb_index"] != 0). \
            unsqueeze(1).float()

        # every head-verb must be included in a relation
        if 'hvc' in constraints:
            column_loss = torch.abs(1 - torch.sum(verb_rel_scores, dim=1))
            column_loss = column_loss[batch_d["verb_index"] != 0]
            hinge_loss += cweights * column_loss.sum()

        # extractions must have at least k-relations with
        # a head verb in them
        if 'hvr' in constraints:
            row_rel_loss = F.relu(batch_d["verb"].sum(dim=1).float() -
                                  torch.max(verb_rel_scores, dim=2)[0].sum(
                                      dim=1))
            hinge_loss += cweights * row_rel_loss.sum()

        # one relation cannot contain more than one head verb
        if 'hve' in constraints:
            ex_loss = F.relu(torch.sum(verb_rel_scores, dim=2) - 1)
            hinge_loss += cweights * ex_loss.sum()

        if 'posm' in constraints:
            bat = batch_d["pos_index"].unsqueeze(1).unsqueeze(3). \
                repeat(1, depth, 1, labels)
            pos_scores = torch.gather(all_depth_scores, 2, bat)
            pos_nnone_scores = \
                torch.max(pos_scores[:, :, :, 1:], dim=-1)[0]
            column_loss = (1 - torch.max(pos_nnone_scores, dim=1)[0]) * \
                          (batch_d["pos_index"] != 0).float()
            hinge_loss += cweights * column_loss.sum()

        return hinge_loss

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if \
            running_train_loss is not None else float('NaN')
        if type(self.trainer.checkpoint_callback.kth_value) != type(0.0):
            best = self.trainer.checkpoint_callback.kth_value.item()
        else:
            best = self.trainer.checkpoint_callback.kth_value
        tqdm = {'loss': '{:.3f}'.format(avg_training_loss), 'best': best}
        return tqdm

    def training_step(self, batch_d, batch_idx, optimizer_idx=-1):
        if self.params_d["multi_opt"]:
            constraints = self.params_d["constraints"].split('_')[
                optimizer_idx]
            cweights = float(
                self.params_d["cweights"].split('_')[optimizer_idx])
        else:
            constraints = self.params_d["constraints"]
            cweights = float(self.params_d["cweights"])

        output_d = self.forward(batch_d, mode='train',
                                   batch_idx=batch_idx,
                                   constraints=constraints,
                                   cweights=cweights)

        tqdm_dict = {"train_loss": output_d['loss']}
        output0_d = OrderedDict({"loss": output_d['loss'], "log": tqdm_dict})

        return output0_d

    def validation_step(self, batch_d, batch_idx):
        output_d = self.forward(
            batch_d,
            mode='val',
            constraints=self.params_d["constraints"],
            cweights=self.params_d["cweights"])

        output0_d = {"predictions": output_d['predictions'],
                   "scores": output_d['scores'],
                   "ground_truth": batch_d["labels"],
                   "meta_data": batch_d["meta_data"]}
        output0_d = OrderedDict(output0_d)

        if self.params_d["mode"] != 'test':
            if self.params_d["write_async"]:
                t = Thread(target=self.write_to_file,
                           args=(output0_d, batch_idx, self.params_d["task"]))
                t.start()
            else:
                self.write_to_file(output0_d, batch_idx, self.params_d["task"])

        return output0_d

    def test_step(self, batch_d, batch_idx):
        return self.validation_step(batch_d, batch_idx)

    def evaluation_end(self, output_d_list, mode):
        result = None
        if self.params_d["mode"] == 'test':
            for output_index, output_d in enumerate(output_d_list):
                output_d['predictions'] = output_d['predictions'].cpu()
                output_d['scores'] = output_d['scores'].cpu()
                output_d['scores'] = (output_d['scores'] * 100).round() / 100
                output_d['ground_truth'] = output_d['ground_truth'].cpu()
                output_d['meta_data'] = output_d['meta_data'].cpu()
        if self.params_d["task"] == 'conj':
            if 'predict' in self.params_d["mode"]:
                metrics = {'P_exact': 0, 'R_exact': 0, 'F1_exact': 0}
            else:
                for output_d in output_d_list:
                    if type(output_d['meta_data'][0]) != type(""):
                        output_d['meta_data'] = [self.auto_tokenizer.decode[m]
                                               for m in output_d['meta_data']]
                    self.metric(output_d['predictions'],
                                 output_d['ground_truth'],
                                 meta_data=output_d['meta_data'])
                metrics = self.metric.get_metric(reset=True, mode=mode)

            val_acc, val_auc = metrics['F1_exact'], 0
            result = {"eval_f1": val_acc, "eval_p": metrics['P_exact'],
                      "eval_r": metrics['R_exact']}

        elif self.params_d["task"] == "ex":
            if 'predict' in self.params_d["mode"]:
                metrics = {'carb_f1': 0, 'carb_auc': 0, 'carb_lastf1': 0}
            else:
                for output_d in output_d_list:
                    if type(output_d['meta_data'][0]) != type(""):
                        output_d['meta_data'] = [self.auto_tokenizer.decode[m]
                                               for m in output_d['meta_data']]
                    self.metric(output_d['predictions'], output_d['meta_data'],
                                 output_d['scores'])
                metrics = self.metric.get_metric(reset=True, mode=mode)

            result = {"eval_f1": metrics['carb_f1'],
                      "eval_auc": metrics['carb_auc'],
                      "eval_lastf1": metrics['carb_lastf1']}

        print('\nResults: ' + str(result))
        # For computing the constraint violations
        # if hasattr(self, 'constraints_d') and \
        # self.params_d["constraints"] != '':
        #     for key in self.constraints_d:
        #         self.constraints_d[key] = sum(self.constraints_d[key]).item()
        #     print('\nViolations: ', self.constraints_d)
        #     self.constraints_d = dict()
        return result

    def validation_epoch_end(self, output_d_list):
        eval_results = self.evaluation_end(output_d_list, 'dev')
        result = {}
        if eval_results != None:
            result = {"log": eval_results,
                      "eval_acc": eval_results['eval_f1']}

        return result

    def test_epoch_end(self, output_d_list):
        eval_results = self.evaluation_end(output_d_list, 'test')
        self.output_d_list = output_d_list
        result = {"log": eval_results,
                  "progress_bar": eval_results,
                  "test_acc": eval_results['eval_f1']}
        self.results = eval_results
        if self.params_d["write_async"]:
            while not sem.acquire(blocking=True):
                pass
            sem.release()
        return result

    # obligatory definitions - pass actual through fit
    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def process_extraction(self, extraction, sentence, score):
        # rel, arg1, arg2, loc, time = [], [], [], [], []
        rel, arg1, arg2, loc_time, args = [], [], [], [], []
        tag_mode = 'none'
        rel_case = 0
        for i, token in enumerate(sentence):
            if '[unused' in token:
                if extraction[i].item() == 2:
                    rel_case = int(
                        re.search('\[unused(.*)\]', token).group(1))
                continue
            if extraction[i] == 1:
                arg1.append(token)
            if extraction[i] == 2:
                rel.append(token)
            if extraction[i] == 3:
                arg2.append(token)
            if extraction[i] == 4:
                loc_time.append(token)

        rel = ' '.join(rel).strip()
        if rel_case == 1:
            rel = 'is ' + rel
        elif rel_case == 2:
            rel = 'is ' + rel + ' of'
        elif rel_case == 3:
            rel = 'is ' + rel + ' from'

        arg1 = ' '.join(arg1).strip()
        arg2 = ' '.join(arg2).strip()
        args = ' '.join(args).strip()
        loc_time = ' '.join(loc_time).strip()
        if not self.params_d["no_lt"]:
            arg2 = (arg2 + ' ' + loc_time + ' ' + args).strip()
        sentence_str = ' '.join(sentence).strip()

        extraction = Extraction_sax(sentence_str,
                                    arg1,
                                    rel,
                                    arg2,
                                    confidence=score)
        extraction.add_arg1(arg1)
        extraction.add_arg2(arg2)

        return extraction

    def write_to_file(self, output_d, batch_idx, task):
        if self.params_d["write_async"]:
            while not sem.acquire(blocking=True):
                # print("No Semaphore available")
                pass
            # print('Got semaphore')
        output_d['predictions'] = output_d['predictions'].cpu()
        output_d['scores'] = output_d['scores'].cpu()
        output_d['ground_truth'] = output_d['ground_truth'].cpu()
        output_d['meta_data'] = output_d['meta_data'].cpu()
        output_d['meta_data'] = [self.auto_tokenizer.decode[m] for m
                               in output_d['meta_data']]
        if task == "ex":
            predictions = output_d['predictions']
            sentences = output_d['meta_data']
            scores = output_d['scores']
            num_sentences, extractions, max_sentence_len = predictions.shape
            assert num_sentences == len(sentences)
            all_predictions = {}
            for i, sentence_str in enumerate(sentences):
                words = sentence_str.split() + \
                        ['[unused1]', '[unused2]', '[unused3]']
                orig_sentence = sentence_str.split('[unused1]')[0].strip()
                if self.metric.mapping:
                    if self.metric.mapping[
                        orig_sentence] not in all_predictions:
                        all_predictions[
                            self.metric.mapping[orig_sentence]] = []
                else:
                    if orig_sentence not in all_predictions:
                        all_predictions[orig_sentence] = []
                for j in range(extractions):
                    extraction = predictions[i][j][:len(words)]
                    if sum(extraction) == 0:  # extractions completed
                        break
                    pro_extraction = self.process_extraction(
                        extraction, words, scores[i][j].item())
                    if pro_extraction.arg1_pair[1] != '' and \
                            pro_extraction.rel_pair[1] != '':
                        if self.metric.mapping:
                            if not pro_extraction.get_str() in \
                                   all_predictions[
                                    self.metric.mapping[orig_sentence]]:
                                all_predictions[self.metric.mapping[
                                    orig_sentence]].append(pro_extraction)
                        else:
                            if not pro_extraction.get_str() in \
                                   all_predictions[orig_sentence]:
                                all_predictions[orig_sentence].append(
                                    pro_extraction)
            all_pred = []
            all_pred_allennlp = []
            for example_id, sentence in enumerate(all_predictions):
                predicted_extractions = all_predictions[sentence]
                # write only the results in text file
                # if 'predict' in self.params_d["mode"]:
                sentence_str = f'{sentence}\n'
                for extraction in predicted_extractions:
                    if self.params_d["type"] == 'sentences':
                        ext_str = extraction.get_str() + '\n'
                    else:
                        ext_str = extraction.get_str() + '\n'
                    sentence_str += ext_str
                all_pred.append(sentence_str)
                sentence_str_allennlp = ''
                for extraction in predicted_extractions:
                    arg1 = extraction.arg1
                    ext_str = \
                        f'{sentence}\t<arg1> {extraction.arg1} </arg1> ' \
                        f'<rel> {extraction.pred} </rel> ' \
                        f'<arg2> {arg1} </arg2>\t{extraction.confidence}\n'
                    sentence_str_allennlp += ext_str
                    sentence_str_allennlp.strip('\n')
                all_pred_allennlp.append(sentence_str_allennlp)
            self.all_ex_predictions.extend(all_pred)
        if task == 'conj':
            example_id, correct = 0, True
            total1, total2 = 0, 0
            predictions = output_d['predictions']
            gt = output_d['ground_truth']
            meta_data = output_d['meta_data']
            total_depth = predictions.shape[1]
            all_pred = []
            all_conjunct_words = []
            all_sentence_indices = []
            for idx in range(len(meta_data)):
                example_id += 1
                sentence = meta_data[idx]
                words = sentence.split()
                sentence_predictions, sentence_gt = [], []
                for depth in range(total_depth):
                    depth_predictions = predictions[idx][depth][:len(
                        words)].tolist()
                    sentence_predictions.append(depth_predictions)
                pred_ccnodes = self.metric.get_ccnodes(sentence_predictions)

                words = sentence.split()
                sentence_str = sentence + '\n'
                tree = CCTree(simple_sent, depth_predictions)
                split_sentences, conj_words, sentence_indices_list = \
                    tree.get_simple_sents()
                all_sentence_indices.append(sentence_indices_list)
                all_conjunct_words.append(conj_words)
                total1 += len(split_sentences)
                total2 += 1 if len(split_sentences) > 0 else 0
                sentence_str += '\n'.join(split_sentences) + '\n'

                all_pred.append(sentence_str)
            self.all_cc_words.extend(all_conjunct_words)
            self.all_cc_predictions.extend(all_pred)
            self.all_cc_sent_locs.extend(all_sentence_indices)
        if self.params_d["out"] != None:
            directory = os.path.dirname(self.params_d["out"])
            if directory != '' and not os.path.exists(directory):
                os.makedirs(directory)
            out_fp = f'{self.params_d["out"]}.{self.params_d["task"]}'
            # print('Predictions written to ', out_fp)
            if batch_idx == 0:
                predictions_f = open(out_fp, 'w')
            else:
                predictions_f = open(out_fp, 'a')
            predictions_f.write('\n'.join(all_pred) + '\n')
            predictions_f.close()
        if task == "ex" and self.params_d["write_allennlp"]:
            if batch_idx == 0:
                predictions_f_allennlp = open(
                    f'{self.params_d["out"]}.allennlp',
                    'w')
                self.predictions_f_allennlp = predictions_f_allennlp.name
            else:
                predictions_f_allennlp = open(
                    f'{self.params_d["out"]}.allennlp',
                    'a')
            predictions_f_allennlp.write(''.join(all_pred_allennlp))
            predictions_f_allennlp.close()
        if self.params_d["write_async"]:
            sem.release()
