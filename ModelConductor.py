import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import shutil
from glob import glob
from time import time
from Model import *
from my_globals import *
from dict_utils import *


class ModelConductor:
    def __init__(self, task):
        self.self.model = None
        assert task in ["ex", "cc"]
        self.task = task
        self.saved = False
        self.save_dir = "data/save"
        self.has_cuda = torch.cuda.is_available()

        if task == 'cc':
            self.train_fp = 'data/openie-data/ptb-train.labels'
            self.dev_fp = 'data/openie-data/ptb-dev.labels'
            self.test_fp = 'data/openie-data/ptb-test.labels'
        elif task == 'ex':
            self.train_fp = 'data/openie-data/openie4_labels'
            self.dev_fp = 'data/carb-data/dev.txt'
            self.test_fp = 'data/carb-data/test.txt'

        train_dataset, val_dataset, test_dataset, \
            meta_data_vocab, all_sentences = data.process_data(params_d)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=params_d["batch_size"],
                                           collate_fn=data.pad_data,
                                           shuffle=True,
                                           num_workers=1)
        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=params_d["batch_size"],
                                         collate_fn=data.pad_data,
                                         num_workers=1)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=params_d["batch_size"],
                                          collate_fn=data.pad_data,
                                          num_workers=1)

    def set_checkpoint_callback(self):
        if self.saved:
            self.checkpoint_callback = ModelCheckpoint(
                filepath=self.save_dir + '/{epoch:02d}_{eval_acc:.3f}',
                verbose=True,
                monitor='eval_acc',
                mode='max',
                save_top_k=params_d["save_k"] if not params_d["debug"] else 0,
                period=0)
        else:
            self.checkpoint_callback = None

    def get_all_checkpoint_paths(self):
        return glob(self.save_dir + '/*.ckpt')

    def get_checkpoint_path(self):
        all_paths = glob(self.save_dir + '/*.ckpt')
        assert len(all_paths) == 1
        return all_paths[0]

    def get_logger(self, mode):
        log_dir = params_d["save"] + '/logs/'
        if os.path.exists(log_dir + f'{mode}'):
            mode_logs = list(glob(log_dir + f'/{mode}_*'))
            new_mode_index = len(mode_logs) + 1
            print('Moving old log to...')
            print(shutil.move(params_d["save"] + f'/logs/{mode}',
                              params_d[
                                  "save"] + f'/logs/{mode}_{new_mode_index}'))
        logger = TensorBoardLogger(
            save_dir=params_d["save"],
            name='logs',
            version=mode + '.part')
        return logger

    def get_trainer(self, logger, checkpoint_path=None):
        # trainer = Trainer(
        #     accumulate_grad_batches = int(params_d.accumulate_grad_batches),
        #     checkpoint_callback = self.checkpoint_callback,
        #     gpus = params_d.gpus,
        #     gradient_clip_val = params_d.gradient_clip_val,
        #     logger = logger,
        #     max_epochs = params_d.epochs,
        #     min_epochs = params_d.epochs,
        #     num_sanity_val_steps = params_d.num_sanity_val_steps,
        #     num_tpu_cores = params_d.num_tpu_cores,
        #     resume_from_checkpoint = checkpoint_path,
        #     show_progress_bar = True,
        #     track_grad_norm = params_d.track_grad_norm,
        #     train_percent_check = params_d.train_percent_check,
        #     use_tpu = params_d.use_tpu,
        #     val_check_interval = params_d.val_check_interval)

        trainer = Trainer(
            checkpoint_callback=self.checkpoint_callback,
            logger=logger,
            resume_from_checkpoint=checkpoint_path,
            show_progress_bar=True,
            **params_d)
        return trainer

    def update_params_d(self, checkpoint_path,
                       **final_changes_params_d):
        if self.has_cuda:
            loaded_params_d = torch.load(checkpoint_path)['params_d']
        else:
            loaded_params_d = torch.load(
                checkpoint_path,
                map_location=torch.device('cpu'))['params_d']

        update_dict(params_d, loaded_params_d)
        if final_changes_params_d:
            update_dict(params_d, final_changes_params_d)

    def train(self):
        self.set_checkpoint_callback()
        self.model = Model()
        logger = self.get_logger('train')
        trainer = self.get_trainer(logger)
        trainer.fit(self.model, train_dataloader=self.train_dataloader,
                    val_dataloaders=self.val_dataloader)
        shutil.move(params_d["save"] + f'/logs/train.part',
                    params_d["save"] + f'/logs/train')

    def resume(self, **final_changes_params_d):
        self.set_checkpoint_callback()
        checkpoint_path = self.get_checkpoint_path()
        self.update_params_d(checkpoint_path, **final_changes_params_d)
        self.model = Model()
        logger = self.get_logger('resume')
        trainer = self.get_trainer(logger, checkpoint_path)
        trainer.fit(self.model, train_dataloader=self.train_dataloader,
                    val_dataloaders=self.val_dataloader)
        shutil.move(params_d.save + f'/logs/resume.part',
                    params_d.save + f'/logs/resume')

    def test(self, train,
             mapping=None, conj_word_mapping=None,
             **final_changes_params_d):
        self.set_checkpoint_callback()
        all_checkpoint_paths = self.get_all_checkpoint_paths()
        checkpoint_path = all_checkpoint_paths[0]
        if not train:
            self.update_params_d(checkpoint_path,
                                **final_changes_params_d)

        self.model = Model()
        if mapping != None:
            self.model._metric.mapping = mapping
        if conj_word_mapping != None:
            self.model._metric.conj_word_mapping = conj_word_mapping

        logger = self.get_logger('test')
        test_f = open(params_d.save + '/logs/test.txt', 'w')

        for checkpoint_path in all_checkpoint_paths:
            trainer = Trainer(logger=logger,
                              gpus=params_d["gpus"],
                              resume_from_checkpoint=checkpoint_path)
            trainer.test(self.model, test_dataloaders=self.test_dataloader)
            result = self.model.results
            test_f.write(f'{checkpoint_path}\t{result}\n')
            test_f.flush()
        test_f.close()
        shutil.move(params_d.save + f'/logs/test.part',
                    params_d.save + f'/logs/test')

    def predict(self,
                mapping=None, conj_word_mapping=None,
                **final_changes_params_d):
        self.set_checkpoint_callback()

        # def predict(params_d, checkpoint_callback, meta_data_vocab,
        #             train_dataloader,
        #             val_dataloader, test_dataloader, all_sentences, mapping=None,
        #             conj_word_mapping=None):
        if params_d.task == 'conj':
            params_d.checkpoint = params_d.conj_model
        if params_d.task == 'oie':
            params_d.checkpoint = params_d.oie_model

        checkpoint_path = self.get_checkpoint_path()
        self.update_params_d(checkpoint_path, **final_changes_params_d)

        self.model = Model()

        if mapping != None:
            self.model._metric.mapping = mapping
        if conj_word_mapping != None:
            self.model._metric.conj_word_mapping = conj_word_mapping

        trainer = Trainer(gpus=params_d.gpus, logger=None,
                          resume_from_checkpoint=checkpoint_path)
        start_time = time()
        self.model.all_sentences = all_sentences
        trainer.test(self.model, test_dataloaders=self.test_dataloader)
        end_time = time()
        print(f'Total Time taken = {(end_time - start_time) / 60:2f} minutes')

    def splitpredict(self):
        self.set_checkpoint_callback()

        # def splitpredict(params_d, checkpoint_callback, meta_data_vocab,
        #                  train_dataloader, val_dataloader, test_dataloader,
        #                  all_sentences):
        mapping, conj_word_mapping = {}, {}
        params_d.write_allennlp = True
        if params_d.split_fp == '':
            params_d.task = 'conj'
            params_d.checkpoint = params_d.conj_model
            params_d.model_str = 'bert-base-cased'
            params_d.mode = 'predict'
            model = predict(params_d, None, meta_data_vocab, None, None,
                            test_dataloader, all_sentences)
            conj_predictions = model.all_cc_predictions
            sentences_indices = model.all_cc_sent_locs
            # conj_predictions = model.predictions
            # sentences_indices = model.all_sentence_indices
            assert len(conj_predictions) == len(sentences_indices)
            all_conj_words = model.all_cc_words

            sentences, orig_sentences = [], []
            for i, sentences_str in enumerate(conj_predictions):
                list_sentences = sentences_str.strip('\n').split('\n')
                conj_words = all_conj_words[i]
                if len(list_sentences) == 1:
                    orig_sentences.append(
                        list_sentences[0] + ' [unused1] [unused2] [unused3]')
                    mapping[list_sentences[0]] = list_sentences[0]
                    conj_word_mapping[list_sentences[0]] = conj_words
                    sentences.append(
                        list_sentences[0] + ' [unused1] [unused2] [unused3]')
                elif len(list_sentences) > 1:
                    orig_sentences.append(
                        list_sentences[0] + ' [unused1] [unused2] [unused3]')
                    conj_word_mapping[list_sentences[0]] = conj_words
                    for sent in list_sentences[1:]:
                        mapping[sent] = list_sentences[0]
                        sentences.append(
                            sent + ' [unused1] [unused2] [unused3]')
                else:
                    assert False
            sentences.append('\n')

            count = 0
            for sentence_indices in sentences_indices:
                if len(sentence_indices) == 0:
                    count += 1
                else:
                    count += len(sentence_indices)
            assert count == len(sentences) - 1

        else:
            with open(params_d.predict_fp, 'r') as f:
                lines = f.read()
                lines = lines.replace("\\", "")

            sentences = []
            orig_sentences = []
            extra_str = " [unused1] [unused2] [unused3]"
            for line in lines.split('\n\n'):
                if len(line) > 0:
                    list_sentences = line.strip().split('\n')
                    if len(list_sentences) == 1:
                        mapping[list_sentences[0]] = list_sentences[0]
                        sentences.append(list_sentences[0] + extra_str)
                        orig_sentences.append(list_sentences[0] + extra_str)
                    elif len(list_sentences) > 1:
                        orig_sentences.append(list_sentences[0] + extra_str)
                        for sent in list_sentences[1:]:
                            mapping[sent] = list_sentences[0]
                            sentences.append(sent + extra_str)
                    else:
                        assert False

        params_d.task = 'oie'
        params_d.checkpoint = params_d.oie_model
        params_d.model_str = 'bert-base-cased'
        _, _, split_test_dataset, meta_data_vocab, _ = data.process_data(
            params_d,
            sentences)
        split_test_dataloader = DataLoader(split_test_dataset,
                                           batch_size=params_d.batch_size,
                                           collate_fn=data.pad_data,
                                           num_workers=1)

        model = self.predict(params_d, None, meta_data_vocab, None, None,
                             split_test_dataloader,
                             mapping=mapping,
                             conj_word_mapping=conj_word_mapping,
                             all_sentences=all_sentences)

        if 'labels' in params_d.type:
            label_lines = get_labels(params_d, model, sentences, orig_sentences,
                                     sentences_indices)
            f = open(params_d.out + '.labels', 'w')
            f.write('\n'.join(label_lines))
            f.close()

        if params_d.rescoring:
            print()
            print("Starting re-scoring ...")
            print()

            sentence_line_nums, prev_line_num, no_extractions = set(), 0, dict()
            curr_line_num = 0
            for sentence_str in model.all_predictions_oie:
                sentence_str = sentence_str.strip('\n')
                num_extrs = len(sentence_str.split('\n')) - 1
                if num_extrs == 0:
                    if curr_line_num not in no_extractions:
                        no_extractions[curr_line_num] = []
                    no_extractions[curr_line_num].append(sentence_str)
                    continue
                curr_line_num = prev_line_num + num_extrs
                sentence_line_nums.add(
                    curr_line_num)  # check extra empty lines, example with no extractions
                prev_line_num = curr_line_num

            # testing rescoring
            inp_fp = model.predictions_f_allennlp
            rescored = rescore(inp_fp, model_dir=params_d.rescore_model,
                               batch_size=256)

            all_predictions, sentence_str = [], ''
            for line_i, line in enumerate(rescored):
                fields = line.split('\t')
                sentence = fields[0]
                confidence = float(fields[2])

                if line_i == 0:
                    sentence_str = f'{sentence}\n'
                    exts = []
                if line_i in sentence_line_nums:
                    exts = sorted(exts, reverse=True,
                                  key=lambda x: float(x.split()[0][:-1]))
                    exts = exts[:params_d.num_extractions]
                    all_predictions.append(sentence_str + ''.join(exts))
                    sentence_str = f'{sentence}\n'
                    exts = []
                if line_i in no_extractions:
                    for no_extraction_sentence in no_extractions[line_i]:
                        all_predictions.append(f'{no_extraction_sentence}\n')

                arg1 = re.findall("<arg1>.*</arg1>", fields[1])[0].strip(
                    '<arg1>').strip('</arg1>').strip()
                rel = re.findall("<rel>.*</rel>", fields[1])[0].strip(
                    '<rel>').strip('</rel>').strip()
                arg2 = re.findall("<arg2>.*</arg2>", fields[1])[0].strip(
                    '<arg2>').strip('</arg2>').strip()
                extraction = Extraction(pred=rel, head_pred_index=None,
                                        sent=sentence,
                                        confidence=math.exp(confidence),
                                        index=0)
                extraction.addArg(arg1)
                extraction.addArg(arg2)
                if params_d.type == 'sentences':
                    ext_str = data.ext_to_sentence(extraction) + '\n'
                else:
                    ext_str = data.ext_to_string(extraction) + '\n'
                exts.append(ext_str)

            exts = sorted(exts, reverse=True,
                          key=lambda x: float(x.split()[0][:-1]))
            exts = exts[:params_d.num_extractions]
            all_predictions.append(sentence_str + ''.join(exts))

            if line_i + 1 in no_extractions:
                for no_extraction_sentence in no_extractions[line_i + 1]:
                    all_predictions.append(f'{no_extraction_sentence}\n')

            if params_d.out != None:
                print('Predictions written to ', params_d.out)
                predictions_f = open(params_d.out, 'w')
            predictions_f.write('\n'.join(all_predictions) + '\n')
            predictions_f.close()
            return
