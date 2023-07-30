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
            meta_data_vocab, all_sentences = data.process_data(hparams)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=hparams["batch_size"],
                                           collate_fn=data.pad_data,
                                           shuffle=True,
                                           num_workers=1)
        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=hparams["batch_size"],
                                         collate_fn=data.pad_data,
                                         num_workers=1)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=hparams["batch_size"],
                                          collate_fn=data.pad_data,
                                          num_workers=1)

    def set_checkpoint_callback(self):
        if self.saved:
            self.checkpoint_callback = ModelCheckpoint(
                filepath=self.save_dir + '/{epoch:02d}_{eval_acc:.3f}',
                verbose=True,
                monitor='eval_acc',
                mode='max',
                save_top_k=hparams["save_k"] if not hparams["debug"] else 0,
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
        log_dir = hparams["save"] + '/logs/'
        if os.path.exists(log_dir + f'{mode}'):
            mode_logs = list(glob(log_dir + f'/{mode}_*'))
            new_mode_index = len(mode_logs) + 1
            print('Moving old log to...')
            print(shutil.move(hparams["save"] + f'/logs/{mode}',
                              hparams[
                                  "save"] + f'/logs/{mode}_{new_mode_index}'))
        logger = TensorBoardLogger(
            save_dir=hparams["save"],
            name='logs',
            version=mode + '.part')
        return logger

    def get_trainer(self, logger, checkpoint_path=None):
        # trainer = Trainer(
        #     accumulate_grad_batches = int(hparams.accumulate_grad_batches),
        #     checkpoint_callback = self.checkpoint_callback,
        #     gpus = hparams.gpus,
        #     gradient_clip_val = hparams.gradient_clip_val,
        #     logger = logger,
        #     max_epochs = hparams.epochs,
        #     min_epochs = hparams.epochs,
        #     num_sanity_val_steps = hparams.num_sanity_val_steps,
        #     num_tpu_cores = hparams.num_tpu_cores,
        #     resume_from_checkpoint = checkpoint_path,
        #     show_progress_bar = True,
        #     track_grad_norm = hparams.track_grad_norm,
        #     train_percent_check = hparams.train_percent_check,
        #     use_tpu = hparams.use_tpu,
        #     val_check_interval = hparams.val_check_interval)

        trainer = Trainer(
            checkpoint_callback=self.checkpoint_callback,
            logger=logger,
            resume_from_checkpoint=checkpoint_path,
            show_progress_bar=True,
            **hparams)
        return trainer

    def update_hparams(self, checkpoint_path,
                       **final_changes_hparams):
        if self.has_cuda:
            loaded_hparams = torch.load(checkpoint_path)['hparams']
        else:
            loaded_hparams = torch.load(
                checkpoint_path,
                map_location=torch.device('cpu'))['hparams']

        update_dict(hparams, loaded_hparams)
        if final_changes_hparams:
            update_dict(hparams, final_changes_hparams)

    def train(self):
        self.set_checkpoint_callback()
        self.model = Model()
        logger = self.get_logger('train')
        trainer = self.get_trainer(logger)
        trainer.fit(self.model, train_dataloader=self.train_dataloader,
                    val_dataloaders=self.val_dataloader)
        shutil.move(hparams["save"] + f'/logs/train.part',
                    hparams["save"] + f'/logs/train')

    def resume(self, **final_changes_hparams):
        self.set_checkpoint_callback()
        checkpoint_path = self.get_checkpoint_path()
        self.update_hparams(checkpoint_path, **final_changes_hparams)
        self.model = Model()
        logger = self.get_logger('resume')
        trainer = self.get_trainer(logger, checkpoint_path)
        trainer.fit(self.model, train_dataloader=self.train_dataloader,
                    val_dataloaders=self.val_dataloader)
        shutil.move(hparams.save + f'/logs/resume.part',
                    hparams.save + f'/logs/resume')

    def test(self, train,
             mapping=None, conj_word_mapping=None,
             **final_changes_hparams):
        self.set_checkpoint_callback()
        all_checkpoint_paths = self.get_all_checkpoint_paths()
        checkpoint_path = all_checkpoint_paths[0]
        if not train:
            self.update_hparams(checkpoint_path,
                                **final_changes_hparams)

        self.model = Model()
        if mapping != None:
            self.model._metric.mapping = mapping
        if conj_word_mapping != None:
            self.model._metric.conj_word_mapping = conj_word_mapping

        logger = self.get_logger('test')
        test_f = open(hparams.save + '/logs/test.txt', 'w')

        for checkpoint_path in all_checkpoint_paths:
            trainer = Trainer(logger=logger,
                              gpus=hparams["gpus"],
                              resume_from_checkpoint=checkpoint_path)
            trainer.test(self.model, test_dataloaders=self.test_dataloader)
            result = self.model.results
            test_f.write(f'{checkpoint_path}\t{result}\n')
            test_f.flush()
        test_f.close()
        shutil.move(hparams.save + f'/logs/test.part',
                    hparams.save + f'/logs/test')

    def predict(self,
                mapping=None, conj_word_mapping=None,
                **final_changes_hparams):
        self.set_checkpoint_callback()

        # def predict(hparams, checkpoint_callback, meta_data_vocab,
        #             train_dataloader,
        #             val_dataloader, test_dataloader, all_sentences, mapping=None,
        #             conj_word_mapping=None):
        if hparams.task == 'conj':
            hparams.checkpoint = hparams.conj_model
        if hparams.task == 'oie':
            hparams.checkpoint = hparams.oie_model

        checkpoint_path = self.get_checkpoint_path()
        self.update_hparams(checkpoint_path, **final_changes_hparams)

        self.model = Model()

        if mapping != None:
            self.model._metric.mapping = mapping
        if conj_word_mapping != None:
            self.model._metric.conj_word_mapping = conj_word_mapping

        trainer = Trainer(gpus=hparams.gpus, logger=None,
                          resume_from_checkpoint=checkpoint_path)
        start_time = time()
        self.model.all_sentences = all_sentences
        trainer.test(self.model, test_dataloaders=self.test_dataloader)
        end_time = time()
        print(f'Total Time taken = {(end_time - start_time) / 60:2f} minutes')

    def splitpredict(self):
        self.set_checkpoint_callback()

        # def splitpredict(hparams, checkpoint_callback, meta_data_vocab,
        #                  train_dataloader, val_dataloader, test_dataloader,
        #                  all_sentences):
        mapping, conj_word_mapping = {}, {}
        hparams.write_allennlp = True
        if hparams.split_fp == '':
            hparams.task = 'conj'
            hparams.checkpoint = hparams.conj_model
            hparams.model_str = 'bert-base-cased'
            hparams.mode = 'predict'
            model = predict(hparams, None, meta_data_vocab, None, None,
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
            with open(hparams.predict_fp, 'r') as f:
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

        hparams.task = 'oie'
        hparams.checkpoint = hparams.oie_model
        hparams.model_str = 'bert-base-cased'
        _, _, split_test_dataset, meta_data_vocab, _ = data.process_data(
            hparams,
            sentences)
        split_test_dataloader = DataLoader(split_test_dataset,
                                           batch_size=hparams.batch_size,
                                           collate_fn=data.pad_data,
                                           num_workers=1)

        model = self.predict(hparams, None, meta_data_vocab, None, None,
                             split_test_dataloader,
                             mapping=mapping,
                             conj_word_mapping=conj_word_mapping,
                             all_sentences=all_sentences)

        if 'labels' in hparams.type:
            label_lines = get_labels(hparams, model, sentences, orig_sentences,
                                     sentences_indices)
            f = open(hparams.out + '.labels', 'w')
            f.write('\n'.join(label_lines))
            f.close()

        if hparams.rescoring:
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
            rescored = rescore(inp_fp, model_dir=hparams.rescore_model,
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
                    exts = exts[:hparams.num_extractions]
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
                if hparams.type == 'sentences':
                    ext_str = data.ext_to_sentence(extraction) + '\n'
                else:
                    ext_str = data.ext_to_string(extraction) + '\n'
                exts.append(ext_str)

            exts = sorted(exts, reverse=True,
                          key=lambda x: float(x.split()[0][:-1]))
            exts = exts[:hparams.num_extractions]
            all_predictions.append(sentence_str + ''.join(exts))

            if line_i + 1 in no_extractions:
                for no_extraction_sentence in no_extractions[line_i + 1]:
                    all_predictions.append(f'{no_extraction_sentence}\n')

            if hparams.out != None:
                print('Predictions written to ', hparams.out)
                predictions_f = open(hparams.out, 'w')
            predictions_f.write('\n'.join(all_predictions) + '\n')
            predictions_f.close()
            return
