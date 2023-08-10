"""

Torch lightning

Dataset stores the samples and their corresponding labels, and DataLoader
wraps an iterable around the Dataset to enable easy access to the samples.

DataLoader is located in torch.utils.data

loader = torch.utils.dataloader(dset)
for input, target in loader:
     output = model(input)
     loss = loss_fn(output, target)
     loss.backward()
     optimizer.step()

Often, batch refers to the output of loader, but not in SentenceAx
for batch_index, batch in enumerate(loader):
    input, target = batch


batch_d={
    "labels"= np.array of ints, shape=(batch_size, depth, labels_length)

    "meta_data"= any

    "pos_index"= int

    "text"= str

    "verb"= list[int], a list of 0, 1, 1 if word in text is a verb and 0 if not

    "verb_index"= list[int], locations of verbs  in text

    "word_starts"=
}

"""
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import math
import shutil
from glob import glob
from time import time
from Model import *
from DLoader import *
from sax_globals import *
from sax_utils import *
import tqdm


class MConductor:  # formerly run.py
    def __init__(self):
        self.params_d = PARAMS_D
        self.task = TASK
        assert self.task in ["ex", "cc"]
        self.model = None
        self.saved = False
        self.save_dir = "data/save"
        self.has_cuda = torch.cuda.is_available()

        if self.task == 'cc':
            self.train_fp = 'data/openie-data/ptb-train.labels'
            self.dev_fp = 'data/openie-data/ptb-dev.labels'
            self.test_fp = 'data/openie-data/ptb-test.labels'
        elif self.task == 'ex':
            self.train_fp = 'data/openie-data/openie4_labels'
            self.dev_fp = 'data/carb-data/dev.txt'
            self.test_fp = 'data/carb-data/test.txt'

    def set_checkpoint_callback(self):
        if self.saved:
            self.checkpoint_callback = ModelCheckpoint(
                filepath=self.save_dir + '/{epoch:02d}_{eval_acc:.3f}',
                verbose=True,
                monitor='eval_acc',
                mode='max',
                save_top_k=self.params_d["save_k"] if not self.params_d[
                    "debug"] else 0,
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
        log_dir = self.params_d["save"] + '/logs/'
        if os.path.exists(log_dir + f'{mode}'):
            mode_logs = list(glob(log_dir + f'/{mode}_*'))
            new_mode_index = len(mode_logs) + 1
            print('Moving old log to...')
            print(shutil.move(self.params_d["save"] + f'/logs/{mode}',
                              self.params_d[
                                  "save"] + f'/logs/{mode}_{new_mode_index}'))
        logger = TensorBoardLogger(
            save_dir=self.params_d["save"],
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
            **self.params_d)
        return trainer

    def update_params_d(self, checkpoint_path,
                        **final_changes_params_d):
        if self.has_cuda:
            loaded_params_d = torch.load(checkpoint_path)['params_d']
        else:
            loaded_params_d = torch.load(
                checkpoint_path,
                map_location=torch.device('cpu'))['params_d']

        update_dict(self.params_d, loaded_params_d)
        if final_changes_params_d:
            update_dict(self.params_d, final_changes_params_d)

    def train(self):
        self.set_checkpoint_callback()
        self.model = Model()
        logger = self.get_logger('train')
        trainer = self.get_trainer(logger)
        mdl = DLoader(self.params_d)
        trainer.fit(self.model,
                    train_dataloader=mdl.get_ttt_dataloaders("train"),
                    val_dataloaders=mdl.get_ttt_dataloaders("val"))
        shutil.move(self.params_d["save"] + f'/logs/train.part',
                    self.params_d["save"] + f'/logs/train')

    def resume(self, **final_changes_params_d):
        self.set_checkpoint_callback()
        checkpoint_path = self.get_checkpoint_path()
        self.update_params_d(checkpoint_path, **final_changes_params_d)
        self.model = Model()
        logger = self.get_logger('resume')
        trainer = self.get_trainer(logger, checkpoint_path)
        mdl = DLoader(self.params_d)
        trainer.fit(self.model,
                    train_dataloader=mdl.get_ttt_dataloaders("train"),
                    val_dataloaders=mdl.get_ttt_dataloaders("val"))
        shutil.move(self.params_d["save"] + f'/logs/resume.part',
                    self.params_d["save"] + f'/logs/resume')

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
        test_f = open(self.params_d["save"] + '/logs/test.txt', 'w')

        for checkpoint_path in all_checkpoint_paths:
            trainer = Trainer(logger=logger,
                              gpus=self.params_d["gpus"],
                              resume_from_checkpoint=checkpoint_path)
            mdl = DLoader(self.params_d)
            trainer.test(self.model,
                         test_dataloaders=mdl.get_ttt_dataloaders("test"))
            result = self.model.results
            test_f.write(f'{checkpoint_path}\t{result}\n')
            test_f.flush()
        test_f.close()
        shutil.move(self.params_d["save"] + f'/logs/test.part',
                    self.params_d["save"] + f'/logs/test')

    def predict(self,
                mapping=None, conj_word_mapping=None,
                **final_changes_params_d):
        self.set_checkpoint_callback()

        # def predict(params_d, checkpoint_callback, meta_data_vocab,
        #             train_dataloader,
        #             val_dataloader, test_dataloader, all_sentences, mapping=None,
        #             conj_word_mapping=None):
        if self.params_d["task"] == 'conj':
            self.params_d["checkpoint"] = self.params_d["conj_model"]
        if self.params_d["task"] == 'oie':
            self.params_d["checkpoint"] = self.params_d["oie_model"]

        checkpoint_path = self.get_checkpoint_path()
        self.update_params_d(checkpoint_path, **final_changes_params_d)

        self.model = Model()

        if mapping != None:
            self.model._metric.mapping = mapping
        if conj_word_mapping != None:
            self.model._metric.conj_word_mapping = conj_word_mapping

        trainer = Trainer(gpus=self.params_d["gpus"], logger=None,
                          resume_from_checkpoint=checkpoint_path)
        start_time = time()
        self.model.all_sentences = all_sentences
        mdl = DLoader(self.params_d)
        trainer.test(self.model,
                     test_dataloaders=mdl.get_ttt_dataloaders("test"))
        end_time = time()
        print(f'Total Time taken = {(end_time - start_time) / 60:2f} minutes')

    def splitpredict(self):
        self.set_checkpoint_callback()

        # def splitpredict(params_d, checkpoint_callback, meta_data_vocab,
        #                  train_dataloader, val_dataloader, test_dataloader,
        #                  all_sentences):
        mapping, conj_word_mapping = {}, {}
        self.params_d["write_allennlp"] = True
        if self.params_d["split_fp"] == '':
            self.params_d["task"] = 'conj'
            self.params_d["checkpoint"] = self.params_d["conj_model"]
            self.params_d["mode"]l_str = 'bert-base-cased'
            self.params_d["mode"] = 'predict'
            mdl = DLoader(self.params_d)
            model = self.predict(None,
                                 meta_data_vocab,
                                 None,
                                 None,
                                 mdl.get_ttt_dataloaders("test"),
                                 all_sentences)
            conj_predictions = model.all_cc_predictions
            sentences_indices_list = model.all_cc_sent_locs
            # conj_predictions = model.predictions
            # sentences_indices = model.all_sentence_indices
            assert len(conj_predictions) == len(sentences_indices_list)
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
            for sentence_indices in sentences_indices_list:
                if len(sentence_indices) == 0:
                    count += 1
                else:
                    count += len(sentence_indices)
            assert count == len(sentences) - 1

        else:
            with open(self.params_d["predict_fp"], 'r') as f:
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

        self.params_d["task"] = 'oie'
        self.params_d["checkpoint"] = self.params_d["oie_model"]
        self.params_d["mode"]l_str = 'bert-base-cased'
        mdl = DLoader(self.params_d)
        _, _, split_test_dataset, meta_data_vocab, _ = \
            mdl.get_ttt_datasets(predict_sentences=sentences)
        split_test_dataloader = DataLoader(
            split_test_dataset,
            batch_size=self.params_d["batch_size"],
            collate_fn=mdl.pad_data,
            num_workers=1)

        model = self.predict(self.params_d,
                             None,
                             meta_data_vocab,
                             None,
                             None,
                             split_test_dataloader,
                             mapping=mapping,
                             conj_word_mapping=conj_word_mapping,
                             all_sentences=all_sentences)

        if 'labels' in self.params_d["type"]:
            label_lines = self.get_extags(model, sentences,
                                          orig_sentences,
                                          sentences_indices_list)
            f = open(self.params_d["out"] + '.labels', 'w')
            f.write('\n'.join(label_lines))
            f.close()

        if self.params_d["rescoring"]:
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
            rescored = self.rescore(inp_fp,
                                    model_dir=self.params_d["rescore_model"],
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
                    exts = exts[:self.params_d["num_extractions"]]
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

                # why must confidence be exponentiated?

                extraction = Extraction_sax(arg1=arg1,
                                            rel=rel,
                                            arg2=arg2,
                                            ex_sent=sentence,
                                            confidence=math.exp(confidence))
                extraction.arg1 = arg1
                extraction.arg2 = arg2
                if self.params_d["type"] == 'sentences':
                    ext_str = extraction.get_str() + '\n'
                else:
                    ext_str = extraction.get_str() + '\n'
                exts.append(ext_str)

            exts = sorted(exts, reverse=True,
                          key=lambda x: float(x.split()[0][:-1]))
            exts = exts[:self.params_d["num_extractions"]]
            all_predictions.append(sentence_str + ''.join(exts))

            if line_i + 1 in no_extractions:
                for no_extraction_sentence in no_extractions[line_i + 1]:
                    all_predictions.append(f'{no_extraction_sentence}\n')

            if self.params_d["out"] != None:
                print('Predictions written to ', self.params_d["out"])
                predictions_f = open(self.params_d["out"], 'w')
                predictions_f.write('\n'.join(all_predictions) + '\n')
                predictions_f.close()
            return
