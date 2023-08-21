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
from sax_utils import *
from sax_globals import *


class MConductor:
    """
    similar to run.py
    
    
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
        "lll_label": np.array of ints, shape:(batch_size, depth, labels_length)
    
        "meta_data": any
    
        "pos_index": int
    
        "text": str
    
        "verb_mask": list[int], a list of 0, 1, 1 if word in text is a verb and 0 if not
    
        "verb_locs": list[int], locations of verbs  in text
    
        "word_starts":
    }


    Refs:
    https://spacy.io/usage/spacy-101/

    """

    def __init__(self, pred_fname):
        """


        """
        self.pred_fname = pred_fname
        self.params_d = PARAMS_D
        self.model = None
        self.saved = False
        self.has_cuda = torch.cuda.is_available()

        if TASK == 'cc':
            self.train_fp = CCTAGS_TRAIN_FP
            self.dev_fp = CCTAGS_TUNE_FP
            self.test_fp = CCTAGS_TEST_FP
        elif TASK == 'ex':
            self.train_fp = EXTAGS_TRAIN_FP
            self.dev_fp = EXTAGS_TUNE_FP
            self.test_fp = EXTAGS_TEST_FP

        self.checkpoint_callback = self.get_checkpoint_callback()

        do_lower_case = ('uncased' in self.params_d["model_str"])
        self.auto_tokenizer = AutoTokenizer.from_pretrained(
            self.params_d["model_str"],
            do_lower_case=do_lower_case,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)

        # encode() (a.k.a. convert_tokens_to_ids())
        # replaces vocab.stoi() (stoi=string to integer)
        self.encode = self.auto_tokenizer.encode
        # decode()
        # replaces vocab.itos() (itos=integer to string)
        self.decode = self.auto_tokenizer.decode
        self.sent_pad_id = self.encode(self.auto_tokenizer.pad_token)

        self.dloader = DLoader(self.auto_tokenizer,
                               self.train_fp,
                               self.dev_fp,
                               self.test_fp)

    def get_checkpoint_callback(self):
        """

        Returns
        -------

        """
        return ModelCheckpoint(
            filepath=WEIGHTS_DIR + "/" +
                     TASK + '_model/{epoch:02d}_{eval_acc:.3f}',
            verbose=True,
            monitor='eval_acc',
            mode='max',
            save_top_k=self.params_d["save_k"] \
                if not self.params_d["debug"] else 0,
            period=0)

    def get_all_checkpoint_paths(self):
        """
        similar to run.get_checkpoint_path()

        Returns
        -------

        """
        if self.params_d["checkpoint_fp"]:
            return [self.params_d["checkpoint_fp"]]

        else:
            return glob(WEIGHTS_DIR + '/*.ckpt')

    def get_checkpoint_path(self):
        """

        Returns
        -------

        """
        all_paths = self.get_all_checkpoint_paths()
        assert len(all_paths) == 1
        return all_paths[0]

    def get_logger(self):
        """
        similar to run.get_logger()

        Parameters
        ----------

        Returns
        -------

        """

        # the current log file will have no number prefix,
        # stored ones will.
        if os.path.exists(LOG_DIR + f'/{MODE}'):
            num_numbered_logs = len(list(glob(LOG_DIR + f'/{MODE}_*')))
            new_id = num_numbered_logs + 1
            print('Retiring current log file by changing its name')
            print(shutil.move(LOG_DIR + f'/{MODE}',
                              LOG_DIR + f'/{MODE}_{new_id}'))
        logger = TensorBoardLogger(
            save_dir=WEIGHTS_DIR,
            name=TASK + '_logs',
            version=MODE + '.part')
        return logger

    def get_trainer(self, logger, checkpoint_path,
                    use_minimal=False):
        """

        Parameters
        ----------
        logger
        checkpoint_path

        Returns
        -------

        """
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
        if MODE == "resume":
            resume_from_cpoint = checkpoint_path
        else:
            resume_from_cpoint = None

        if use_minimal:
            trainer = Trainer(
                gpus=self.params_d["gpus"],
                logger=logger,
                resume_from_checkpoint=checkpoint_path)
        else:
            trainer = Trainer(
                accumulate_grad_batches=\
                    int(self.params_d["accumulate_grad_batches"]),
                checkpoint_callback=self.checkpoint_callback,
                logger=logger,
                max_epochs = self.params_d["epochs"],
                min_epochs = self.params_d["epochs"],
                resume_from_checkpoint=resume_from_cpoint,
                show_progress_bar=True,
                **self.params_d)
        return trainer

    def update_params_d(self, checkpoint_path):
        """
        similar to in run.test() and data.override_args()


        Parameters
        ----------
        checkpoint_path

        Returns
        -------

        """
        if self.has_cuda:
            loaded_params_d = torch.load(
                self.checkpoint_path)['params_d']
        else:
            mloc = torch.device('cpu')
            loaded_params_d = torch.load(
                self.checkpoint_path, map_location=mloc)['params_d']

        self.params_d = merge_dicts(loaded_params_d,
                                    default_d=self.params_d)

    def train(self):
        """
        similar to run.train()

        Returns
        -------

        """
        # train is the only mode that doesn't require update_params_d()
        self.model = Model(self.params_d, self.auto_tokenizer)
        logger = self.get_logger()
        trainer = self.get_trainer(logger)
        trainer.fit(
            self.model,
            train_dataloader=self.dloader.get_ttt_dataloaders("train"),
            val_dataloaders=self.dloader.get_ttt_dataloaders("val"))
        shutil.move(WEIGHTS_DIR + f'/logs/train.part',
                    WEIGHTS_DIR + f'/logs/train')

    def resume(self):
        """
        similar to run.resume()


        Parameters
        ----------
        final_changes_params_d

        Returns
        -------

        """
        checkpoint_path = self.get_checkpoint_path()
        self.update_params_d(checkpoint_path)
        self.model = Model(self.params_d, self.auto_tokenizer)
        logger = self.get_logger()
        trainer = self.get_trainer(logger, checkpoint_path)
        trainer.fit(
            self.model,
            train_dataloader=self.dloader.get_ttt_dataloaders("train"),
            val_dataloaders=self.dloader.get_ttt_dataloaders("val"))
        shutil.move(WEIGHTS_DIR + f'/logs/resume.part',
                    WEIGHTS_DIR + f'/logs/resume')

    def test(self,
             orig_sent_to_ex_sent=None,
             orig_sent_to_cc_sent=None):
        """
        similar to run.test()


        Parameters
        ----------
        train
        mapping
        conj_word_mapping
        final_changes_params_d

        Returns
        -------

        """
        checkpoint_path = self.get_checkpoint_path()
        if 'train' not in MODE:
            # train is the only mode that doesn't require update_params_d()
            self.update_params_d(checkpoint_path)

        self.model = Model(self.params_d, self.auto_tokenizer)
        if orig_sent_to_ex_sent:
            self.model.metric.mapping = orig_sent_to_ex_sent
        if  orig_sent_to_cc_sent:
            self.model.metric.conj_word_mapping = orig_sent_to_cc_sent

        logger = self.get_logger()
        test_f = open(WEIGHTS_DIR + '/logs/test.txt', 'w')

        for checkpoint_path in self.get_all_checkpoint_paths():
            trainer = self.get_trainer(logger,
                                       checkpoint_path,
                                       use_minimal = True)
            # trainer.fit() and trainer.test() are different
            trainer.test(
                self.model,
                test_dataloaders=self.dloader.get_ttt_dataloaders("test"))
            result = self.model.results
            test_f.write(f'{checkpoint_path}\t{result}\n')
            # note test_f created outside loop.
            # refresh/clear/flush test_f after each write
            test_f.flush()
        test_f.close()
        shutil.move(WEIGHTS_DIR + f'/logs/test.part',
                    WEIGHTS_DIR + f'/logs/test')

    def predict(self,
             orig_sent_to_ex_sent=None,
             orig_sent_to_cc_sent=None):
        """
        similar to run.predict()

        Parameters
        ----------
        mapping
        conj_word_mapping
        final_changes_params_d

        Returns
        -------

        """

        # def predict(checkpoint_callback,
        #             train_dataloader,
        #             val_dataloader, test_dataloader, all_sentences,
        #             mapping=None,
        #             conj_word_mapping=None):
        if self.params_d["task"] == 'cc':
            self.params_d["checkpoint_fp"] = self.params_d["cc_model_fp"]
        if self.params_d["task"] == 'ex':
            self.params_d["checkpoint_fp"] = self.params_d["ex_model_fp"]

        checkpoint_path = self.get_checkpoint_path()
        self.update_params_d(checkpoint_path)
        self.model = Model(self.params_d, self.auto_tokenizer)

        if orig_sent_to_ex_sent:
            self.model.metric.mapping = orig_sent_to_ex_sent
        if orig_sent_to_cc_sent:
            self.model.metric.conj_word_mapping = orig_sent_to_cc_sent

        logger = None
        trainer = self.get_trainer(logger,
                                   checkpoint_path,
                                   use_minimal=True)
        start_time = time()
        # self.model.all_sentences = all_sentences
        trainer.test(
            self.model,
            test_dataloaders=self.dloader.get_ttt_dataloaders("test"))
        end_time = time()
        print(f'Total Time taken = {(end_time - start_time) / 60:2f} minutes')

    def splitpredict(self):
        """
        similar to run.splitpredict()


        Returns
        -------

        """

        # def splitpredict(params_d, checkpoint_callback,
        #                  train_dataloader, val_dataloader, test_dataloader,
        #                  all_sentences):
        orig_sent_to_ex_sent = {}
        orig_sent_to_cc_sent = {}
        self.params_d["write_allennlp"] = True
        if self.params_d["split_fp"] == '':
            self.params_d["task"] = 'cc'
            TASK = "cc"
            self.params_d["checkpoint_fp"] = self.params_d["cc_model_fp"]
            self.params_d["model_str"] = 'bert-base-cased'
            self.params_d["mode"] = 'predict'
            MODE = "predict"
            model = self.predict(
                orig_sent_to_ex_sent=None,
                orig_sent_to_cc_sent=None)
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
                    orig_sent_to_ex_sent[list_sentences[0]] = list_sentences[0]
                    orig_sent_to_cc_sent[list_sentences[0]] = conj_words
                    sentences.append(
                        list_sentences[0] + ' [unused1] [unused2] [unused3]')
                elif len(list_sentences) > 1:
                    orig_sentences.append(
                        list_sentences[0] + ' [unused1] [unused2] [unused3]')
                    orig_sent_to_cc_sent[list_sentences[0]] = conj_words
                    for sent in list_sentences[1:]:
                        orig_sent_to_ex_sent[sent] = list_sentences[0]
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

        else: # no split_fp
            with open(INP_FP, 'r') as f:
                lines = f.read()
                lines = lines.replace("\\", "")

            sentences = []
            orig_sentences = []
            extra_str = " [unused1] [unused2] [unused3]"
            for line in lines.split('\n\n'):
                if len(line) > 0:
                    list_sentences = line.strip().split('\n')
                    if len(list_sentences) == 1:
                        orig_sent_to_ex_sent[list_sentences[0]] = list_sentences[0]
                        sentences.append(list_sentences[0] + extra_str)
                        orig_sentences.append(list_sentences[0] + extra_str)
                    elif len(list_sentences) > 1:
                        orig_sentences.append(list_sentences[0] + extra_str)
                        for sent in list_sentences[1:]:
                            orig_sent_to_ex_sent[sent] = list_sentences[0]
                            sentences.append(sent + extra_str)
                    else:
                        assert False

        self.params_d["task"] = 'ex'
        TASK = "ex"
        self.params_d["checkpoint_fp"] = self.params_d["ex_model_fp"]
        self.params_d["model_str"] = 'bert-base-cased'
        _, _, split_test_dataset = \
            self.dloader.get_ttt_datasets(predict_sentences=sentences)
        split_test_dataloader = DataLoader(
            split_test_dataset,
            batch_size=self.params_d["batch_size"],
            # collate_fn=mdl.pad_data,
            num_workers=1)

        model = self.predict(
            None,
            None,
            None,
            split_test_dataloader,
            mapping=orig_sent_to_ex_sent,
            conj_word_mapping=orig_sent_to_cc_sent)
        # all_sentences=all_sentences)

        if 'labels' in self.params_d["type"]:
            label_lines = self.get_extags(model, sentences,
                                          orig_sentences,
                                          sentences_indices_list)
            f = open(PREDICTIONS_DIR + '/ex_labels.txt', 'w')
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
                                    model_dir=RESCORE_DIR,
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

                extraction = SAXExtraction(arg1=arg1,
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

            if self.pred_fname:
                fpath = PREDICTIONS_DIR + "/" + self.pred_fname
                print('Predictions written to ', fpath)
                predictions_f = open(fpath, 'w')
                predictions_f.write('\n'.join(all_predictions) + '\n')
                predictions_f.close()
            return
