from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import warnings

import os
import math
import shutil
from glob import glob
from time import time
from Model import *
from SaxDataLoader import *
from sax_utils import *
from sax_globals import *
from sample_classes import *


class MConductor:
    """
    similar to run.py
    
    
    Torch lightning
    
    Dataset stores the samples and their corresponding labels, and DataLoader
    wraps an iterable around the Dataset to enable easy access to the samples.
    
    DataLoader is located in torch.utils.data
    
    loader = torch.utils.dataloader(dset)
    for input, target in loader:
         output.meta_data = model(input)
         loss_fun = loss_fn(output.meta_data, target)
         loss_fun.backward()
         optimizer.step()
    
    Often, batch refers to the output.meta_data of loader, but not in SentenceAx
    for batch_index, batch in enumerate(loader):
        input, target = batch
    
    


    Refs:
    https://spacy.io/usage/spacy-101/

    """

    def __init__(self, pred_fname=None):
        """


        """
        self.pred_fname = pred_fname
        self.params_d = PARAMS_D
        self.has_been_saved = False
        self.has_cuda = torch.cuda.is_available()
        warnings.filterwarnings('ignore')

        if TASK == 'cc':
            self.train_fp = CCTAGS_TRAIN_FP
            self.val_fp = CCTAGS_TUNE_FP
            self.test_fp = CCTAGS_TEST_FP
        elif TASK == 'ex':
            self.train_fp = EXTAGS_TRAIN_FP
            self.val_fp = EXTAGS_TUNE_FP
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
        self.pad_icode = self.encode(self.auto_tokenizer.pad_token)

        self.dloader = SaxDataLoader(self.auto_tokenizer,
                                  self.pad_icode,
                                  self.train_fp,
                                  self.val_fp,
                                  self.test_fp)

        self.constraints_str_d = dict()

        self.cc_ll_spanned_word = []
        self.cc_ll_spanned_loc = []
        self.cc_l_pred_str = []

        self.l_pred_sentL = None
        self.l_orig_sentL = None

        self.ex_l_pred_str = None

        self.model = None
        self.ex_fit_d = None
        self.cc_fit_d = None

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
        assert os.path.exists(LOG_DIR + "/" + MODE)
        num_numbered_logs = len(list(glob(LOG_DIR + f'/{MODE}_*')))
        new_id = num_numbered_logs + 1
        print('Retiring current log file by changing its name')
        print(shutil.move(LOG_DIR + f'/{MODE}',
                          LOG_DIR + f'/{MODE}_{new_id}'))
        logger = TensorBoardLogger(
            save_dir=WEIGHTS_DIR,
            name='logs',
            version=MODE + '.part')
        return logger

    def get_trainer(self, logger, checkpoint_path,
                    use_minimal):
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

        if use_minimal:
            trainer = Trainer(
                gpus=self.params_d["gpus"],
                logger=logger,
                resume_from_checkpoint=checkpoint_path)
        else:
            trainer = Trainer(
                accumulate_grad_batches= \
                    int(self.params_d["accumulate_grad_batches"]),
                checkpoint_callback=self.checkpoint_callback,
                logger=logger,
                max_epochs=self.params_d["epochs"],
                min_epochs=self.params_d["epochs"],
                resume_from_checkpoint= \
                    checkpoint_path if MODE == "resume" else None,
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
            loaded_params_d = torch.load(checkpoint_path)["params_d"]
        else:
            map_loc = torch.device('cpu')
            loaded_params_d = torch.load(
                checkpoint_path, map_location=map_loc)["params_d"]

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
        trainer = self.get_trainer(logger,
                                   checkpoint_path=None,
                                   use_minimal=False)
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
        # train is the only mode that doesn't require
        # update_params_d() because it is called first
        self.update_params_d(checkpoint_path)
        self.model = Model(self.params_d, self.auto_tokenizer)
        logger = self.get_logger()
        trainer = self.get_trainer(logger,
                                   checkpoint_path,
                                   use_minimal=False)
        trainer.fit(
            self.model,
            train_dataloader=self.dloader.get_ttt_dataloaders("train"),
            val_dataloaders=self.dloader.get_ttt_dataloaders("val"))
        shutil.move(WEIGHTS_DIR + '/logs/resume.part',
                    WEIGHTS_DIR + '/logs/resume')

    def test(self):
        """
        similar to run.test()


        Parameters
        ----------

        Returns
        -------

        """
        checkpoint_path = self.get_checkpoint_path()
        if 'train' not in MODE:
            # train is the only mode that doesn't require
            # update_params_d() because it is called first
            self.update_params_d(checkpoint_path)

        self.model = Model(self.params_d, self.auto_tokenizer)
        if TASK == "ex" and self.ex_fix_d:
            self.model.metric.fix_d = self.ex_fix_d
        if TASK == "cc" and self.cc_fix_d:
            self.model.metric.fix_d = self.cc_fix_d

        logger = self.get_logger()
        with open(WEIGHTS_DIR + '/logs/test.txt', 'w') as test_f:
            for checkpoint_path in self.get_all_checkpoint_paths():
                trainer = self.get_trainer(logger,
                                           checkpoint_path,
                                           use_minimal=True)
                # trainer.fit() and trainer.test() are different
                trainer.test(
                    self.model,
                    test_dataloaders=self.dloader.get_ttt_dataloaders("test"))
                eval_out_d = self.model.eval_out_d
                test_f.write(f'{checkpoint_path}\t{eval_out_d}\n')
                # note test_f created outside loop.
                # refresh/clear/flush test_f after each write
                test_f.flush()
        shutil.move(WEIGHTS_DIR + f'/logs/test.part',
                    WEIGHTS_DIR + f'/logs/test')

    def predict(self, test_dloader=None):
        """
        similar to run.predict()

        Parameters
        ----------


        Returns
        -------

        """

        # def predict(checkpoint_callback,
        #             train_dataloader,
        #             val_dataloader, test_dataloader, all_sentences,
        #             mapping=None,
        #             conj_word_mapping=None):
        if TASK == 'cc':
            self.params_d["checkpoint_fp"] = self.params_d["cc_model_fp"]
        if TASK == 'ex':
            self.params_d["checkpoint_fp"] = self.params_d["ex_model_fp"]

        checkpoint_path = self.get_checkpoint_path()
        self.update_params_d(checkpoint_path)
        self.model = Model(self.params_d, self.auto_tokenizer)

        if TASK == "ex" and self.ex_fix_d:
            self.model.metric.fix_d = self.ex_fix_d
        elif TASK == "cc" and self.cc_fix_d:
            self.model.metric.fix_d = self.cc_fix_d

        logger = None
        trainer = self.get_trainer(logger,
                                   checkpoint_path,
                                   use_minimal=True)
        start_time = time()
        # self.model.all_sentences = all_sentences
        trainer.test(
            self.model,
            test_dataloaders=test_dloader if
                test_dloader else self.dloader.get_ttt_dataloaders("test"))
        end_time = time()
        print(f'Total Time taken = {(end_time - start_time) / 60:2f} minutes')

    def splitpredict_do_cc_first(self):
        self.ex_fix_d = {}
        self.cc_fix_d = {}
        if not PRED_IN_FP:
            self.params_d["task"] = TASK = 'cc'
            self.params_d["checkpoint_fp"] = self.params_d["cc_model_fp"]
            self.params_d["model_str"] = 'bert-base-cased'
            self.params_d["mode"] = MODE = 'predict'
            self.predict()
            l_pred_str = self.cc_l_pred_str
            ll_spanned_loc = self.cc_ll_spanned_loc
            assert len(l_pred_str) == len(ll_spanned_loc)
            ll_spanned_word = self.cc_ll_spanned_word

            l_pred_sentL = []
            l_orig_sentL = []
            for sample_id, pred_str in enumerate(l_pred_str):
                l_pred_sent = pred_str.strip('\n').split('\n')

                # not done when reading from PRED_IN_FP
                words = ll_spanned_word[sample_id]
                self.cc_fix_d[l_pred_sent[0]] = " ".join(words)

                l_orig_sentL.append(l_pred_sent[0] + UNUSED_TOKENS_STR)
                for sent in l_pred_sent:
                    self.ex_fix_d[sent] = l_pred_sent[0]
                    l_pred_sentL.append(sent + UNUSED_TOKENS_STR)
            # this not done when reading from PRED_IN_FP
            # l_orig_sentL.append('\

            # Never used:
            # count = 0
            # for l_spanned_loc in ll_spanned_loc:
            #     if len(l_spanned_loc) == 0:
            #         count += 1
            #     else:
            #         count += len(l_spanned_loc)
            # assert count == len(l_orig_sentL) - 1

        else:
            with open(PRED_IN_FP, 'r') as f:
                content = f.read()
            content = content.replace("\\", "")
            lines = content.split('\n\n')

            l_pred_sentL = []
            l_orig_sentL = []
            for line in lines:
                if len(line) > 0:
                    l_pred_sent = line.strip().split('\n')
                    l_orig_sentL.append(l_pred_sent[0] + UNUSED_TOKENS_STR)
                    for sent in l_pred_sent:
                        self.ex_fix_d[sent] = l_pred_sent[0]
                        l_pred_sentL.append(sent + UNUSED_TOKENS_STR)
        self.l_pred_sentL = l_pred_sentL
        self.l_orig_sentL = l_orig_sentL

    def splitpredict_do_ex_second(self):
        self.params_d["task"] = TASK = 'ex'
        self.params_d["checkpoint_fp"] = self.params_d["ex_model_fp"]
        self.params_d["model_str"] = 'bert-base-cased'
        pred_test_dataloader = self.dloader.get_ttt_dataloaders(
            "test", self.l_pred_sentL)

        self.predict(test_dloader=pred_test_dataloader)

        samples = self.model.batch_out.l_sample
        samples.absorb_all_possible()
        path = PREDICTIONS_DIR + "/" + self.pred_fname +"-extags.txt"
        with_scores = False
        # Does same thing as Openie6's run.get_labels()
        # this replaces self.write_extags_file_from_predictions()
        write_extags_file(samples, path, with_scores)

    def splitpredict_do_rescoring(self):
        self.params_d["write_allennlp"] = True
        print()
        print("Starting re-scoring ...")
        print()

        sentence_line_nums = set()
        prev_line_num = 0
        no_extractions = {}
        curr_line_num = 0
        for sentence_str in self.all_predictions_oie:
            sentence_str = sentence_str.strip('\n')
            num_extrs = len(sentence_str.split('\n')) - 1
            if num_extrs == 0:
                if curr_line_num not in no_extractions:
                    no_extractions[curr_line_num] = []
                no_extractions[curr_line_num].append(sentence_str)
                continue
            curr_line_num = prev_line_num + num_extrs
            sentence_line_nums.add(
                curr_line_num)  # check extra empty lines,
            # example with no extractions
            prev_line_num = curr_line_num

        # testing rescoring
        in_fp = self.model.predictions_f_allennlp
        rescored = self.rescore(in_fp,
                                model_dir=RESCORE_DIR,
                                batch_size=256)

        all_predictions = []
        sentence_str = ''
        for line_i, line in enumerate(rescored):
            fields = line.split('\t')
            sentence = fields[0]
            score = float(fields[2])

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

            # why must score be exponentiated?

            extraction = SaxExtraction(orig_sentL=orig_senL,
                                       arg1=arg1,
                                       rel=rel,
                                       arg2=arg2,
                                       score=math.exp(score))
            extraction.arg1 = arg1
            extraction.arg2 = arg2
            if self.params_d["type"] == 'sentences':
                ext_str = extraction.get_simple_sent() + '\n'
            else:
                ext_str = extraction.get_simple_sent() + '\n'
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

    def splitpredict(self):
        """
        similar to run.splitpredict()


        Returns
        -------

        """

        # def splitpredict(params_d, checkpoint_callback,
        #                  train_dataloader, val_dataloader, test_dataloader,
        #                  all_sentences):

        self.splitpredict_do_cc_first()
        self.splitpredict_do_ex_second()
        if self.params_d["rescoring"]:
            self.splitpredict_do_rescoring()

    def write_extags_file_from_predictions(self,
                                           l_sentL,  # original sentences
                                           ll_sent_loc):
        """
        similar to run.get_labels()
        ICODE_TO_EXTAG={0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
                 4: 'ARG2', 5: 'NONE'}


        Parameters
        ----------
        l_sentL
        ll_sent_loc

        Returns
        -------

        """

        lines = []
        sample_id = 0
        ex_id = 0
        word_id = 0
        l_sample = self.model.batch_out.l_sample
        num_samples = len(l_sample)

        for sample_id in range(num_samples):
            l_child = l_sample[sample_id].l_child
            if len(ll_sent_loc[sample_id]) == 0:
                words = get_words(l_sentL[sample_id].split('[unused1]')[0])
                ll_sent_loc[sample_id].append(list(range(len(words))))

            lines.append(
                '\n' + l_sentL[sample_id].split('[unused1]')[0].strip())
            for ex_id in range(len(ll_sent_loc[sample_id])):
                assert len(ll_sent_loc[sample_id][ex_id]) == len(
                    get_words(l_child[ex_id].orig_sent))
                sentL = l_child[ex_id].strip() + UNUSED_TOKENS_STR
                assert sentL == l_sentL[sample_id]
                ll_icode = l_child[ex_id].icodes
                for l_icode in ll_icode:
                    # You can use x.item() to get a Python number
                    # from a torch tensor that has one element
                    if pred_icodes.sum().item() == 0:
                        break

                    icodes = [0] * len(get_words(sentL))
                    pred_icodes = pred_icodes[:len(sentL.split())].tolist()
                    for k, loc in enumerate(
                            sorted(ll_sent_loc[sample_id][ex_id])):
                        icodes[loc] = pred_icodes[k]

                    icodes = icodes[:-3]
                    # 1: arg1, 2: rel
                    if 1 not in pred_icodes and 2 not in pred_icodes:
                        continue

                    str_extags = \
                        ' '.join([ICODE_TO_EXTAG[i] for i in icodes])
                    lines.append(str_extags)

                word_id += 1
                ex_id += 1
                if ex_id == len(l_sample):
                    ex_id = 0
                    sample_id += 1

        lines.append('\n')
        assert self.pred_fname
        with open(PREDICTIONS_DIR + "/" + self.pred_fname +
                  "-extags.txt") as f:
            f.writelines(lines)
