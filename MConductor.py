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
from Params import *


class MConductor:
    """
    similar to Openie6.run.py
    
    
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

    def __init__(self, params, pred_fname=None):
        """


        """
        self.pred_fname = pred_fname
        self.params = params
        self.has_been_saved = False
        self.has_cuda = torch.cuda.is_available()
        warnings.filterwarnings('ignore')

        if self.params.task == 'cc':
            self.train_fp = CCTAGS_TRAIN_FP
            self.tune_fp = CCTAGS_TUNE_FP
            self.test_fp = CCTAGS_TEST_FP
        elif self.params.task == 'ex':
            self.train_fp = EXTAGS_TRAIN_FP
            self.tune_fp = EXTAGS_TUNE_FP
            self.test_fp = EXTAGS_TEST_FP

        self.checkpoint_callback = self.get_checkpoint_callback()

        do_lower_case = ('uncased' in self.params.d["model_str"])
        self.auto_tokenizer = AutoTokenizer.from_pretrained(
            self.params.d["model_str"],
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
        self.pad_ilabel = self.encode(self.auto_tokenizer.pad_token)

        self.dloader = SaxDataLoader(self.auto_tokenizer,
                                     self.pad_ilabel,
                                     self.train_fp,
                                     self.tune_fp,
                                     self.test_fp)

        self.constraint_str_d = dict()

        self.cc_ll_spanned_word = []
        self.cc_ll_spanned_loc = []
        self.cc_l_pred_str = []

        self.ex_l_pred_str = None

        self.model = None
        self.ex_fit_d = {}
        self.cc_fit_d = {}

    def get_checkpoint_callback(self):
        """

        Returns
        -------

        """
        return ModelCheckpoint(
            filepath=WEIGHTS_DIR + "/" +
                     self.params.task + '_model/{epoch:02d}_{eval_acc:.3f}',
            verbose=True,
            monitor='eval_acc',
            mode='max',
            save_top_k=self.params.d["save_k"] \
                if not self.params.d["debug"] else 0,
            period=0)

    def get_all_checkpoint_paths(self):
        """
        similar to Openie6.run.get_checkpoint_path()

        Returns
        -------

        """
        if "checkpoint_fp" in self.params.d:
            return [self.params.d["checkpoint_fp"]]

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
        similar to Openie6.run.get_logger()

        Parameters
        ----------

        Returns
        -------

        """

        # the current log file will have no number prefix,
        # stored ones will.
        assert os.path.exists(self.params.log_dir() + "/" + self.params.mode)
        num_numbered_logs = len(
            list(glob(self.params.log_dir() + f'/{self.params.mode}_*')))
        new_id = num_numbered_logs + 1
        print('Retiring current log file by changing its name')
        print(shutil.move(self.params.log_dir() + f'/{self.params.mode}',
                          self.params.log_dir() + f'/{self.params.mode}_{new_id}'))
        logger = TensorBoardLogger(
            save_dir=WEIGHTS_DIR,
            name='logs',
            version=self.self.params.mode + '.part')
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
                gpus=self.params.d["gpus"],
                logger=logger,
                resume_from_checkpoint=checkpoint_path)
        else:
            trainer = Trainer(
                accumulate_grad_batches= \
                    int(self.params.d["accumulate_grad_batches"]),
                checkpoint_callback=self.checkpoint_callback,
                logger=logger,
                max_epochs=self.params.d["epochs"],
                min_epochs=self.params.d["epochs"],
                resume_from_checkpoint= \
                    checkpoint_path if self.params.mode == "resume" else None,
                show_progress_bar=True,
                **self.params.d)
        return trainer

    def update_params(self, checkpoint_path):
        """
        similar to Openie6.run.test() and data.override_args()


        Parameters
        ----------
        checkpoint_path

        Returns
        -------

        """
        if self.has_cuda:
            loaded_params = torch.load(checkpoint_path)["params"]
        else:
            map_loc = torch.device('cpu')
            loaded_params = torch.load(
                checkpoint_path, map_location=map_loc)["params"]

        self.params.d = merge_dicts(loaded_params.d,
                                    default_d=self.params.d)

    def train(self):
        """
        similar to Openie6.run.train()

        Returns
        -------

        """
        # train is the only mode that doesn't require update_params()
        self.model = Model(self.params.d, self.auto_tokenizer)
        logger = self.get_logger()
        trainer = self.get_trainer(logger,
                                   checkpoint_path=None,
                                   use_minimal=False)
        trainer.fit(
            self.model,
            train_dataloader=self.dloader.get_ttt_dataloaders("train"),
            tune_dataloaders=self.dloader.get_ttt_dataloaders("tune"))
        shutil.move(WEIGHTS_DIR + f'/logs/train.part',
                    WEIGHTS_DIR + f'/logs/train')

    def resume(self):
        """
        similar to Openie6.run.resume()


        Returns
        -------

        """
        checkpoint_path = self.get_checkpoint_path()
        # train is the only mode that doesn't require
        # update_params() because it is called first
        self.update_params(checkpoint_path)
        self.model = Model(self.params.d, self.auto_tokenizer)
        logger = self.get_logger()
        trainer = self.get_trainer(logger,
                                   checkpoint_path,
                                   use_minimal=False)
        trainer.fit(
            self.model,
            train_dataloader=self.dloader.get_ttt_dataloaders("train"),
            tune_dataloaders=self.dloader.get_ttt_dataloaders("tune"))
        shutil.move(WEIGHTS_DIR + '/logs/resume.part',
                    WEIGHTS_DIR + '/logs/resume')

    def test(self):
        """
        similar to Openie6.run.test()


        Parameters
        ----------

        Returns
        -------

        """
        checkpoint_path = self.get_checkpoint_path()
        if 'train' not in self.params.mode:
            # train is the only mode that doesn't require
            # update_params() because it is called first
            self.update_params(checkpoint_path)

        self.model = Model(self.params.d, self.auto_tokenizer)
        if self.params.task == "ex" and self.ex_fix_d:
            self.model.metric.fix_d = self.ex_fix_d
        if self.params.task == "cc" and self.cc_fix_d:
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
        similar to Openie6.run.predict()

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
        if self.params.task == 'cc':
            self.params.d["checkpoint_fp"] = self.params.d["cc_model_fp"]
        if self.params.task == 'ex':
            self.params.d["checkpoint_fp"] = self.params.d["ex_model_fp"]

        checkpoint_path = self.get_checkpoint_path()
        self.update_params(checkpoint_path)
        self.model = Model(self.params.d, self.auto_tokenizer)

        if self.params.task == "ex" and self.ex_fix_d:
            self.model.metric.fix_d = self.ex_fix_d
        elif self.params.task == "cc" and self.cc_fix_d:
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
            self.params.d["task"] = self.params.task = 'cc'
            self.params.d["checkpoint_fp"] = self.params.d["cc_model_fp"]
            self.params.d["model_str"] = 'bert-base-cased'
            self.params.d["mode"] = self.params.mode = 'predict'
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
                self.cc_fix_d[l_pred_sent[??]] = " ".join(words)

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
        self.params.d["task"] = self.params.task = 'ex'
        self.params.d["checkpoint_fp"] = self.params.d["ex_model_fp"]
        self.params.d["model_str"] = 'bert-base-cased'
        pred_test_dataloader = self.dloader.get_ttt_dataloaders(
            "test", self.l_pred_sentL)

        self.predict(test_dloader=pred_test_dataloader)

        path = PREDICTIONS_DIR + "/" + self.pred_fname + "-extags.txt"
        with_confis = False
        # Does same thing as Openie6's run.get_labels()
        self.write_extags_file_from_predictions()

    # def splitpredict_do_rescoring(self):
    #     self.params.d["write_allennlp"] = True
    #     print()
    #     print("Starting re-scoring ...")
    #     print()
    #
    #     sentence_line_nums = set()
    #     prev_line_num = 0
    #     no_extractions = {}
    #     curr_line_num = 0
    #     for sentence_str in self.all_predictions_oie:
    #         sentence_str = sentence_str.strip('\n')
    #         num_extrs = len(sentence_str.split('\n')) - 1
    #         if num_extrs == 0:
    #             if curr_line_num not in no_extractions:
    #                 no_extractions[curr_line_num] = []
    #             no_extractions[curr_line_num].append(sentence_str)
    #             continue
    #         curr_line_num = prev_line_num + num_extrs
    #         sentence_line_nums.add(
    #             curr_line_num)  # check extra empty lines,
    #         # example with no extractions
    #         prev_line_num = curr_line_num
    #
    #     # testing rescoring
    #     in_fp = self.model.predictions_f_allennlp
    #     rescored = self.rescore(in_fp,
    #                             model_dir=RESCORE_DIR,
    #                             batch_size=256)
    #
    #     all_predictions = []
    #     sentence_str = ''
    #     for line_i, line in enumerate(rescored):
    #         fields = line.split('\t')
    #         sentence = fields[0]
    #         confi = float(fields[2])
    #
    #         if line_i == 0:
    #             sentence_str = f'{sentence}\n'
    #             exts = []
    #         if line_i in sentence_line_nums:
    #             exts = sorted(exts, reverse=True,
    #                           key=lambda x: float(x.split()[0][:-1]))
    #             exts = exts[:self.params.d["num_extractions"]]
    #             all_predictions.append(sentence_str + ''.join(exts))
    #             sentence_str = f'{sentence}\n'
    #             exts = []
    #         if line_i in no_extractions:
    #             for no_extraction_sentence in no_extractions[line_i]:
    #                 all_predictions.append(f'{no_extraction_sentence}\n')
    #
    #         arg1 = re.findall("<arg1>.*</arg1>", fields[1])[0].strip(
    #             '<arg1>').strip('</arg1>').strip()
    #         rel = re.findall("<rel>.*</rel>", fields[1])[0].strip(
    #             '<rel>').strip('</rel>').strip()
    #         arg2 = re.findall("<arg2>.*</arg2>", fields[1])[0].strip(
    #             '<arg2>').strip('</arg2>').strip()
    #
    #         # why must confi be exponentiated?
    #
    #         extraction = SaxExtraction(orig_sentL=orig_senL,
    #                                    arg1=arg1,
    #                                    rel=rel,
    #                                    arg2=arg2,
    #                                    confi=math.exp(confi))
    #         extraction.arg1 = arg1
    #         extraction.arg2 = arg2
    #         if self.params.d["type"] == 'sentences':
    #             ext_str = extraction.get_simple_sent() + '\n'
    #         else:
    #             ext_str = extraction.get_simple_sent() + '\n'
    #         exts.append(ext_str)
    #
    #     exts = sorted(exts, reverse=True,
    #                   key=lambda x: float(x.split()[0][:-1]))
    #     exts = exts[:self.params.d["num_extractions"]]
    #     all_predictions.append(sentence_str + ''.join(exts))
    #
    #     if line_i + 1 in no_extractions:
    #         for no_extraction_sentence in no_extractions[line_i + 1]:
    #             all_predictions.append(f'{no_extraction_sentence}\n')
    #
    #     if self.pred_fname:
    #         fpath = PREDICTIONS_DIR + "/" + self.pred_fname
    #         print('Predictions written to ', fpath)
    #         predictions_f = open(fpath, 'w')
    #         predictions_f.write('\n'.join(all_predictions) + '\n')
    #         predictions_f.close()

    def splitpredict(self):
        """
        similar to Openie6.run.splitpredict()


        Returns
        -------

        """

        # def splitpredict(params_d, checkpoint_callback,
        #                  train_dataloader, val_dataloader, test_dataloader,
        #                  all_sentences):

        self.splitpredict_do_cc_first()
        self.splitpredict_do_ex_second()
        if "rescoring" in self.params.d:
            # self.splitpredict_do_rescoring()
            print("rescoring not implented yet")

    def write_extags_file_from_predictions(self):
        """
        similar to Openie6.run.get_labels()
        ILABEL_TO_EXTAG={0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
                 4: 'ARG2', 5: 'NONE'}


        Parameters
        ----------
        l_sentL
        l_word_locs

        Returns
        -------

        """
        bout = self.model.batch_out
        num_samples = len(bout.l_orig_sent)
        l_sentL = redoL(bout.l_orig_sent)
        lll_ilabel = bout.lll_ilabel
        lll_word_loc = []

        lines = []
        sam_id0 = 0
        ex_id0 = 0
        word_id0 = 0

        for sam_id in range(num_samples):
            words = get_words(undoL(l_sentL[sam_id]))
            lll_word_loc[sam_id].append(list(range(len(words))))

            lines.append('\n' + undoL(l_sentL[sam_id]))
            for ex_id in range(len(lll_word_loc[sam_id])):
                assert len(lll_word_loc[sam_id][ex_id]) == \
                       len(bout.l_orig_sent[sam_id0])

                sentL = l_sentL[sam_id0].strip() + UNUSED_TOKENS_STR
                ll_ilabel = lll_ilabel[sam_id]
                for ilabels in ll_ilabel:
                    # You can use x.item() to get a Python number
                    # from a torch tensor that has one element
                    if pred_ilabels.sum().item() == 0:
                        break

                    ilabels = [0] * len(get_words(sentL))
                    pred_ilabels = pred_ilabels[:len(sentL.split())].tolist()
                    for k, loc in enumerate(
                            sorted(lll_word_loc[sam_id][ex_id])):
                        ilabels[loc] = pred_ilabels[k]

                    ilabels = ilabels[:-3]
                    # 1: arg1, 2: rel
                    if 1 not in pred_ilabels and 2 not in pred_ilabels:
                        continue  # not a pass

                    str_extags = \
                        ' '.join([ILABEL_TO_EXTAG[i] for i in ilabels])
                    lines.append(str_extags)

                word_id0 += 1
                ex_id0 += 1
                if ex_id0 == len(bout.l_orig_sent[sam_id0]):
                    ex_id0 = 0
                    sam_id0 += 1

        lines.append('\n')
        assert self.pred_fname
        with open(PREDICTIONS_DIR + "/" + self.pred_fname +
                  "-extags.txt", "w") as f:
            f.writelines(lines)
# NOTE
# run.prepare_test_dataset() never used
# def prepare_test_dataset(hparams, model, sentences, orig_sentences,
#                          sentence_indices_list):
#     label_dict = {0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
#                   4: 'LOC', 5: 'TYPE'}
#
#     lines = []
#
#     outputs = model.outputs
#
#     idx1, idx2, idx3 = 0, 0, 0
#     count = 0
#     for i in range(len(sentence_indices_list)):
#         if len(sentence_indices_list[i]) == 0:
#             sentence = orig_sentences[i].split('[unused1]')[0].strip().split()
#             sentence_indices_list[i].append(list(range(len(sentence))))
#
#         for j in range(len(sentence_indices_list[i])):
#             try:
#                 assert len(sentence_indices_list[i][j]) == len(
#                     outputs[idx1]['meta_data'][
#                         idx2].strip().split()), ipdb.set_trace()
#             except:
#                 ipdb.set_trace()
#             sentence = outputs[idx1]['meta_data'][
#                            idx2].strip() + ' [unused1] [unused2] [unused3]'
#             assert sentence == sentences[idx3]
#             original_sentence = orig_sentences[i]
#             predictions = outputs[idx1]['predictions'][idx2]
#
#             all_extractions, all_str_labels, len_exts = [], [], []
#             for prediction in predictions:
#                 if prediction.sum().item() == 0:
#                     break
#
#                 if hparams.rescoring != 'others':
#                     lines.append(original_sentence)
#
#                 labels = [0] * len(original_sentence.strip().split())
#                 prediction = prediction[:len(sentence.split())].tolist()
#                 for idx, value in enumerate(sorted(sentence_indices_list[i][j])):
#                     labels[value] = prediction[idx]
#                 labels[-3:] = prediction[-3:]
#                 str_labels = ' '.join([label_dict[x] for x in labels])
#                 if hparams.rescoring == 'first':
#                     lines.append(str_labels)
#                 elif hparams.rescoring == 'max':
#                     for _ in range(5):
#                         lines.append(str_labels)
#                 elif hparams.rescoring == 'others':
#                     all_str_labels.append(str_labels)
#                     labels_3 = np.array(labels[:-3])
#                     extraction = ' '.join(np.array(original_sentence.split())[
#                                               np.where(labels_3 != 0)])
#                     all_extractions.append(extraction)
#                     len_exts.append(len(extraction.split()))
#                 else:
#                     assert False
#
#             if hparams.rescoring == 'others':
#                 for ext_i, extraction in enumerate(all_extractions):
#                     other_extractions = ' '.join(
#                         all_extractions[:ext_i] + all_extractions[ext_i + 1:])
#                     other_len_exts = sum(len_exts[:ext_i]) + sum(
#                         len_exts[ext_i + 1:])
#                     input = original_sentence + ' ' + other_extractions
#                     lines.append(input)
#                     output = all_str_labels[ext_i] + ' ' + ' '.join(
#                         ['NONE'] * other_len_exts)
#                     lines.append(output)
#
#             idx3 += 1
#             idx2 += 1
#             if idx2 == len(outputs[idx1]['meta_data']):
#                 idx2 = 0
#                 idx1 += 1
#
#     lines.append('\n')
#     return lines
