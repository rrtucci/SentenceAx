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
    
    
    # NOTE
    # run.prepare_test_dataset() never used

    Refs:
    https://spacy.io/usage/spacy-101/

    """

    def __init__(self, params, save=True):
        """
        A new
        ModelCheckpoint
        AutoTokenizer,
        SaxDataLoader,
        TensorBoardLogger

        is created everytime this constructor is called


        """
        self.params = params
        self.has_cuda = torch.cuda.is_available()
        warnings.filterwarnings('ignore')

        if self.params.task == 'cc':
            self.train_fp = CCTAGS_TRAIN_FP
            self.tune_fp = CCTAGS_TUNE_FP
            self.test_fp = CCTAGS_TEST_FP
            self.pred_fp = CC_PRED_FP
        elif self.params.task == 'ex':
            self.train_fp = EXTAGS_TRAIN_FP
            self.tune_fp = EXTAGS_TUNE_FP
            self.test_fp = EXTAGS_TEST_FP
            self.pred_fp = EX_PRED_FP

        if save:
            self.checkpoint_callback = self.get_checkpoint_callback()
        else:
            self.checkpoint_callback = None

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
        self.pad_icode = self.encode(self.auto_tokenizer.pad_token)

        self.dloader = SaxDataLoader(self.auto_tokenizer,
                                     self.pad_icode,
                                     self.train_fp,
                                     self.tune_fp,
                                     self.test_fp)

        self.pred_out_fp = PRED_OUT_FP
        self.pred_in_fp = PRED_IN_FP
        self.rescore_in_fp = RESCORE_IN_FP
        self.rescore_out_fp = RESCORE_OUT_FP

        self.model = None

    def get_checkpoint_callback(self):
        """

        Returns
        -------
        ModelCheckpoint

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

    def get_all_checkpoint_fp(self):
        """
        similar to Openie6.run.get_checkpoint_fp()

        more than one checkpoint only used by self.test()
        and those who call it.

        Returns
        -------
        list[str]

        """
        if "suggested_checkpoint_fp" in self.params.d:
            return [self.params.d["suggested_checkpoint_fp"]]

        else:
            return glob(WEIGHTS_DIR + '/*.ckpt')

    def get_checkpoint_fp(self):
        """

        Returns
        -------
        str

        """
        all_paths = self.get_all_checkpoint_fp()
        assert len(all_paths) == 1
        return all_paths[0]

    def get_logger(self, ttt):
        """
        similar to Openie6.run.get_logger()

        Logger depends on params.task and params.mode.
        Start new logger everytime start a new trainer.

        Parameters
        ----------

        Returns
        -------
        TensorBoardLogger

        """

        # the current log file will have no number prefix,
        # stored ones will.
        assert os.path.exists(
            self.params.log_dir() + "/" + ttt)
        num_numbered_logs = len(
            list(glob(self.params.log_dir() + f'/{ttt}_*')))
        new_id = num_numbered_logs + 1
        print('Retiring current log file by changing its name')
        print(shutil.move(
            self.params.log_dir() + f'/{ttt}',
            self.params.log_dir() + f'/{ttt}_{new_id}'))
        logger = TensorBoardLogger(
            save_dir=WEIGHTS_DIR,
            name='logs',
            version=ttt + '.part')
        return logger

    def get_trainer(self, logger, checkpoint_fp, use_minimal):
        """

        Parameters
        ----------
        logger: TensorBoardLogger | None
        checkpoint_fp: str | None
        use_minimal: bool

        Returns
        -------
        Trainer

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
        #     resume_from_checkpoint = checkpoint_fp,
        #     show_progress_bar = True,
        #     track_grad_norm = params_d.track_grad_norm,
        #     train_percent_check = params_d.train_percent_check,
        #     use_tpu = params_d.use_tpu,
        #     val_check_interval = params_d.val_check_interval)

        if use_minimal:
            trainer = Trainer(
                gpus=self.params.d["gpus"],
                logger=logger,
                resume_from_checkpoint=checkpoint_fp)
        else:
            trainer = Trainer(
                accumulate_grad_batches= \
                    int(self.params.d["accumulate_grad_batches"]),
                checkpoint_callback=self.checkpoint_callback,
                logger=logger,
                max_epochs=self.params.d["epochs"],
                min_epochs=self.params.d["epochs"],
                resume_from_checkpoint= \
                    checkpoint_fp if self.params.mode == "resume" else None,
                show_progress_bar=True,
                **self.params.d)
        return trainer

    def update_params(self, checkpoint_fp):
        """
        similar to Openie6.run.test() and data.override_args()


        Parameters
        ----------
        checkpoint_fp: str

        Returns
        -------
        None

        """
        if self.has_cuda:
            loaded_params = torch.load(checkpoint_fp)["params"]
        else:
            map_loc = torch.device('cpu')
            loaded_params = torch.load(
                checkpoint_fp, map_location=map_loc)["params"]

        self.params.d = merge_dicts(loaded_params,
                                    default_d=self.params.d)

    def train(self):
        """
        similar to Openie6.run.train()

        trainer.fit()

        Returns
        -------

        """
        # train is the only mode that doesn't require update_params()
        self.model = Model(self.params.d, self.auto_tokenizer)
        trainer = self.get_trainer(self.get_logger("train"),
                                   checkpoint_fp=None,
                                   use_minimal=False)
        trainer.fit(
            self.model,
            train_dataloader=self.dloader.get_ttt_dataloaders("train"),
            val_dataloaders=self.dloader.get_ttt_dataloaders("tune"))
        shutil.move(WEIGHTS_DIR + f'/logs/train.part',
                    WEIGHTS_DIR + f'/logs/train')

    def resume(self):
        """
        similar to Openie6.run.resume()

        trainer.fit()


        Returns
        -------
        None

        """
        checkpoint_fp = self.get_checkpoint_fp()
        # train is the only mode that doesn't require
        # update_params() because it is called first
        self.update_params(checkpoint_fp)
        self.model = Model(self.params.d, self.auto_tokenizer)
        trainer = self.get_trainer(self.get_logger("tune"),
                                   checkpoint_fp,
                                   use_minimal=False)
        trainer.fit(
            self.model,
            train_dataloader=self.dloader.get_ttt_dataloaders("train"),
            val_dataloaders=self.dloader.get_ttt_dataloaders("tune"))
        shutil.move(WEIGHTS_DIR + '/logs/resume.part',
                    WEIGHTS_DIR + '/logs/resume')

    def test(self):
        """
        similar to Openie6.run.test()
        trainer.test()

        Returns
        -------
        None

        """
        checkpoint_fp = self.get_checkpoint_fp()
        if 'train' not in self.params.mode:
            # train is the only mode that doesn't require
            # update_params() because it is called first
            self.update_params(checkpoint_fp)

        self.model = Model(self.params.d, self.auto_tokenizer)
        # if self.params.task == "ex" and self.ex_sent_to_sent:
        #     self.model.metric.sent_to_sent = self.ex_sent_to_sent
        # if self.params.task == "cc" and self.cc_sent_to_words:
        #     self.model.metric.sent_to_words = self.cc_sent_to_words

        with open(WEIGHTS_DIR + '/logs/test.txt', "w") as test_f:
            logger = self.get_logger("test")
            # one checkpoint at end of each epoch
            for checkpoint_fp in self.get_all_checkpoint_fp():
                trainer = self.get_trainer(logger,
                                           checkpoint_fp,
                                           use_minimal=True)
                # trainer.fit() and trainer.test() are different
                trainer.test(
                    self.model,
                    test_dataloaders=self.dloader.get_ttt_dataloaders("test"))
                eval_epoch_end_d = self.model.eval_epoch_end_d
                test_f.write(f'{checkpoint_fp}\t{eval_epoch_end_d}\n')
                # note test_f created outside loop.
                # refresh/clear/flush test_f after each write
                test_f.flush()
        shutil.move(WEIGHTS_DIR + f'/logs/test.part',
                    WEIGHTS_DIR + f'/logs/test')

    def predict(self):
        """
        similar to Openie6.run.predict()

        trainer.test()

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
            self.params.d["suggested_checkpoint_fp"] = CC_FIN_WEIGHTS_FP
        if self.params.task == 'ex':
            self.params.d["suggested_checkpoint_fp"] = EX_FIN_WEIGHTS_FP

        checkpoint_fp = self.get_checkpoint_fp()
        self.update_params(checkpoint_fp)
        self.model = Model(self.params.d, self.auto_tokenizer)

        # if self.params.task == "ex" and self.ex_sent_to_sent:
        #     self.model.metric.sent_to_sent = self.ex_sent_to_sent
        # elif self.params.task == "cc" and self.cc_sent_to_words:
        #     self.model.metric.sent_to_words = self.cc_sent_to_words

        logger = None
        trainer = self.get_trainer(logger,
                                   checkpoint_fp,
                                   use_minimal=True)
        start_time = time()
        # self.model.all_sentences = all_sentences
        trainer.test(
            self.model,
            test_dataloaders=test_dloader if
            test_dloader else self.dloader.get_ttt_dataloaders("test"))
        end_time = time()
        minutes = (end_time - start_time) / 60
        print(f'Total Time taken = {minutes : 2f} minutes')

    def splitpredict_do_cc(self, pred_in_fp):
        """
        no trainer

        Parameters
        ----------
        pred_in_fp

        Returns
        -------

        """
        if not pred_in_fp:
            self.params.d["suggested_checkpoint_fp"] = CC_FIN_WEIGHTS_FP
            self.params.d["model_str"] = 'bert-base-cased'
            self.params.d["mode"] = self.params.mode = 'predict'
            self.predict()
            l_pred_str = self.l_cc_pred_str
            ll_spanned_loc = self.ll_cc_spanned_loc
            assert len(l_pred_str) == len(ll_spanned_loc)
            ll_spanned_word = self.ll_cc_spanned_word

            l_pred_sentL = []
            l_osentL = []
            for sample_id, pred_str in enumerate(l_pred_str):
                l_pred_sent = pred_str.strip('\n').split('\n')

                # not done when reading from split_pred_fp
                words = ll_spanned_word[sample_id]
                self.cc_sent_to_words[l_pred_sent[??]] = words

                l_osentL.append(redoL(l_pred_sent[0]))
                for sent in l_pred_sent:
                    self.ex_sent_to_words[sent] = l_pred_sent[0]
                    l_pred_sentL.append(redoL(sent))
            # this not done when reading from split_pred_fp
            # l_osentL.append('\

            # Never used:
            # count = 0
            # for l_spanned_loc in ll_spanned_loc:
            #     if len(l_spanned_loc) == 0:
            #         count += 1
            #     else:
            #         count += len(l_spanned_loc)
            # assert count == len(l_osentL) - 1

        else:
            with open(pred_in_fp, "r") as f:
                content = f.read()
            content = content.replace("\\", "")
            lines = content.split('\n\n')

            l_pred_sentL = []
            l_osentL = []
            for line in lines:
                if len(line) > 0:
                    l_pred_sent = line.strip().split('\n')
                    l_osentL.append(l_pred_sent[0] + UNUSED_TOKENS_STR)
                    for sent in l_pred_sent:
                        self.ex_sent_to_words[sent] = l_pred_sent[0]
                        l_pred_sentL.append(sent + UNUSED_TOKENS_STR)

        return l_osentL, l_ccsentL, lll_cc_spanned_loc

    def splitpredict_do_ex(self,
                           pred_out_fp,
                           l_osentL,
                           l_ccsentL,
                           lll_cc_spanned_loc):
        """
        no trainer

        Returns
        -------

        """
        self.params.d["suggested_checkpoint_fp"] = EX_FIN_WEIGHTS_FP
        self.params.d["model_str"] = 'bert-base-cased'
        pred_test_dataloader = self.dloader.get_ttt_dataloaders("test")

        self.predict(test_dloader=pred_test_dataloader)

        with_confis = False
        # Does same thing as Openie6's run.get_labels()
        if self.params.d["write_extags_file"]:
            self.write_extags_file_from_preds(l_osentL,
                                              l_ccsentL,
                                              lll_cc_spanned_loc,
                                              pred_out_fp)

    def splitpredict_do_rescore(self, rescore_in_fp, rescore_out_fp):
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
        rescored = rescore(rescore_in_fp,
                           model_dir=hparams.rescore_model,
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
                                    confidence=math.exp(confidence), index=0)
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
            print('Predictions written to ', rescore_out_fp)
            predictions_f = open(rescore_out_fp, "w")
            predictions_f.write('\n'.join(all_predictions) + '\n')
            predictions_f.close()
        return

    def splitpredict(self):
        """
        similar to Openie6.run.splitpredict()

        calls self.predict() and self.rescore()


        Returns
        -------

        """

        # def splitpredict(params_d, checkpoint_callback,
        #                  train_dataloader, val_dataloader, test_dataloader,
        #                  all_sentences):

        self.cc_sent_to_words = {}
        self.ex_sent_to_sent = {}
        self.params["write_allen_file"] = True

        self.params.d["task"] = self.params.task = "cc"
        l_osentL, l_ccsentL, lll_cc_spanned_loc = \
            self.splitpredict_do_cc(self.pred_in_fp)

        self.params.d["task"] = self.params.task = "ex"
        self.splitpredict_do_ex(self.pred_out_fp,
                                l_osentL,
                                l_ccsentL,
                                lll_cc_spanned_loc)

        if "rescoring" in self.params.d:
            self.rescore(self.rescore_in_fp, self.rescore_out_fp)

    def write_extags_file_from_preds(
            self,
            l_osentL,  # orig_sentences
            l_ccsentL,  # sentences
            lll_cc_spanned_loc,  # sentence_indices_list
            pred_out_fp):
        """
        similar to Openie6.run.get_labels()
        ILABEL_TO_EXTAG={0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
                 4: 'ARG2', 5: 'NONE'}

        called by `splitpredict_do_ex()`


        Parameters
        ----------
        
        l_osentL
        l_ccsentL
        lll_cc_spanned_loc
        pred_out_fp

        Returns
        -------
        None

        """
        l_m_out = self.model.l_batch_m_out

        lines = []
        batch_id0 = 0  # similar to idx1
        sam_id0 = 0  # similar to idx2
        cum_sam_id0 = 0  # similar to idx3
        # isam similar to i
        # jccsent similar to j

        for isam in range(len(lll_cc_spanned_loc)):
            osent = undoL(l_osentL[isam])
            if len(lll_cc_spanned_loc[isam]) == 0:
                lll_cc_spanned_loc[isam].append(list(range(len(osent))))
            lines.append('\n' + osent)
            num_ccsent = len(lll_cc_spanned_loc[isam])
            for jccsent in range(num_ccsent):
                osent = l_m_out[batch_id0].l_osent[sam_id0]
                osentL = redoL(osent)
                osentL_words = get_words(osentL)
                assert len(lll_cc_spanned_loc[isam][jccsent]) == \
                       len(osentL_words)
                assert osentL == l_ccsentL[cum_sam_id0]
                # similar to predictions
                ll_pred_ilabel = \
                    l_m_out[batch_id0].lll_pred_ilabel[sam_id0]
                for l_pred_ilabel in ll_pred_ilabel:
                    # You can use x.item() to get a Python number
                    # from a torch tensor that has one element
                    if l_pred_ilabel.sum().item() == 0:
                        break

                    l_ilabel = [0] * len(osentL_words)
                    l_pred_ilabel = \
                        l_pred_ilabel[:len(osentL_words)].tolist()
                    for k, loc in enumerate(
                            sorted(lll_cc_spanned_loc[isam][jccsent])):
                        l_ilabel[loc] = l_pred_ilabel[k]

                    assert len(l_ilabel) == len(osentL_words)
                    l_ilabel = l_ilabel[:-3]
                    # 1: arg1, 2: rel
                    if 1 not in l_pred_ilabel and 2 not in l_pred_ilabel:
                        continue  # not a pass

                    str_extags = \
                        ' '.join([ILABEL_TO_EXTAG[i] for i in l_ilabel])
                    lines.append(str_extags)

                cum_sam_id0 += 1
                sam_id0 += 1
                if sam_id0 == len(l_m_out[batch_id0].l_osent):
                    sam_id0 = 0
                    batch_id0 += 1

        lines.append('\n')
        with open(pred_out_fp, "w") as f:
            f.writelines(lines)

    def run(self):
        for process in self.params.mode.split('_'):
            globals()[process]()
