from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import warnings

import os
import math
import shutil
from glob import iglob
from time import time
from Model import *
from SaxDataLoaderTool import *
from sax_utils import *
from Params import *
from transformers import AutoTokenizer
import io

# from rescore import rescore

"""

Note that `pytorch_lightning`, used by Openie6, is now deprecated; it has 
been superceeded by the new package called simply `lightning`. I've been 
using lightning 2.1.0. 

on_test_epoch_end() and on_validation_epoch_end(), which SentenceAx uses, 
have only been available in `lightining` since version 2.0.1 (released Feb 
2023).

Refs.:
https://github.com/Lightning-AI/lightning/releases
https://stackoverflow.com/questions/70790473/pytorch-lightning-epoch-end-validation-epoch-end.

"""
check_module_version("lightning", "2.0.1")

set_seed(SEED)
print("SEED=", SEED)


class ActionConductor:
    """
    Similar to Openie6.run.py
    NOTE Openie6.run.prepare_test_dataset() never used

    This method executes various actions when you call its method run().
    run() calls the action methods: 1. train(), 2. resume( ), 3. test(),
    4. predict(), 5. splitpredict(). All other methods in this class are
    called internally by those 5 action methods. Actions can be combined.
    For example, action train_test calls train() first and test() second.

    Note that 4 different Model instances are created by this class: for
    ttt= train, tune, test, and for predict.

    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    checkpoint_callback: ModelCheckpoint
    decode: function
        this is just a method of the AutoTokenizer class. It transforms a
        list of icode integers into text.
    dloader_tool: SaxDataLoaderTool
    encode: function
        this is just a method of the AutoTokenizer class. It transforms text
        into a list of icode integers.
    has_cuda: bool
    pad_icode: int
        For BERT models, this is 0
    params: Param
        class containing parameters
    tags_test_fp: str
        this is the file path to either a cctaggs or an exctags file,
        depending on whether params.task equals "cc" or "ex". These
        samples are used for testing.
    tags_train_fp: str
        this is the file path to either a cctaggs or an exctags file,
        depending on whether params.task equals "cc" or "ex". These
        samples are used for training.
    tags_tune_fp: str
        this is the file path to either a cctaggs or an exctags file,
        depending on whether params.task equals "cc" or "ex". These
        samples are used for tuning (== validation).
    verbose: bool

    """

    def __init__(self, params, save=True, verbose=False):
        """
        This constructor creates new instances of the following classes:
        ModelCheckpoint, AutoTokenizer, SaxDataLoaderTool.

        Parameters
        ----------
        params: Params
        save: bool
            save= True iff checkpoints (i.e., weights) will be saved after
            training. This is almost always True.
        verbose: bool

        """
        self.params = params
        self.verbose = verbose
        self.has_cuda = torch.cuda.is_available()
        warnings.filterwarnings('ignore')

        if self.params.task == 'cc':
            self.tags_train_fp = CCTAGS_TRAIN_FP
            self.tags_tune_fp = CCTAGS_TUNE_FP
            self.tags_test_fp = CCTAGS_TEST_FP
        elif self.params.task == 'ex':
            self.tags_train_fp = EXTAGS_TRAIN_FP
            self.tags_tune_fp = EXTAGS_TUNE_FP
            self.tags_test_fp = EXTAGS_TEST_FP

        if save:
            self.checkpoint_callback = self.get_new_checkpoint_callback()
        else:
            self.checkpoint_callback = None

        do_lower_case = ('uncased' in self.params.d["model_str"])
        self.auto_tokenizer = AutoTokenizer.from_pretrained(
            params.d["model_str"],
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
        self.pad_icode = self.encode(self.auto_tokenizer.pad_token)[1]

        self.dloader_tool = SaxDataLoaderTool(params,
                                              self.auto_tokenizer,
                                              self.tags_train_fp,
                                              self.tags_tune_fp,
                                              self.tags_test_fp)

        if 'predict' not in params.action:
            # ttt dloaders not used for action="predict", "splitpredict"
            # for those actions, only a predict dloader is used.
            self.dloader_tool.set_all_ttt_dataloaders()
            # always set dataloader before constructing a Model instance

    def get_new_checkpoint_callback(self):
        """
        This method returns an instance of class ModelCheckpoint. That class
        saves checkpoint files (weights) at the end of each epoch.
        `save_top_k=N` means it will keep the N checkpoints with the highest
        `epoch_acc` (epoch accuracy) and delete the rest.

        Returns
        -------
        ModelCheckpoint

        """
        # epoch and epoch_acc known by ModelCheckPoint instance
        # str "epoch_acc"  entered via `monitor` variable.
        return ModelCheckpoint(
            dirpath=f"{WEIGHTS_DIR}/{self.params.task}_model",
            filename='{epoch:02d}_{epoch_acc:.3f}',
            verbose=True,
            monitor='epoch_acc',
            mode='max',
            save_top_k=self.params.d["save_k"])

    def get_all_checkpoint_fp(self):
        """
        similar to Openie6.run.get_checkpoint_path().

        This method returns a list of all the checkpoint file paths,
        in inverse chronological order (latest, most recent first).

        There might be more than one checkpoint (see params.d["save_k"]).
        More than one checkpoint only used by self.test() and those who call
        it. resume() uses the latest checkpoint only.

        Returns
        -------
        list[str]

        """
        paths = iglob(f"{WEIGHTS_DIR}/{self.params.task}_model/*.ckpt")
        # latest first in list
        return sorted(paths, key=os.path.getctime, reverse=True)

    def get_latest_checkpoint_fp(self):
        """
        This method returns the latest (most recent) checkpoint file path.

        Returns
        -------
        str

        """
        return self.get_all_checkpoint_fp()[0]

    def delete_all_checkpoints(self, fname_exceptions=None):
        """

        Parameters
        ----------
        fname_exceptions: list[str]|None

        Returns
        -------
        None

        """
        delete_all_files(f"{WEIGHTS_DIR}/{self.params.task}_model",
                         ending=".ckpt",
                         fname_exceptions=fname_exceptions)

    def get_new_TB_logger(self, name):
        """
        similar to Openie6.run.get_logger()

        This method returns a TB (TensorBoard) logger. We start a new logger
        everytime we start a new Trainer. The current log file will have no
        number suffix. Retired ones will.

        Parameters
        ----------
        name: str

        Returns
        -------
        TensorBoardLogger

        """
        prefix = get_task_logs_dir(self.params.task) + f'/{name}'
        if os.path.exists(prefix):
            fps = iglob(prefix + '_*')
            num_numbered_logs = len(list(fps))
            new_id = num_numbered_logs + 1
            print('Retiring current log file by changing its name')
            print(shutil.move(prefix, prefix + f'_{new_id}'))
        logger = TensorBoardLogger(
            save_dir=LOGS_DIR,
            name=self.params.task,
            version=name + '.part')
        return logger

    def get_new_trainer(self, logger, use_minimal):
        """
        This method returns a Trainer instance.

        Parameters
        ----------
        logger: TensorBoardLogger | None
        use_minimal: bool
            use_minimal=True gives a trainer more default behavior.

        Returns
        -------
        Trainer

        """
        # num_steps_per_epoch specified only for quick testing.
        # Set to None to let Trainer decide maximum
        num_steps = self.params.d["num_steps_per_epoch"]
        if use_minimal:
            trainer = Trainer(
                logger=logger,
                num_sanity_val_steps=0,
                limit_train_batches=num_steps,
                limit_val_batches=num_steps,
                limit_test_batches=num_steps
            )
        else:
            trainer = Trainer(
                accumulate_grad_batches=self.params.d[
                    "accumulate_grad_batches"],
                callbacks=self.checkpoint_callback,
                enable_progress_bar=True,
                gradient_clip_val=self.params.d["gradient_clip_val"],
                logger=logger,
                max_epochs=self.params.d["num_epochs"],
                min_epochs=self.params.d["num_epochs"],
                num_sanity_val_steps=0,
                # gpus=no longer used
                # use_tpu=no longer used,
                # train_percent_check=,
                # track_grad_norm= no longer used
                limit_train_batches=num_steps,
                limit_val_batches=num_steps,
                limit_test_batches=num_steps
            )
        return trainer

    def update_params(self, checkpoint_fp):
        """
        similar to Openie6.run.test() and data.override_args().

        This method loads parameters from the checkpoint file and inserts
        them into the Params instance self.params.

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

        This method does the training. It creates instances of Model and a
        Trainer. Then it asks the trainer to fit the model. Finally,
        it creates log file and stores it in either the logs/ex/train or
        logs/cc/train directories.

        train() and resume() are the only action methods that call fit()


        Returns
        -------
        None

        """
        # train is the only action that doesn't require update_params()

        model = Model(self.params,
                      self.auto_tokenizer,
                      verbose=self.verbose,
                      name="train")
        trainer = self.get_new_trainer(self.get_new_TB_logger("train"),
                                       use_minimal=False)
        trainer.fit(
            model,
            train_dataloaders=self.dloader_tool.train_dloader,
            val_dataloaders=self.dloader_tool.tune_dloader)
        tdir = get_task_logs_dir(self.params.task)
        shutil.move(tdir + '/train.part',
                    tdir + '/train')

    def resume(self):
        """
        similar to Openie6.run.resume()

        This method resumes the training after an interruption. It uses the
        latest checkpoint to retrieve the weights and other params close to
        the ones when it was halted. It creates instances of Model and a
        Trainer. Then it asks the trainer to fit the model. Finally,
        it creates log file and stores it in either the logs/ex/resume or
        logs/cc/resume directories.


        Returns
        -------
        None

        """
        checkpoint_fp = self.get_latest_checkpoint_fp()
        # train is the only action that doesn't require
        # update_params() because it is called first
        self.update_params(checkpoint_fp)
        model = Model(self.params,
                      self.auto_tokenizer,
                      verbose=self.verbose,
                      name="resume")
        trainer = self.get_new_trainer(self.get_new_TB_logger("resume"),
                                       use_minimal=False)
        trainer.fit(
            model,
            train_dataloaders=self.dloader_tool.train_dloader,
            val_dataloaders=self.dloader_tool.tune_dloader,
            ckpt_path=checkpoint_fp)  # only if resuming
        tdir = get_task_logs_dir(self.params.task)
        shutil.move(tdir + '/resume.part',
                    tdir + '/resume')

    def test(self):
        """
        similar to Openie6.run.test()

        Note: this method self.test() is different from
        Trainer.test() which is called inside this method.

        This method does testing. It creates an instance of Model. It then
        goes down the list of checkpoints and creates a Trainer for each
        one. This trainer is used to call trainer.test(), instead of
        trainer.fit( ) as done in self.train(). trainer.test( )
        scores the test data with the weights of that checkpoint file. test
        accuracies for each checkpoint are saved in a file logs/ex/test.txt
        or logs/cc/test.txt.


        Returns
        -------
        None

        """
        checkpoint_paths = self.get_all_checkpoint_fp()
        if 'train' not in self.params.action:
            # here parameters are updated from the latest checkpoint. If
            # train() was called first, then update_params() is unnecessary
            # because no checkpoints exists yet.
            self.update_params(self.get_latest_checkpoint_fp())

        model = Model(self.params,
                      self.auto_tokenizer,
                      verbose=self.verbose,
                      name="test")

        # ex_sent_to_sent and cc_sent_to_word only stored in one place
        # if self.params.task == "ex" and self.ex_sent_to_sent:
        #     model.metric.sent_to_sent = self.ex_sent_to_sent
        # if self.params.task == "cc" and self.cc_sent_to_words:
        #     model.metric.sent_to_words = self.cc_sent_to_words

        tdir = get_task_logs_dir(self.params.task)
        with open(tdir + '/test.txt', "w") as test_f:
            logger = self.get_new_TB_logger("test")
            # might be more than one checkpoint if keep best 2, 3 etc epochs
            for checkpoint_fp in checkpoint_paths:
                trainer = self.get_new_trainer(logger,
                                               use_minimal=True)
                # trainer.fit() and trainer.test() are different
                test_dloader = self.dloader_tool.test_dloader
                trainer.test(model=model,
                             dataloaders=test_dloader,
                             ckpt_path=checkpoint_fp)
                scores_d = model.scores_epoch_end_d
                test_f.write(f'{checkpoint_fp}\t{scores_d}\n')
                # note test_f created outside loop.
                # refresh/clear/flush test_f after each write
                test_f.flush()
        shutil.move(tdir + '/test.part',
                    tdir + '/test')

    def predict(self, pred_in_fp):
        """
        similar to Openie6.run.predict()

        This method does prediction. It creates instances of Model and
        Trainer. The trainer is used to call trainer.test() instead of
        trainer.fit( ) as done in self.train(). trainer.test() is
        used with the test data in self.test(). But here it is
        used with the data at `pred_in_fp` (a file of sentences, one per
        line). This method times how long it takes to predict.

        This method does not write a prediction file! It stores the
        predictions in the `model.l_batch_m_out` variable, but that info is
        not written into a file unless you run self.splitpredict(
        ). Hence, running this action alone, instead of within splitpredict(
        ), is not very useful, except for timing.


        Parameters
        ----------
        pred_in_fp: str
            This file has no tags or ilabels. Only one osent per line for 
            each sample.

        Returns
        -------
        Model

        """
        checkpoint_fp = get_best_checkpoint_path(self.params.task)

        # assert list(self.get_all_checkpoint_fp()) == [checkpoint_fp]

        self.update_params(checkpoint_fp)
        self.dloader_tool.set_predict_dataloader(pred_in_fp)
        # always set dataloader before constructing a Model instance
        model = Model(self.params,
                      self.auto_tokenizer,
                      verbose=self.verbose,
                      name="pred")

        # No
        # model.metric.sent_to_sent = self.ex_sent_to_sent
        # model.metric.sent_to_words = self.cc_sent_to_words

        trainer = self.get_new_trainer(logger=None,
                                       use_minimal=True)
        start_time = time()
        # model.all_sentences = all_sentences # never used
        trainer.test(
            model,
            dataloaders=self.dloader_tool.predict_dloader,
            ckpt_path=checkpoint_fp)
        end_time = time()
        minutes = (end_time - start_time) / 60
        print(f'Total Time taken = {minutes : 2f} minutes')
        return model

    @staticmethod
    def write_extags_file_from_preds(
            l_osentL,  # Openi6.orig_sentences
            l_ccsentL,  # Openie6.sentences
            out_fp,
            model):
        """
        similar to Openie6.run.get_labels()

        This method is called by `self.splitpredict_for_ex()`.

        It writes an extags file at `out_fp` based on the predictions
        stored inside `model.l_batch_m_out`.


        Parameters
        ----------
        l_osentL: list[str]
        l_ccsentL: list[str]
        out_fp: str
        model: Model


        Returns
        -------
        None

        """
        l_m_out = model.l_batch_m_out

        lines = []
        batch_id0 = 0  # similar to Openie6.idx1
        sam_id0 = 0  # similar to Openie6.idx2
        cum_sam_id0 = 0  # similar to Openie6.idx3
        # isam similar to Openie6.i
        # jccsent similar to Openie6.j

        # lll_cc_spanned_loc is similar to
        # sentence_indices_list, model.all_sentence_indices_conj
        lll_cc_spanned_loc = \
            model.lll_cc_spanned_loc
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
                # similar to Openie6.predictions
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
                    # 1: ARG1, 2: REL
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
        with open(out_fp, "w") as f:
            f.writelines(lines)

    def splitpredict_for_cc(self, pred_in_fp):
        """
        This is a private method for self.splitpredict().

        This method reads a file at `pred_in_fp` with the sentences (one
        sentence per line) that one wants to split.

        Parameters
        ----------
        pred_in_fp: str    
            This file has no tags or ilabels. Only one osent per line for 
            each sample.

        Returns
        -------
        list[str], list[str]
            l_osentL, l_ccsentL

        """

        def repeated_splitpredict_for_cc():
            cc_words = ll_cc_spanned_word[sample_id]
            if len(l_pred_sent) == 1:
                l_osentL.append(redoL(l_pred_sent[0]))
                model.ex_sent_to_sent[l_pred_sent[0]] = \
                    l_pred_sent[0]

                l_ccsentL.append(redoL(l_pred_sent[0]))
                # added to Openie6. Makes no difference because
                # model.cc_sent_to_words is never used
                model.cc_sent_to_words[l_pred_sent[0]] = cc_words
            elif len(l_pred_sent) > 1:
                l_osentL.append(redoL(l_pred_sent[0]))
                for sent in l_pred_sent[1:]:
                    model.ex_sent_to_sent[sent] = l_pred_sent[0]
                    l_ccsentL.append(redoL(sent))
                # added to Openie6. Makes no difference because
                # model.cc_sent_to_words is never used
                model.cc_sent_to_words[l_pred_sent[0]] = cc_words
            else:
                assert False

        self.params.d["best_checkpoint_fp"] = CC_BEST_WEIGHTS_FP
        # For task="cc", Openie6 uses large-cased for train_test() and
        # base-cased for splitpredict(). Both are cased, but larger-cased is
        # larger than base-cased. For task="ex", Openie6 uses base-cased
        # throughout
        self.params.d["model_str"] = 'bert-base-cased'
        self.params.d["action"] = self.params.action = 'predict'

        model = self.predict(pred_in_fp)
        model.cc_sent_to_words = {}
        model.ex_sent_to_sent = {}

        l_cc_pred_str = model.l_cc_pred_str
        lll_cc_spanned_loc = model.lll_cc_spanned_loc
        assert len(l_cc_pred_str) == len(lll_cc_spanned_loc)
        ll_cc_spanned_word = model.ll_cc_spanned_word

        l_ccsentL = []  # Openie6.sentences
        l_osentL = []  # Openie6.orig_sentences

        if not pred_in_fp:
            for sample_id, pred_str in enumerate(l_cc_pred_str):
                # similar to Openie6.example_sentences
                l_pred_sent = pred_str.strip('\n').split('\n')
                repeated_splitpredict_for_cc()
            # l_ccsentL.append("\n")
        # count = 0
        # for l_spanned_loc in ll_cc_spanned_loc:
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
            for line in lines:
                if len(line) > 0:
                    # similar to Openie6.example_sentences
                    l_pred_sent = line.strip().split("\n")
                    repeated_splitpredict_for_cc()

        return model, l_osentL, l_ccsentL

    def splitpredict_for_ex(self,
                            l_osentL,
                            l_ccsentL,
                            pred_in_fp,
                            delete_ccsents_file=True):
        """
        This is a private method for self.splitpredict().

        This method writes 
        
        1. an extags file at f'{pred_in_fp.replace(".txt", 
        "")}_extags_out.txt' with the predicted extags.
        
        2. an ss (simple sents) file at f'{pred_in_fp.replace(".txt", 
        "")}_ss_out.txt' with the predicted simple sentences.

        Parameters
        ----------
        l_osentL: list[str]
        l_ccsentL: list[str]
        pred_in_fp: str
            This file has no tags or ilabels. Only one osent per line for 
            each sample.
        delete_ccsents_file: bool

        Returns
        -------
        None

        """
        self.params.d["best_checkpoint_fp"] = EX_BEST_WEIGHTS_FP
        self.params.d["model_str"] = 'bert-base-cased'

        # temporary file, to be deleted after it is used by predict(). This
        # file is not used in Openie6, but is needed in SentenceAx, so as to
        # get the appropriate input for self.predict()
        in_fp = f'{pred_in_fp.replace(".txt", "")}_ccsents.txt'
        with open(in_fp, "w", encoding="utf-8") as f:
            f.write("\n".join(l_ccsentL))

        model = self.predict(in_fp)

        if delete_ccsents_file:
            os.remove(in_fp)

        # Does same thing as Openie6.run.get_labels()
        extags_out_fp = f'{pred_in_fp.replace(".txt", "")}_extags_out.txt'
        ActionConductor.write_extags_file_from_preds(l_osentL,
                                                     l_ccsentL,
                                                     extags_out_fp,
                                                     model)

        allen_fp = f"{VAL_OUT_DIR}/allen.txt"

        # rescored_allen_file = rescore(
        #     allen_fp,
        #     model_dir=,
        #     batch_size=256)

        ss_out_fp = f'{pred_in_fp.replace(".txt", "")}_ss_out.txt'
        print('Predictions written to ' + ss_out_fp)
        al_tool = AllenTool(allen_fp)
        al_tool.write_allen_alternative_file(ss_out_fp, ftype="ss")

    def splitpredict(self, pred_in_fp):
        """
        similar to Openie6.run.splitpredict()

        This method calls the 2 private methods: self.splitpredict_for_cc(), 
        and self.splitpredict_for_ex() in that order.

        Parameters
        ----------
        pred_in_fp: str
            This file has no tags or ilabels. Only one osent per line for 
            each sample.

        Returns
        -------
        None

        """

        # def splitpredict(params_d, checkpoint_callback,
        #                  train_dataloader, val_dataloader, test_dataloader,
        #                  all_sentences):

        self.params.d["task"] = self.params.task = "cc"
        l_osentL, l_ccsentL = self.splitpredict_for_cc(pred_in_fp)

        self.params.d["task"] = self.params.task = "ex"
        self.splitpredict_for_ex(l_osentL,
                                 l_ccsentL,
                                 pred_in_fp)

    def run(self, pred_in_fp=None):
        """
        This method is the only non-private method for the entire class.
        Users should never have to invoke any method of this class other
        than this one. The method can run a combination of actions joined by
        an underscore. For example, if params.task = "ex" and params.action
        = "train_test", then self.run() will run self.train() first and
        self.test() second.

        Parameters
        ----------
        pred_in_fp: str
            This file has no tags or ilabels. Only one osent per line for 
            each sample.

        Returns
        -------
        None

        """
        for process in self.params.action.split('_'):
            if process in ["predict", "splitpredict"]:
                assert pred_in_fp
                getattr(self, process)(pred_in_fp)
            else:
                getattr(self, process)()


if __name__ == "__main__":
    """
        pid, task, action   
        0. "", ""
        1. ex, train_test
        2. ex, test  (appears twice in Openie6 readme)
        3. ex, predict (appears twice in Openie6 readme)
        4. ex, resume
        5. cc, train_test
        6. ex, splitpredict (appears twice in Openie6 readme)

    """


    def main(pid):
        params = Params(pid)
        params.d["refresh_cache"] = False
        params.d["gpus"] = 0
        params.d["num_epochs"] = 1
        params.d["num_steps_per_epoch"] = 1
        params.d["model_str"] = "bert-base-cased"
        params.describe_self()
        conductor = ActionConductor(params, verbose=True)
        print("checkpoints:", conductor.get_all_checkpoint_fp())
        conductor.run()
        print("checkpoints:", conductor.get_all_checkpoint_fp())

    # main(1)
    # main(5)
