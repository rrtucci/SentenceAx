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
from DataLoaderTool import *
from sax_utils import *
from Params import *
from transformers import AutoTokenizer
import io


# from rescore import rescore


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
    
    Often, batch refers to the output.meta_data of loader, but not in
    SentenceAx

    for batch_index, batch in enumerate(loader):
        input, target = batch
    
    
    # NOTE
    # run.prepare_test_dataset() never used

    Refs:
    https://spacy.io/usage/spacy-101/

    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    checkpoint_callback: ModelCheckpoint
    decode: function
    dloader_tool: DataLoaderTool
    encode: function
    has_cuda: bool
    model: Model
    pad_icode: int
    params: Param
    pred_in_fp: str
    pred_out_fp: str
    re_allen_in_fp: str
    re_allen_out_fp: str
    test_fp: str
    train_fp: str
    tune_fp: str

    """

    def __init__(self, params, save=True):
        """
        A new
        ModelCheckpoint
        AutoTokenizer,
        DataLoaderTool,
        TensorBoardLogger

        is created everytime this constructor is called

        Parameters
        ----------
        params: Params
        save: bool



        """
        self.params = params
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

        if save:
            self.checkpoint_callback = self.get_checkpoint_callback()
        else:
            self.checkpoint_callback = None

        do_lower_case = ('uncased' in self.params.d["model_str"])
        self.auto_tokenizer = AutoTokenizer.from_pretrained(
            params.d["model_str"],
            do_lower_case=True,
            use_fast=True,
            data_dir=TTT_CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        # encode() (a.k.a. convert_tokens_to_ids())
        # replaces vocab.stoi() (stoi=string to integer)
        self.encode = self.auto_tokenizer.encode
        # decode()
        # replaces vocab.itos() (itos=integer to string)
        self.decode = self.auto_tokenizer.decode
        self.pad_icode = self.encode(self.auto_tokenizer.pad_token)[1]

        self.dloader_tool = DataLoaderTool(params,
                                           self.auto_tokenizer,
                                           self.train_fp,
                                           self.tune_fp,
                                           self.test_fp)

        self.model = None

    def get_checkpoint_callback(self):
        """

        Returns
        -------
        ModelCheckpoint

        """
        return ModelCheckpoint(
            filepath=WEIGHTS_DIR + "/" + \
                     self.params.task + '_model/{epoch:02d}_{eval_acc:.3f}',
            verbose=True,
            monitor='eval_acc',
            mode='max',
            save_top_k=self.params.d["save_k"],
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
        ttt: str

        Returns
        -------
        TensorBoardLogger

        """

        # the current log file will have no number prefix,
        # stored ones will.
        if os.path.exists(self.params.task_logs_dir() + "/" + ttt):
            num_numbered_logs = len(
                list(glob(self.params.task_logs_dir() + f'/{ttt}_*')))
            new_id = num_numbered_logs + 1
            print('Retiring current log file by changing its name')
            print(shutil.move(
                self.params.task_logs_dir() + f'/{ttt}',
                self.params.task_logs_dir() + f'/{ttt}_{new_id}'))
        logger = TensorBoardLogger(
            save_dir=LOGS_DIR,
            name=self.params.task,
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
                # accumulate_grad_batches=
                # self.params.d["accumulate_grad_batches"],
                checkpoint_callback=self.checkpoint_callback,
                logger=logger,
                max_epochs=self.params.d["epochs"],
                min_epochs=self.params.d["epochs"],
                resume_from_checkpoint=checkpoint_fp if
                self.params.mode == "resume" else None,
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
        None

        """
        # train is the only mode that doesn't require update_params()
        self.model = Model(self.params, self.auto_tokenizer)
        trainer = self.get_trainer(self.get_logger("train"),
                                   checkpoint_fp=None,
                                   use_minimal=False)
        trainer.fit(
            self.model,
            train_dataloader=self.dloader_tool.get_one_ttt_dataloader("train"),
            val_dataloaders=self.dloader_tool.get_one_ttt_dataloader("tune"))
        tdir = self.params.task_logs_dir()
        shutil.move(tdir + '/train.part',
                    tdir + '/train')

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
            train_dataloader=self.dloader_tool.get_one_ttt_dataloader("train"),
            val_dataloaders=self.dloader_tool.get_one_ttt_dataloader("tune"))
        tdir = self.params.task_logs_dir()
        shutil.move(tdir + '/resume.part',
                    tdir + '/resume')

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

        tdir = self.params.task_logs_dir()
        with open(tdir + '/test.txt', "w") as test_f:
            logger = self.get_logger("test")
            # one checkpoint at end of each epoch
            for checkpoint_fp in self.get_all_checkpoint_fp():
                trainer = self.get_trainer(logger,
                                           checkpoint_fp,
                                           use_minimal=True)
                # trainer.fit() and trainer.test() are different
                dloader = self.dloader_tool.get_one_ttt_dataloader("test")
                trainer.test(self.model, test_dataloaders=dloader)
                eval_epoch_end_d = self.model.eval_epoch_end_d
                test_f.write(f'{checkpoint_fp}\t{eval_epoch_end_d}\n')
                # note test_f created outside loop.
                # refresh/clear/flush test_f after each write
                test_f.flush()
        shutil.move(tdir + '/test.part',
                    tdir + '/test')

    def predict(self, pred_in_fp):
        """
        similar to Openie6.run.predict()

        trainer.test()

        Parameters
        ----------
        predict_in_fp: str

        Returns
        -------
        None

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

        # No
        # self.model.metric.sent_to_sent = self.ex_sent_to_sent
        # self.model.metric.sent_to_words = self.cc_sent_to_words

        logger = None
        trainer = self.get_trainer(logger,
                                   checkpoint_fp,
                                   use_minimal=True)
        start_time = time()
        # self.model.all_sentences = all_sentences # never used
        dloader = self.dloader_tool.get_predict_dataloader(pred_in_fp)
        trainer.test(
            self.model,
            test_dataloaders=dloader)
        end_time = time()
        minutes = (end_time - start_time) / 60
        print(f'Total Time taken = {minutes : 2f} minutes')

    def splitpredict_for_cc(self, pred_in_fp):
        """
        no trainer

        reads pred_in_fp

        Parameters
        ----------
        pred_in_fp: str

        Returns
        -------
        list[str], list[str]
            l_osentL, l_ccsentL

        """

        def common_splitpredict_ex():
            cc_words = ll_cc_spanned_word[sample_id]
            if len(l_pred_sent) == 1:
                l_osentL.append(redoL(l_pred_sent[0]))
                self.model.ex_sent_to_sent[l_pred_sent[0]] = \
                    l_pred_sent[0]

                l_ccsentL.append(redoL(l_pred_sent[0]))
                # added
                self.model.cc_sent_to_words[l_pred_sent[0]] = cc_words
            elif len(l_pred_sent) > 1:
                l_osentL.append(redoL(l_pred_sent[0]))
                for sent in l_pred_sent[1:]:
                    self.model.ex_sent_to_sent[sent] = l_pred_sent[0]
                    l_ccsentL.append(redoL(sent))
                # added
                self.model.cc_sent_to_words[l_pred_sent[0]] = cc_words
            else:
                assert False

        self.params.d["suggested_checkpoint_fp"] = CC_FIN_WEIGHTS_FP
        self.params.d["model_str"] = 'bert-base-cased'
        self.params.d["mode"] = self.params.mode = 'predict'
        self.predict("test")
        l_cc_pred_str = self.model.l_cc_pred_str
        lll_cc_spanned_loc = self.model.lll_cc_spanned_loc
        assert len(l_cc_pred_str) == len(lll_cc_spanned_loc)
        ll_cc_spanned_word = self.model.ll_cc_spanned_word

        l_ccsentL = []  # sentences
        l_osentL = []  # orig_sentences

        if not pred_in_fp:
            for sample_id, pred_str in enumerate(l_cc_pred_str):
                # example_sentences
                l_pred_sent = pred_str.strip('\n').split('\n')
                common_splitpredict_ex()
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
                    # example_sentences
                    l_pred_sent = line.strip().split("\n")
                    common_splitpredict_ex()

        return l_osentL, l_ccsentL

    def splitpredict_for_ex(self,
                            l_osentL,
                            l_ccsentL,
                            pred_out_fp,
                            delete_ccsents_file=False):
        """
        no trainer

        writes pred_out_fp

        Parameters
        ----------
        l_osentL: list[str]
        l_ccsentL: list[str]
        pred_out_fp: str

        Returns
        -------
        None

        """
        self.params.d["suggested_checkpoint_fp"] = EX_FIN_WEIGHTS_FP
        self.params.d["model_str"] = 'bert-base-cased'

        in_fp = pred_out_fp.strip(".txt") + "_ccsents.txt"
        with open(in_fp, "r", encoding="utf-8") as f:
            f.write("\n".join(l_ccsentL))

        self.predict(in_fp)

        if delete_ccsents_file:
            os.remove(in_fp)

        # Does same thing as Openie6's run.get_labels()
        if self.params.d["write_extags_file"]:
            self.write_extags_file_from_preds(l_osentL,
                                              l_ccsentL,
                                              pred_out_fp)

    def write_extags_file_from_preds(
            self,
            l_osentL,  # orig_sentences
            l_ccsentL,  # sentences
            pred_out_fp):
        """
        similar to Openie6.run.get_labels()
        ILABEL_TO_EXTAG={0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
                 4: 'ARG2', 5: 'NONE'}

        called by `splitpredict_for_ex()`


        Parameters
        ----------
        l_osentL: list[str]
        l_ccsentL: list[str]
        pred_out_fp: str


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

        # lll_cc_spanned_loc is similar to
        # sentence_indices_list, model.all_sentence_indices_conj
        lll_cc_spanned_loc = \
            self.model.lll_cc_spanned_loc
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

    def splitpredict_for_rescore(self):
        """
        reads re_allen_in_fp
        writes re_allen_out_fp

        Parameters
        ----------
        re_allen_in_fp: str
        re_allen_out_fp: str

        Returns
        -------
        None

        """
        # print()
        # print("*******Starting re-scoring")
        # print()
        #
        # # iline = line number
        # osent_iline_set = set()
        # prev_iline = 0
        # iline_to_exless_sents = {}
        # cur_iline = 0
        # for sample_str in self.model.l_ex_pred_str:
        #     sample_str = sample_str.strip('\n')
        #     num_ex = len(sample_str) - 1
        #     if num_ex == 0:
        #         if cur_iline not in iline_to_exless_sents:
        #             iline_to_exless_sents[cur_iline] = []
        #         iline_to_exless_sents[cur_iline].append(sample_str)
        #     else:
        #         cur_iline = prev_iline + num_ex
        #         # add() is like append, but for a set
        #         osent_iline_set.add(cur_iline)
        #         prev_iline = cur_iline
        #
        # # testing rescoring
        # rescored_allen_file = rescore(
        #     self.re_allen_in_fp,  # f'{self.hparams.out}.allennlp'
        #     model_dir=PRED_DIR,  #
        #     batch_size=256)
        #
        # l_rs_sent = []
        # sent_str = ""
        # for iline, line in enumerate(rescored_allen_file):
        #     fields = line.split('\t')
        #     osent = fields[0]
        #     confi = float(fields[2])
        #
        #     if iline == 0:
        #         sent_str = f'{osent}\n'
        #         l_ex = []
        #     if iline in osent_iline_set:
        #         l_ex = sorted(l_ex, reverse=True,
        #                       key=lambda x: float(x.split()[0][:-1]))
        #         l_ex = l_ex[:EX_NUM_DEPTHS]
        #         l_rs_sent.append(sent_str + ''.join(l_ex))
        #         sent_str = f'{osent}\n'
        #         l_ex = []
        #     if iline in iline_to_exless_sents:
        #         for sent in iline_to_exless_sents[iline]:
        #             l_rs_sent.append(f'{sent}\n')
        #
        #     arg1 = re.findall("<arg1>.*</arg1>", fields[1])[0].strip(
        #         '<arg1>').strip('</arg1>').strip()
        #     rel = re.findall("<rel>.*</rel>", fields[1])[0].strip(
        #         '<rel>').strip('</rel>').strip()
        #     arg2 = re.findall("<arg2>.*</arg2>", fields[1])[0].strip(
        #         '<arg2>').strip('</arg2>').strip()
        #     ex = SaxExtraction(osent,
        #                        arg1,
        #                        rel,
        #                        arg2,
        #                        confi)
        #     l_ex.append(ex.get_simple_sent() + "\n")
        #
        # l_ex = sorted(l_ex, reverse=True,
        #               key=lambda x: float(x.split()[0][:-1]))
        # l_ex = l_ex[:EX_NUM_DEPTHS]
        # l_rs_sent.append(sent_str + ''.join(l_ex))
        #
        # if iline in iline_to_exless_sents:
        #     for sent in iline_to_exless_sents[iline]:
        #         l_rs_sent.append(f'{sent}\n')
        #
        # print('Predictions written to ' + self.re_allen_out_fp)
        # with open(self.re_allen_out_fp, "w") as f:
        #     f.write('\n'.join(l_rs_sent) + '\n')

    def splitpredict(self, pred_in_fp):
        """
        similar to Openie6.run.splitpredict()

        calls self.predict() and self.rescore()

        Parameters
        ----------
        pred_in_fp: str

        Returns
        -------
        None

        """

        # def splitpredict(params_d, checkpoint_callback,
        #                  train_dataloader, val_dataloader, test_dataloader,
        #                  all_sentences):

        self.model.cc_sent_to_words = {}
        self.model.ex_sent_to_sent = {}
        self.params.d["write_allen_file"] = True

        self.params.d["task"] = self.params.task = "cc"
        l_osentL, l_ccsentL = self.splitpredict_for_cc(pred_in_fp)

        self.params.d["task"] = self.params.task = "ex"
        pred_out_fp = pred_in_fp.strip(".txt") + "_extags_out.txt"
        self.splitpredict_for_ex(l_osentL,
                                 l_ccsentL,
                                 pred_out_fp)

        if self.params.d["do_rescoring"]:
            self.splitpredict_for_rescore()
        else:
            print("not doing rescoring")

    def run(self, pred_in_fp=None):
        """

        Parameters
        ----------
        pred_in_fp: str

        Returns
        -------
        None

        """
        for process in self.params.mode.split('_'):
            if process in ["predict", "splitpredict"]:
                assert pred_in_fp
                getattr(self, process)(pred_in_fp)
            else:
                getattr(self, process)()


if __name__ == "__main__":
    """
        pid, task, mode   
        0. "", ""
        1. ex, train_test
        2. ex, test  (appears twice)
        3. ex, predict (appears twice)
        4. ex, resume
        5. cc, train_test
        6. ex, splitpredict (appears twice)

    """


    def main(pid):
        params = Params(pid)
        params.d["gpus"] = 0
        conductor = MConductor(params)
        conductor.run()


    # main(1)
    main(5)
