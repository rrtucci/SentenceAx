from SaxExtraction import *
from ExMetric import *
from CCMetric import *
from CCTree import *
from MOutput import *
from PaddedMInput import *
from SaxDataset import *
from PickleList import *
from utils_l_sample_str import write_l_sample_str

import os
from copy import copy, deepcopy
from collections import OrderedDict
import logging
import regex as re
from pprint import pprint

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
import lightning as L
from transformers import AdamW, AutoModel

# prevents printing of model weights, etc
logging.getLogger(
    'transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger(
    'transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


class Model(L.LightningModule):
    """
    
    This class inherits from L.LightningModule some powerful methods that
    loop through the batches of an epoch. It can either:

    1. calculate the loss when training (here the weights are changing)

    2. calculate the accuracy when tuning or testing or
    extracting (here the weights are fixed).

    The class loops over batches of an epoch for the 3 actions ttt=train,
    tune (a.k.a. validation) and test.

    This class has an abstract class as its parent. To distinguish between 
    inherited and uninherited methods, we add a prefix "sax_" to the name of 
    all uninherited methods.

    Note that `batch` has a different meaning in Openie6 and SentenceAx. In 
    Openie6, it's a dictionary with the fallowing keys:
    
    batch={
        "lll_label":
        "meta_data":
        "pos_locs":
        "text":
        "verb_bools":
        "verb_locs":
        "word_starts":
    }

    In SentenceAx, (see sax_get_batch_in_dicts()), we have instead:

    x, y, l_orig_sent, xname_to_l_dim1 = batch
    
    SentenceAX is a fine-tuning of bert-base-cased and bert-large-cased.
    Both of these weights/models are cased (meaning they both distinguish
    between upper and lower cases), but bert-large-cased > bert-base-cased.

    embedding = nn.Embedding(num_embeddings=L, embedding_dim=d)
    d= hidden_size = 768 for BERT base
    L = 100

    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    con_to_weight: dict[str, float]
    dropout_fun: Dropout
    embedding: Embedding
    hidden_size: int
    ilabelling_layer: Linear
    iterative_transformer: ModuleList
    l_batch_m_out: list[MOutput]
    l_cc_epoch_sample_str: list[str]
    l_cc_epoch_spanned_word: list[list[str]]
    lll_cc_epoch_spanned_loc: list[list[list[int]]]
    loss_fun: CrossEntropyLoss
    merge_layer: Linear
    metric: CCMetric | ExMetric
    name: str
    name_to_param0: dict[str, Any]
    osent_to_words: dict[str, list[str]]
    params: Params
    scores_epoch_end_d: dict[str, Any]
    base_model: BertModel
    sub_osent_to_osent: dict[str, str]
        dictionary that maps sentences to sentences.
        Both Model and ExMetric possess a pointer to this dictionary.
    verbose: bool
    # some inherited attributes that won't be used
    # hparams (dictionary, Used by Openie6, but not by us.
    #    We use the class Params instead.)
    # logger
    # trainer
    # on_gpu

    """

    def __init__(self,
                 params,
                 auto_tokenizer,
                 verbose=False,
                 name=""):
        """
        Constructor

        Parameters
        ----------
        params: Params
        auto_tokenizer: AutoTokenizer
        verbose: bool
        name: str
            name of Model instance if more than one is being used at the
            same time. ActionConductor declares 4 Model instances which it
            calls "train", "resume", "test", "pred"
        """
        super().__init__()

        # This stores `pi_test=3.14` in logs/ex/test/hparams.yaml. This is
        # here for illustrative purposes only. In SentenceAx, instead of
        # hparams, we use the Params class and sax_globals.py
        if verbose:
            self.hparams["pi_test"] = 3.14
            self.save_hyperparameters(self.hparams)
            print("Saving self.hparams= ", self.hparams)

        self.params = params
        self.auto_tokenizer = auto_tokenizer
        self.verbose = verbose
        self.name = name

        # return_dict=False avoids error message from Dropout
        self.base_model = AutoModel.from_pretrained(
            self.params.d["model_str"],
            cache_dir=CACHE_DIR,
            return_dict=False)
        self.hidden_size = self.base_model.config.hidden_size
        if self.verbose:
            print("Model init")
            print(f"\tname={self.name}, hidden_size={self.hidden_size}")

        # Actually, self.params.d["num_iterative_layers"]=2 for all Params.pid
        if self.params.d["num_iterative_layers"] > 0:
            num_layers = len(self.base_model.encoder.layer)
            num_encoder_layers = \
                num_layers - self.params.d["num_iterative_layers"]
            self.iterative_transformer = \
                self.base_model.encoder.layer[
                num_encoder_layers:num_layers]
            # this truncation of self.base_model.encoder.layer must
            # be done after, not before defining self.iterative_transformer
            self.base_model.encoder.layer = \
                self.base_model.encoder.layer[0:num_encoder_layers]
            if verbose:
                print("num_iterative_layers= ", num_layers -
                      num_encoder_layers)
                print("num_encoder_layers= ", num_encoder_layers)
                print("total num layers= ", num_layers)
        else:
            self.iterative_transformer = []

        self.dropout_fun = nn.Dropout(p=PROB_DROPOUT)  # 0.0

        self.embedding = nn.Embedding(
            MAX_NUM_OSENTL_WORDS,  # maximum number of words analyzed, 100
            self.hidden_size)  # dim of embedding space, 768
        self.merge_layer = nn.Linear(self.hidden_size,  # 768
                                     MERGE_DIM)  # 300
        self.ilabelling_layer = nn.Linear(MERGE_DIM,  # 300
                                          NUM_ILABELS)  # 6

        # ignore_index=-100 is the default, but including it
        # explicitly for clarity
        # see file misc/CrossEntropyLoss-examples.py for examples of usage
        self.loss_fun = nn.CrossEntropyLoss(ignore_index=-100)

        self.sub_osent_to_osent = {}
        # self.osent_to_words is similar to Openie6 conj_word_mapping
        # Note that self.osent_to_words is never used;
        # It is filled in ActionConductor but never used.
        # We include it in SentenceAx to follow Openie6.
        self.osent_to_words = {}

        if self.params.task == "ex":
            # ExMetric gets a pointer (address) to the sub_osent_to_osent
            # dict. This dictionary is initially empty, but if we add
            # elements to it later on, both Model and ExMetric will know
            # about it because the dictionary pointer will not have changed,
            # only its contents.
            self.metric = ExMetric(
                sub_osent_to_osent=self.sub_osent_to_osent,
                verbose=self.verbose)
        elif self.params.task == "cc":
            self.metric = CCMetric(verbose=self.verbose)

        self.scores_epoch_end_d = {}  # filled in test_epoch_end()

        if "multi_opt" not in self.params.d \
                or not self.params.d["multi_opt"]:
            constraint_str = ""
            con_weight_str = ""
        else:
            if "constraint_str" not in self.params.d or \
                    "con_weight_str" not in self.params.d:
                constraint_str = 'posm_hvc_hvr_hve'
                con_weight_str = '1_1_1_1'
            else:
                constraint_str = self.params.d["constraint_str"]
                con_weight_str = self.params.d["con_weight_str"]
        l_constraint = constraint_str.split('_')
        l_con_weight = con_weight_str.split('_')
        assert len(l_constraint) == len(l_con_weight)
        self.con_to_weight = {l_constraint[k]: float(l_con_weight[k])
                              for k in range(len(l_constraint)) if
                              l_constraint[k]}

        # no longer used
        # Openie6.all_predictions_conj
        # self.l_cc_epoch_sample_str = []

        # Openie6.all_conjunct_words_conj
        # self.l_cc_epoch_spanned_word = []

        # Openie6.all_sentence_indices_conj
        # self.lll_cc_epoch_spanned_loc = []

        # not used
        # self.l_ex_pred_sample_str = []  # Openie6.all_predictions_oie

        self.l_batch_m_out = \
            PickleList(f"action_{self.name}_l_batch_m_out_dir")

        self.name_to_param0 = None

    def configure_optimizers(self):
        """
        similar to Openie6.model.configure_optimizers()

        This method returns a list of optimizers, one for each constraint in
        self.con_to_weight. Optimizers can be either all Adam or all AdamW.

        This is how ChatGPT explains the Openie6.model.configure_optimizers(
        ) method:

        This PyTorch code is a method called `configure_optimizers` inside a
        PyTorch Lightning module or a subclass of it. This method is
        responsible for configuring the optimizers used during the training
        process. Let's break down the code:

        1. `all_params = list(self.named_parameters())`: This line retrieves
        all the parameters of the model along with their names.

        2. `bert_params = []` and `other_params = []`: These lists are used
        to separate parameters that belong to the BERT model (presumably a
        pre-trained language model) and other parameters (possibly
        task-specific layers or embeddings).

        3. `no_decay = ["bias", "gamma", "beta"]`: This list contains names
        of parameters for which weight decay is not applied. These are
        typically bias terms or normalization parameters like those in
        BatchNorm layers.

        4. `opt_params`: This list contains dictionaries, each specifying
        the parameters for a particular optimizer. The parameters are
        separated based on whether they should undergo weight decay or not.

        - For parameters that do not contain any strings in the `no_decay`
        list and contain the string `'base_model'` in their name, a weight
        decay rate of 0.01 is applied.

        - For parameters that contain strings in the `no_decay` list and
        contain the string `'base_model'` in their name, no weight decay is
        applied.

        - For parameters that do not contain the string `'base_model'` in
        their name, no weight decay is applied.

        5. `if self.hparams.optimizer == 'adamW':` and `elif
        self.hparams.optimizer == 'adam':`: These conditions select between
        the AdamW optimizer and the Adam optimizer based on the value of
        `self.hparams.optimizer`.

        6. `if self.hparams.multi_opt and self.hparams.constraints != None:`:
        This condition checks if multiple optimizers are to be used and if
        constraints are provided.

        - If both conditions are true, the number of optimizers is
        determined by the number of constraints provided, and a list of
        optimizers is returned with each optimizer having the same
        configuration.

        - If the condition is not met, a single optimizer is returned.

        Finally, the method returns a list containing the selected
        optimizer(s) for training.

        Returns
        -------
        list[Adam|AdamW]

        """
        # self.named_parameters() is a method inherited from parent class
        # Its type is Iterator[Tuple[str, Parameter]]. Apply dict() to
        # to turn it into dict[str, Parameter] or list() to turn into
        # list[tuple(str, Parameter)].

        # self.named_parameters() contains all (name, value) pairs of 
        # weights to be optimized
        all_pairs = list(self.named_parameters())

        # opt = optimizer
        # apple = parameter
        # pair = ("apple", apple)

        def base_model_pairs():
            return [pair for pair in all_pairs if
                    "base_model" in pair[0]]

        def non_base_model_pairs():
            return [pair for pair in all_pairs if
                    "base_model" not in pair[0]]

        # parameters that do not decay, fixed during optimization
        xnames = ["bias", "gamma", "beta"]

        def pair_in_xnames(pair):
            return any((pair[0] in xname) for xname in xnames)

        opt_param_d = [
            {"params": [pair[1] for pair in base_model_pairs() if
                        not pair_in_xnames(pair)],
             "weight_decay_rate": 0.01,
             'lr': self.params.d["lr"]},
            {"params": [pair[1] for pair in base_model_pairs() if
                        pair_in_xnames(pair)],
             "weight_decay_rate": 0.0,
             'lr': self.params.d["lr"]},
            {"params": [pair[1] for pair in non_base_model_pairs()],
             'lr': self.params.d["lr"]}
        ]

        if self.params.d["optimizer"] == 'adamW':
            optimizer = AdamW(opt_param_d, lr=1e-3)
        elif self.params.d["optimizer"] == 'adam':
            optimizer = Adam(opt_param_d, lr=1e-3)
        else:
            assert False

        if "multi_opt" in self.params.d:
            num_optimizers = len(self.con_to_weight)
            return [optimizer] * num_optimizers
        else:
            return [optimizer]

    def get_progress_bar_dict(self):
        """
        similar to Openie6.get_progress_bar_dict()

        Use this inherited method to add to super( ).get_progress_bar_dict()
        additional items to be displayed in the progress bar. We will not
        add any. The modified dictionary is returned by the method.

        Openie6 uses tqdm for all progress bars, including this one. We do
        too, except for this one. For this one, we use the one built into
        lightning.

        tqdm derives from the Arabic word taqaddum which can mean "progress"
        and is also an abbreviation for "I love you so much" in Spanish (te
        quiero demasiado).

        Returns
        -------
        Dict[str, int | str]
            Dictionary with the items to be displayed in the progress bar.

        """
        # take a look at what Openie6 does
        # ----------------------------------
        # # get avg_training_loss
        # running_train_loss = self.trainer.running_loss.mean()
        # avg_training_loss = running_train_loss.cpu().item() if \
        #     running_train_loss else float('NaN')
        # # get `best` as float
        # if type(self.trainer.checkpoint_callback.kth_value) \
        #         not in [int, float]:
        #     best = self.trainer.checkpoint_callback.kth_value.item()
        # else:
        #     best = self.trainer.checkpoint_callback.kth_value
        #
        # tqdm_d = OrderedDict()
        # tqdm_d['loss_fun'] = '{:.3f}'.format(avg_training_loss)
        # tqdm_d['best'] = best
        # return tqdm_d
        # ----------------------------------

        # # Get the losses
        # losses = self.log_dict.pop('val_loss', None)
        # val_losses = losses if losses is not None else self.log_dict.pop(
        #     'val_main_loss', None)

        # Get the progress bar
        progress_bar_d = super().get_progress_bar_dict()

        # # Add parameters to the progress bar
        # progress_bar_d['loss'] = self.log_dict['loss']
        # progress_bar_d['epoch_acc'] = self.log_dict['epoch_acc']
        return progress_bar_d

    @staticmethod
    def sax_get_batch_in_dicts(batch):
        """
        This method takes as input `batch`:

        x, y, l_orig_sent, xname_to_l_dim1 = batch

        and returns as output 3 dictionaries:

            x_d, y_d, meta_d

        Parameters
        ----------
        batch: tuple[torch.Tensor,
            torch.Tensor, list[str], dict[str, list[int]]

        Returns
        -------
        OrderedDict, dict[str, torch.Tensor], dict[str, list[str]]
            x_d, y_d, meta_d

        """
        x, y, l_orig_sent, xname_to_l_dim1 = batch
        y_d = {"lll_ilabel": y}
        meta_d = {"l_orig_sent": l_orig_sent}
        xname_to_dim1 = OrderedDict(
            {xname: int(l_dim1[0]) for xname, l_dim1 in
             xname_to_l_dim1.items()})
        x_d = SaxDataset.invert_cat(x, xname_to_dim1)
        return x_d, y_d, meta_d

    def sax_get_llll_word_score(self, x_d, ttt, verbose=False):
        """

        This method is used inside self.forward() and is the heart of that
        method. It contains a while loop over depths that drives a batch
        through the layers of the model and returns `llll_word_score`.
        Setting `verbose` to True prints out a detailed trail of what occurs
        in this method. The following example was obtained from such a
        verbose trail.

        Assume:
        batch_size= 24,
        hidden_size= 768,
        NUM_ILABELS= 6,
        MERGE_DIM= 300
        2 iterative layers and 5 depths.

        lll_word_score is the output of the last ilabelling_layer for each
        depth

        llll_word_score is a list of lll_word_score

        len(llll_word_score)= 5 = num_depths

        Note that llll_word_scoreT = Ten(llll_word_score)

        Parameters
        ----------
        x_d: OrderedDict
        ttt: str
        verbose: bool

        Returns
        -------
        list[torch.Tensor]
            llll_word_score

        """
        # lll_label is similar to Openie6.labels
        # first (outer) list over batch/sample of events
        # second list over extractions
        # third (inner) list over number of labels in a line
        # after padding and adding the 3 unused tokens

        # batch_size, num_depths, num_words = y_d["lll_ilabel"].shape
        # sometimes num_depths will exceed max.
        # This doesn't happen when training, because
        # num_depths is specified when training.
        num_depths = get_num_depths(self.params.task)

        # `loss_fun` is not used in this function anymore
        # loss_fun, lstm_loss = 0, 0

        # batch_text = " ".join(redoL(meta_d["l_orig_sent"]))
        # base_model_input = \
        #     torch.Tensor(self.auto_tokenizer.encode(batch_text))
        if verbose:
            print("Entering model.get_llll_word_score()")
        hstate_count = Counter(verbose, "lll_hidstate")
        word_hstate_count = Counter(verbose, "lll_word_hidstate")
        lll_hidstate, _ = self.base_model(x_d["ll_osent_icode"])
        hstate_count.new_one(reset=True)
        comment(
            verbose,
            prefix="after base_model",
            params_d={
                "ll_osent_icode.shape": x_d["ll_osent_icode"].shape,
                "lll_hidstate.shape": lll_hidstate.shape})
        lll_word_score = Ten([0])  # this statement is unnecessary
        llll_word_score = []  # ~ Openie6.all_depth_scores
        depth = 0
        # loop over depths
        while True:
            for ilay, layer in enumerate(self.iterative_transformer):
                comment(verbose,
                        prefix="*********** Starting iterative layer",
                        params_d={"ilay": ilay})
                # layer(lll_hidstate)[0] returns a copy
                # of the tensor lll_hidstate after transforming it
                # in some way. [0] chooses first component
                comment(
                    verbose,
                    prefix="Before iterative layer",
                    params_d={
                        "ilay": ilay,
                        "depth": depth,
                        "lll_hidstate.shape": lll_hidstate.shape})
                lll_hidstate = layer(lll_hidstate)[0]
                hstate_count.new_one()
                comment(
                    verbose,
                    prefix="After iterative layer",
                    params_d={
                        "ilay": ilay,
                        "depth": depth,
                        "lll_hidstate.shape": lll_hidstate.shape})
            comment(verbose,
                    prefix="Before dropout",
                    params_d={
                        "depth": depth,
                        "lll_hidstate.shape": lll_hidstate.shape})
            lll_hidstate = self.dropout_fun(lll_hidstate)
            hstate_count.new_one()
            comment(verbose,
                    prefix="After dropout",
                    params_d={
                        "depth": depth,
                        "lll_hidstate.shape": lll_hidstate.shape})
            lll_loc = x_d["ll_osent_wstart_loc"].unsqueeze(2). \
                repeat(1, 1, lll_hidstate.shape[2])
            lll_word_hidstate = torch.gather(
                input=lll_hidstate,
                dim=1,
                index=lll_loc)
            comment(
                verbose,
                prefix="Gather's 2 inputs, then output",
                params_d={
                    "lll_hidstate.shape": lll_hidstate.shape,
                    "lll_loc.shape": lll_loc.shape,
                    "lll_word_hidstate.shape": lll_word_hidstate.shape})
            word_hstate_count.new_one(reset=True)
            if depth != 0:
                comment(
                    verbose,
                    prefix="before argmax",
                    params_d={"lll_word_score.shape": lll_word_score.shape})
                ll_greedy_ilabel = torch.argmax(lll_word_score, dim=-1)
                comment(
                    verbose,
                    prefix="after argmax",
                    params_d={"ll_greedy_ilabel.shape":
                                  ll_greedy_ilabel.shape})
                # not an integer code/embedding
                comment(
                    verbose,
                    prefix="before embedding",
                    params_d={"ll_greedy_ilabel.shape":
                                  ll_greedy_ilabel.shape})
                lll_pred_code = self.embedding(ll_greedy_ilabel)
                comment(
                    verbose,
                    prefix="after embedding",
                    params_d={"lll_word_hidstate.state":
                                  lll_word_hidstate.shape})
                lll_word_hidstate += lll_pred_code
                word_hstate_count.new_one()
                comment(
                    verbose,
                    prefix="just summed two signals with this shape",
                    params_d={
                        "depth": depth,
                        "lll_word_hidstate.shape": lll_word_hidstate.shape})
            comment(verbose,
                    prefix="Before merge layer",
                    params_d={
                        "depth": depth,
                        "lll_word_hidstate.shape": lll_word_hidstate.shape})
            lll_word_hidstate = self.merge_layer(lll_word_hidstate)
            comment(
                verbose,
                prefix="After merge layer",
                params_d={
                    "depth": depth,
                    "lll_word_hidstate.shape": lll_word_hidstate.shape})
            comment(
                verbose,
                prefix="Before ilabelling",
                params_d={
                    "depth": depth,
                    "lll_word_hidstate.shape": lll_word_hidstate.shape})
            lll_word_score = self.ilabelling_layer(lll_word_hidstate)
            comment(
                verbose,
                prefix="After ilabelling",
                params_d={
                    "depth": depth,
                    "lll_word_score.shape": lll_word_score.shape})
            llll_word_score.append(lll_word_score)

            depth += 1
            if depth >= num_depths:
                break
            if ttt != 'train':
                # torch.max() returns a tuple (max, argmax)
                ll_pred_ilabel = torch.max(lll_word_score, dim=2)[1]
                valid_extraction = False
                # if not training, leave while loop if
                # ll_pred_ilabel has no valid extractions in it
                for l_pred_ilabel in ll_pred_ilabel:
                    if is_valid_label_list(
                            l_pred_ilabel, self.params.task, "ilabels"):
                        valid_extraction = True
                        break
                if not valid_extraction:
                    break
        comment(verbose,
                prefix="Leaving Model.sax_get_llll_word_score()",
                params_d={
                    "len(llll_word_score)": len(llll_word_score),
                    "llll_word_score[0].shape": llll_word_score[0].shape})
        return llll_word_score

    @staticmethod
    def sax_penalty_loss(x_d,
                         llll_word_scoreT,
                         con_to_weight):
        """
        similar to Openie6.model.constrained_loss()

        This method is called inside sax_batch_loss(). It returns the
        penalty loss.

        Parameters
        ----------
        x_d: OrderedDict
        llll_word_scoreT: torch.Tensor
        con_to_weight: dict[str, float]

        Returns
        -------
        float
            penalty_loss

        """
        batch_size, num_depths, num_words, icode_dim = \
            llll_word_scoreT.shape
        penalty_loss = 0
        llll_index = x_d["ll_osent_verb_loc"]. \
            unsqueeze(1).unsqueeze(3).repeat(1, num_depths, 1, icode_dim)
        llll_verb_trust = torch.gather(
            input=llll_word_scoreT,
            dim=2,
            index=llll_index)
        lll_verb_rel_trust = llll_verb_trust[:, :, :, 2]
        # (batch_size, depth, num_words)
        lll_bool = (x_d["ll_osent_verb_loc"] != 0).unsqueeze(1).float()

        lll_verb_rel_trust = lll_verb_rel_trust * lll_bool
        # every head-verb must be included in a relation
        if 'hvc' in con_to_weight:
            ll_column_loss = \
                torch.abs(1 - torch.sum(lll_verb_rel_trust, dim=1))
            ll_column_loss = \
                ll_column_loss[x_d["ll_osent_verb_loc"] != 0]
            penalty_loss += con_to_weight['hvc'] * ll_column_loss.sum()

        # extractions must have at least k-relations with 
        # a head verb in them
        if 'hvr' in con_to_weight:
            l_a = x_d["ll_osent_verb_bool"].sum(dim=1).float()
            l_b = torch.max(lll_verb_rel_trust, dim=2)[0].sum(dim=1)
            row_rel_loss = F.relu(l_a - l_b)
            penalty_loss += con_to_weight['hvr'] * row_rel_loss.sum()

        # one relation cannot contain more than one head verb
        if 'hve' in con_to_weight:
            ll_ex_loss = \
                F.relu(torch.sum(lll_verb_rel_trust, dim=2) - 1)
            penalty_loss += con_to_weight['hve'] * ll_ex_loss.sum()

        if 'posm' in con_to_weight:
            llll_index = \
                x_d["ll_osent_pos_loc"].unsqueeze(1).unsqueeze(3). \
                    repeat(1, num_depths, 1, icode_dim)
            llll_pred_trust = torch.gather(
                input=llll_word_scoreT,
                dim=2,
                index=llll_index)
            lll_pos_not_none_trust = \
                torch.max(llll_pred_trust[:, :, :, 1:], dim=-1)[0]
            ll_column_loss = \
                (1 - torch.max(lll_pos_not_none_trust, dim=1)[0]) * \
                (x_d["ll_osent_pos_loc"] != 0).float()
            penalty_loss += con_to_weight['posm'] * ll_column_loss.sum()

        return penalty_loss

    def sax_get_con_to_l_penalty_loss(self,
                                      x_d,
                                      llll_word_score,
                                      lll_pred_ilabel0):
        """
        This method returns a dictionary con_to_l_penalty_loss. Although
        Openie6 calculates con_to_l_penalty_loss inside self.forward(),
        it never uses it. SentenceAx doesn't either.

        con_to_l_penalty_loss similar to Openie6._constD.

        Parameters
        ----------
        x_d: OrderedDict
        llll_word_score: list[torch.Tensor]
        lll_pred_ilabel0: torch.Tensor

        Returns
        -------
        dict[str, list[float]]

        """
        con_to_l_penalty_loss = {}
        # this calculates llll_word_score
        if self.constraint_str and \
                'extract' not in self.params.action and \
                self.params.d["batch_size"] != 1:
            # reshape llll_word_score
            llll_word_scoreT = torch.cat([lll.unsqueeze(1) for
                                          lll in llll_word_score], dim=1)
            # this fills tensor with 0's
            llll_word_scoreT.fill_(0)

            # for checking test set
            # lll_ilabel = copy(lll_pred_ilabel)
            # ll_ilabel[lll_ilabel == -100] = 0
            lll_ilabel = copy(lll_pred_ilabel0)

            llll_ilabel = lll_ilabel.unsqueeze(-1)
            number_depths = llll_ilabel.shape[1]
            llll_word_scoreT = llll_word_scoreT[:, :number_depths, :, :]
            llll_word_scoreT.scatter_(
                dim=3,
                index=llll_ilabel.long(),
                src=1)

            # this uses llll_word_score that was calculated previously
            # to calculate con_to_l_penalty_loss
            for constraint, con_weight in self.con_to_weight.items():
                penalty_loss = Model.sax_penalty_loss(
                    x_d,
                    llll_word_scoreT,
                    {constraint: con_weight})
                if constraint not in con_to_l_penalty_loss:
                    con_to_l_penalty_loss[constraint] = []
                con_to_l_penalty_loss[constraint].append(penalty_loss)
        return con_to_l_penalty_loss

    def forward(self, batch, batch_idx, ttt):
        """
        This method returns an instance of MOutput named batch_m_out.
        batch_m_out is the output after a batch passes through all the
        layers of the neural net.

        signature of parent method:  def forward(self, *args, **kwargs)

        The following methods invoke forward() once:
        training_step(), validation_step(), test_step()

        lll_word_score = Openie6.word_scores
        llll_word_score = Openie6.all_depth_scores (shape=(5,..))

        lll_pred_ilabel0 = Openie6.predictions
        llll_pred_ilabel = Openie6.all_depth_predictions

        ll_pred_confidence0 = Openie6.confidences
        lll_pred_confidence = Openie6.all_depth_confidences

        the outermost l in "all_depths_*" is for depth \in range(5)

        Many of the tensor contortions in this method are done in order to
        move that depth index in llll_word_score from the outermost position
        to the dim=1, where it is located in lll_ilabel. Also, we need to
        get rid (by argmax) of the innermost index corresponding to the 6
        possible ilabels (classes).


        if A and B are of shape (3, 4):
        torch.cat([A, B], dim=0) will be of shape (6, 4)
        torch.stack([A, B], dim=0) will be of shape (2, 3, 4)


        Parameter
        ----------
        batch: tuple[torch.Tensor,
            torch.Tensor, list[str], dict[str, list[int]]
        batch_idx: int
        ttt: str

        Returns
        -------
        MOutput
            batch_m_out

        """
        x_d, y_d, meta_d = Model.sax_get_batch_in_dicts(batch)
        # print_tensor("y_d['lll_ilabel']", y_d['lll_ilabel'])
        # print_list("y_d['lll_ilabel'][0][0]", y_d['lll_ilabel'][0][0])

        use_wreg = "wreg" in self.params.d and self.params.d["wreg"] != 0
        if use_wreg:
            # wreg=weight regulator
            # name_to_param0 is self.named_parameters() when
            # forward() is first called
            if not self.name_to_param0:
                name_to_param0 = deepcopy(
                    dict(self.named_parameters()))

        # lll_label is similar to Openie6.labels
        # first (outer) list over batch/sample of events
        # second list over extractions
        # third (inner) list over number of labels in a line
        # after padding and adding the 3 unused tokens
        # batch_size, num_depths, num_words = y_d["lll_ilabel"].shape

        # `loss_fun` is not used in this function anymore
        # loss_fun, lstm_loss = 0, 0

        # batch_text = " ".join(redoL(meta_d["l_orig_sent"]))
        # base_model_input = \
        #     torch.Tensor(self.auto_tokenizer.encode(batch_text))

        llll_word_score = self.sax_get_llll_word_score(
            x_d, ttt, self.verbose)

        # print_tensor("lll_word_score", lll_word_score)
        # print("vvbg", "len(llll_word_score)", len(llll_word_score))
        # print_tensor("llll_word_score[0]", llll_word_score[0])
        loss = 0
        llll_pred_ilabel = []
        lll_pred_confidence = []
        batch_size, num_words, xxx = llll_word_score[0].shape
        # len(llll_word_score)=5, the num_depths
        # xxx = 6, the number of different ilabels (num_classes)
        # y_d["lll_ilabel"] = \
        #     y_d["lll_ilabel"].long()
        for depth, lll_word_score in enumerate(llll_word_score):
            if ttt == 'train':
                # Here -1 will be
                # num_ilabels=6=number of classes to classify.
                # In general, reshape(x, -1) means final shape = (x, y)
                # where y is whatever it takes to get original num of entries
                ll_loss_input = \
                    lll_word_score.reshape(batch_size * num_words, -1)
                # print_tensor("lll_word_score", lll_word_score)
                # print_tensor("ll_loss_input", ll_loss_input)

                # ll_loss_input.shape = (batch_size * num_words, num_classes=6)
                #                       = data
                # l_loss_target.shape = (batch_size * num_words, )
                #                       = theory
                # l_loss_target[i] \in range(6)
                # loss is scalar

                l_loss_target = \
                    y_d["lll_ilabel"][:, depth, :num_words].reshape(-1)
                loss += self.loss_fun(ll_loss_input,
                                      l_loss_target)

                # print("loss shape", loss.shape)
                # print_tensor("l_loss_target", l_loss_target)
                # print("loss", loss)
            if ttt != "train":
                lll_soft_word_score = \
                    torch.log_softmax(lll_word_score, dim=2)
                ll_max_log_prob, ll_pred_ilabel = \
                    torch.max(lll_soft_word_score, dim=2)
                # print_tensor("ll_max_log_prob", ll_max_log_prob)
                # print_tensor("ll_pred_ilabel", ll_pred_ilabel)

                # print("task=", ttt)
                # print_tensor("lll_word_score", lll_word_score)
                # print_tensor("lll_soft_word_score", lll_soft_word_score)
                # print_tensor("ll_pred_ilabel", ll_pred_ilabel)
                # print("sum(""ll_pred_ilabel=", torch.sum(ll_pred_ilabel))

                # remember: lll_ilabel was similar to Openie6.labels
                # first (outermost) list over batch events
                # second list over extractions
                # third (innermost) list over number of ilabels in a line

                # print("ttt, action", ttt, self.params.action)
                # print_tensor("lll_ilabel", y_d["lll_ilabel"])

                # For ttt!=train, y_d["lll_ilabel"] entries are all \in  [
                # 0, -100] because we store that info in Carb benchmarks.
                # That is fine because we only need y_d["lll_ilabel"] to
                # create this template
                ll_nonpad_bool = \
                    (y_d["lll_ilabel"][:, 0, :] != -100).float()
                # print("dfrt", {name: x_d[name].shape for name in x_d.keys()})
                # print_tensor("ll_nonpad_bool", ll_nonpad_bool)
                # print_tensor("(ll_pred_ilabel != 0)",
                #              (ll_pred_ilabel != 0).float())
                # * is element-wise multiplication of tensors

                ll_nonpad_bool = ll_nonpad_bool[:, :ll_pred_ilabel.shape[1]]
                ll_nonpad_bool = \
                    (ll_pred_ilabel != 0).float() * ll_nonpad_bool
                ll_norm_log_prob = \
                    (ll_max_log_prob * ll_nonpad_bool) \
                    / (1 + ll_nonpad_bool.sum(dim=0))
                l_confidence = torch.exp(
                    torch.sum(ll_norm_log_prob, dim=1))

                # this unsqueezes depth dim=1
                llll_pred_ilabel.append(ll_pred_ilabel.unsqueeze(1))
                lll_pred_confidence.append(l_confidence.unsqueeze(1))
        # } end of loop over depth, lll_word_score

        if ttt == 'train':
            if self.con_to_weight:
                # unsqueeze dim=1, then cat() along dim=1. This
                # removes the outermost l and "fattens" dim=1
                llll_word_scoreT = torch.cat(
                    [lll.unsqueeze(1) for lll in llll_word_score], dim=1)
                llll_word_scoreT = torch.softmax(llll_word_scoreT, dim=-1)

                penalty_loss = Model.sax_penalty_loss(
                    x_d,
                    llll_word_scoreT,
                    self.con_to_weight) / batch_size

                # IMPORTANT Openie6 has
                # loss = const_loss
                # instead of
                # loss += const_loss
                loss += penalty_loss

            if use_wreg:
                weight_diff = 0
                # name_to_param0 is self.named_parameters()
                # when forward() is first called
                name_to_param = dict(self.named_parameters())
                for name in name_to_param0:
                    weight_diff += torch.linalg.vector_norm(
                        name_to_param[name] - name_to_param0[name])
                loss += self.params.d["wreg"] * weight_diff

        # llll_pred_ilabel: list[tensor]
        # lll_pred_confidence: list[tensor]
        if ttt != "train":
            # unsqueeze dim=1, then cat along dim=1. This
            # removes the outermost l and "fattens" dim=1
            lll_pred_ilabel0 = torch.cat(llll_pred_ilabel, dim=1)
            ll_pred_confidence0 = torch.cat(lll_pred_confidence, dim=1)
            # never used
            # self.con_to_l_loss = self.sax_get_con_to_l_loss(
            #     x_d,
            #     llll_word_scoreT,
            #     lll_pred_ilabel0)
            assert loss == 0
        else:
            lll_pred_ilabel0 = Ten([0])
            ll_pred_confidence0 = Ten([0])

        batch_m_out = MOutput(meta_d["l_orig_sent"],
                              y_d["lll_ilabel"],
                              lll_pred_ilabel0,
                              ll_pred_confidence0,
                              loss)
        return batch_m_out

    def sax_write_if_task_ex(self, batch_idx, batch_m_out):
        """
        This method is called by `sax_write_batch_sents_out()`. it writes:

        1. an ssents (simple sentences) file at f"{M_OUT_DIR}/ex_ssents.txt"

        2. an Allen file at f"{M_OUT_DIR}/ex_allen.txt"


        Parameters
        ----------
        batch_idx: int
        batch_m_out: MOutput

        Returns
        -------
        None

        """
        lll_pred_ilabel = batch_m_out.lll_pred_ilabel
        ll_confidence = batch_m_out.ll_pred_confidence
        num_samples, num_depths, _ = lll_pred_ilabel.shape
        l_orig_sent = batch_m_out.l_orig_sent

        osent_to_l_pred_ex = {}
        for sample_id, orig_sent in enumerate(l_orig_sent):
            orig_sentL = redoL(orig_sent)
            add_key_to_this_d(key=orig_sent,
                              grow_d=self.sub_osent_to_osent,
                              this_d=osent_to_l_pred_ex)
            for depth in range(num_depths):
                num_words = len(get_words(orig_sentL))
                ex_ilabels = lll_pred_ilabel[sample_id][depth][:num_words]
                if sum(ex_ilabels) == 0:  # extractions completed
                    break
                ex = SaxExtraction.get_ex_from_ilabels(
                    ex_ilabels, orig_sentL, ll_confidence[sample_id][depth])
                if ex.arg1 and ex.rel:
                    add_key_value_pair_to_this_d(
                        key=orig_sent,
                        value=ex,
                        grow_d=self.sub_osent_to_osent,
                        this_d=osent_to_l_pred_ex)
        l_pred_sample_str = []  # ~ Openie6.all_pred
        l_pred_al_sample_str = []  # ~ Openie6.all_pred_allen_nlp
        for osent, l_pred_ex in osent_to_l_pred_ex.items():
            orig_sentL = redoL(osent)
            str0 = orig_sentL + "\n"
            for pred_ex in l_pred_ex:
                str0 += pred_ex.get_simple_sent() + '\n'
            l_pred_sample_str.append(str0.strip())
            al_sample_str = ""
            for pred_ex in l_pred_ex:
                arg1 = pred_ex.arg1
                rel = pred_ex.rel
                arg2 = pred_ex.arg2
                al_sample_str += f"{orig_sentL}\t"
                al_sample_str += f"<arg1> {arg1} </arg1>"
                al_sample_str += f"<rel> {rel} </rel>"
                al_sample_str += f"<arg2> {arg2} </arg2>\t"
                al_sample_str += f"{pred_ex.confidence}\n"
            l_pred_al_sample_str.append(al_sample_str.strip())

        out_fp = f"{M_OUT_DIR}/ex_ssents.txt"
        appended = False if batch_idx == 0 else True
        write_l_sample_str(l_pred_sample_str,
                           out_fp,
                           appended,
                           numbered=False)

        allen_out_fp = f"{M_OUT_DIR}/ex_allen.txt"
        write_l_sample_str(l_pred_al_sample_str,
                           allen_out_fp,
                           appended,
                           numbered=False)

        # self.l_ex_pred_sample_str = l_pred_sample_str

    def sax_write_if_task_cc(self, batch_idx, batch_m_out):
        """

        This method is called by `sax_write_batch_sents_out()`. It writes:

        1. an ssents (simple sentences) file at f"{M_OUT_DIR}/cc_ssents.txt"

        Parameters
        ----------
        batch_idx: int
        batch_m_out: MOutput

        Returns
        -------
        None

        """

        # correct = True
        # total_num_ccsents1 = 0
        # total_num_ccsents2 = 0
        lll_pred_ilabel = batch_m_out.lll_pred_ilabel
        num_samples, num_depths, _ = lll_pred_ilabel.shape
        # true_lll_ilabel = self.true_batch_m_out.lll_label
        l_orig_sent = batch_m_out.l_orig_sent
        l_pred_sample_str = []
        # ll_spanned_word = []
        # lll_spanned_loc = []
        for isam, orig_sent in enumerate(l_orig_sent):
            ll_ilabel = []
            for depth in range(num_depths):
                num_words = len(get_words(orig_sent))
                l_ilabel = lll_pred_ilabel[isam][depth][:num_words].tolist()
                ll_ilabel.append(l_ilabel)

            pred_sample_str = redoL(orig_sent) + '\n'

            # CCTree.set_ccsents() ~ Openie6.data.coords_to_sentences()
            # ccsents ~ Openie6.split_sentences,
            #           ~ Openie.6.word_sentences
            # l_spanned_word ~ Openie6.conj_words,
            # ll_spanned_loc ~ Openie6.sentence_indices_list

            tree = CCTree(orig_sent, ll_ilabel)
            ccsents = tree.ccsents  # ~ Openie6.split_sentences
            # print("orig_sent", orig_sent)
            # print("ll_ilabel", ll_ilabel)
            # print_list("ccsents", ccsents)
            # tree.draw_self()
            # spanned_words = \
            #     tree.l_spanned_word  # ~ Openie6.conj_words
            # ll_spanned_loc = \
            #     tree.ll_spanned_loc  # ~ Openie6.sentence_indices_list
            # ll_spanned_word.append(spanned_words)
            # lll_spanned_loc.append(ll_spanned_loc)
            # not used
            # total_num_ccsents1 += len(ccsents)
            # total_num_ccsents2 += 1 if len(ccsents) > 0 else 0
            pred_sample_str += '\n'.join(ccsents)
            l_pred_sample_str.append(pred_sample_str)
        # list1 + list2 is the same as list1.extend(list2)
        # left sides accumulate over all batches
        # no longer used
        # self.l_cc_epoch_spanned_word += ll_spanned_word
        # self.l_cc_epoch_sample_str += l_pred_sample_str
        # self.lll_cc_epoch_spanned_loc += lll_spanned_loc

        appended = False if batch_idx == 0 else True
        out_fp = f"{M_OUT_DIR}/cc_ssents.txt"
        write_l_sample_str(l_pred_sample_str,
                           out_fp,
                           appended,
                           numbered=False)

    def sax_write_batch_sents_out(self, batch_idx, batch_m_out):
        """
        similar to Openie6.model.write_to_file()

        This method is called by sax_ttt_step() when ttt="tune".

        For task="ex", it appends stuff, after each step (i.e., batch),
        to the files at f"{M_OUT_DIR}/extags.txt" and f"{
        M_OUT_DIR}/allen.txt".

        For task="cc", it appends stuff, after each step, to the file at f"{
        M_OUT_DIR}/cctags.txt"


        Parameters
        ----------
        batch_idx: int
        batch_m_out: MOutput

        Returns
        -------
        None

        """
        # batch_m_out = self.l_batch_m_out[batch_idx]
        # batch_m_out.move_to_cpu()

        if self.params.task == "ex":
            self.sax_write_if_task_ex(batch_idx, batch_m_out)
        elif self.params.task == "cc":
            self.sax_write_if_task_cc(batch_idx, batch_m_out)
        else:
            assert False

    def sax_ttt_step(self, batch, batch_idx, ttt):
        """
        This method calls forward() which returns batch_m_out and loss =
        batch_m_out.loss. The method checks that loss==0 for ttt!="train".
        Then it logs the loss (for all ttt).

        If "extract" in params.action, the method writes to a file by
        calling self.sax_write_batch_sents_out().

        If ttt != "train", the method stores a list of batch_m_out.
        l_batch_m_out is similar to Openie6.outputs.


        Parameters
        ----------
        batch: tuple[torch.Tensor,
            torch.Tensor, list[str], dict[str, list[int]]
        batch_idx: int
        ttt: str

        Returns
        -------
        None

        """
        batch_m_out = self.forward(batch, batch_idx, ttt)
        loss_d = {}
        loss_d[ttt + "_loss"] = batch_m_out.loss
        if ttt == "train":
            loss_d["loss"] = batch_m_out.loss

        if "extract" in self.params.action:
            # Openie6 only writes on validation (tune) step
            # We will write iff "extract" in action
            # because that is the only time these files are
            # read later on.
            self.sax_write_batch_sents_out(batch_idx, batch_m_out)

        if ttt != "train":
            # only remember batch_m_out if going to score it
            # at end of epoch. This only happens if ttt != "train".
            self.l_batch_m_out.append(batch_m_out)
        if self.verbose or (not self.verbose and batch_idx==0):
            ttt_to_long = {"train": "training",
                           "tune": "validation",
                           "test": "test"}
            print(f"Inside Model.{ttt_to_long[ttt]}_step method, "
                  f"batch_idx={batch_idx}",
                  round_dict_values(loss_d))

        self.log_dict(
            loss_d,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True)
        return loss_d

    def training_step(self, batch, batch_idx):
        """
        This method returns self.sax_ttt_step() so go there for an explanation.


        Parameters
        ----------
        batch: tuple[torch.Tensor,
            torch.Tensor, list[str], dict[str, list[int]]
        batch_idx: int

        Returns
        -------
        None

        """
        return self.sax_ttt_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        This method returns self.sax_ttt_step() so go there for an explanation.


        Parameters
        ----------
        batch: tuple[torch.Tensor,
            torch.Tensor, list[str], dict[str, list[int]]
        batch_idx: int

        Returns
        -------
        None

        """
        return self.sax_ttt_step(batch, batch_idx, "tune")

    def test_step(self, batch, batch_idx):
        """
        This method returns self.sax_ttt_step() so go there for an explanation.


        Parameters
        ----------
        batch: tuple[torch.Tensor,
            torch.Tensor, list[str], dict[str, list[int]]
        batch_idx: int

        Returns
        -------
        None

        """
        return self.sax_ttt_step(batch, batch_idx, "test")

    def sax_get_scores_on_ttt_epoch_end(self, ttt):
        """
        similar to Openie6.model.evaluation_end()

        This method prints and returns scores_epoch_end_d. That dictionary
        includes `epoch_acc` among its keys. The method is called inside
        self.sax_on_ttt_epoch_end()


        Parameters
        ----------
        ttt: str
            either "train", "tune", "test"

        Returns
        -------
        dict[str, Any]
            scores_epoch_end_d

        """
        # if self.params.action == 'test':
        #     for batch_m_out in self.l_batch_m_out:
        #         batch_m_out.move_to_cpu()

        if 'extract' in self.params.action:
            score_d = self.metric.get_zero_score_d()
        else:
            for k, batch_m_out in enumerate(self.l_batch_m_out):
                if self.params.task == "cc":
                    self.metric(
                        batch_m_out.l_orig_sent,  # meta data
                        batch_m_out.lll_pred_ilabel,  # predictions
                        batch_m_out.lll_ilabel)  # ground truth
                elif self.params.task == "ex":
                    # for task="ex", ground truth in Carb benchmarks
                    self.metric(
                        batch_m_out.l_orig_sent,  # meta data
                        batch_m_out.lll_pred_ilabel,  # predictions
                        batch_m_out.ll_pred_confidence)  # scores
                    # print("qswed", batch_m_out.ll_pred_confidence)
            score_d = self.metric.get_score_d(ttt)


        if self.params.task == "cc":
            epoch_acc = score_d["acc_nsam_exact"][0]
        elif self.params.task == "ex":
            epoch_acc = score_d["F1"]
        else:
            assert False
        scores_epoch_end_d = dict(score_d)
        scores_epoch_end_d["epoch_acc"] = epoch_acc

        if self.params.task == "ex":
            scores_epoch_end_d = round_dict_values(scores_epoch_end_d)

        print('\nScores at end of epoch ' +
              str(self.trainer.current_epoch) + ":")
        pprint(scores_epoch_end_d)
        # For computing the constraint violations
        # if hasattr(self, 'con_to_l_penalty_loss') and \
        # self.params.d["constraint_str"] != '':
        #     for key in self.con_to_l_penalty_loss:
        #         self.con_to_l_penalty_loss[key] =
        #         sum(self.con_to_l_penalty_loss[key]).item()
        #     print('\nViolations: ', self.con_to_l_penalty_loss)
        #     self.con_to_l_penalty_loss = dict()
        return scores_epoch_end_d

    def sax_on_ttt_epoch_end(self, ttt):
        """
        This method calculates the score called `epoch_acc` (i.e.,
        the accuracy at epoch's end), and logs it. It also resets
        l_batch_m_out to []. Scores are only relevant for ttt!="train",
        so this method is called by on_validation_epoch_end() and
        on_test_epoch_end() but not by on_train_epoch_end().

        Parameters
        ----------
        ttt: str

        Returns
        -------
        None

        """
        if self.verbose:
            if ttt == "train":
                assert False
            elif ttt == "tune":
                str0 = "on_validation_epoch_end"
            elif ttt == "test":
                str0 = "on_test_epoch_end"
            else:
                assert False
            if self.verbose:
                print(f"Entering Model.{str0} method")

        self.scores_epoch_end_d = \
            self.sax_get_scores_on_ttt_epoch_end(ttt)
        # epoch_end_d = {"log": scores_epoch_end_d,
        #                "epoch_acc": scores_epoch_end_d["epoch_acc"]}
        # if ttt == "test":
        #     epoch_end_d["progress_bar"] = self.scores_epoch_end_d

        epoch_acc = self.scores_epoch_end_d["epoch_acc"]
        acc_d = {}
        acc_d[ttt + "_epoch_acc"] = float(epoch_acc)
        self.log_dict(acc_d,
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True)

        self.l_batch_m_out.restart()
        # self.l_batch_m_out.clear()  # free memory
        return acc_d

    def on_validation_epoch_end(self):
        """
        This method returns self.sax_on_ttt_epoch_end() so go there for an
        explanation.

        Returns
        -------
        None

        """
        return self.sax_on_ttt_epoch_end("tune")

    def on_test_epoch_end(self):
        """
        This method returns self.sax_on_ttt_epoch_end() so go there for an
        explanation.

        Returns
        -------
        None

        """
        return self.sax_on_ttt_epoch_end("test")

    def train_dataloader(self):
        """
        This method does nothing. It's here to override the parent method.

        Returns
        -------
        None

        """
        return

    def val_dataloader(self):
        """
        This method does nothing. It's here to override the parent method.

        Returns
        -------
        None

        """
        return
