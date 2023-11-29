from sax_globals import *
from pprint import pprint


class Params:
    """

    The purpose of this class is to carry a dictionary self.d of parameters
    for SentenceAx. In SentenceAx, globals that are expected to change only
    once in a blue mooon, are capitalized and declared in the file
    sax_globals.py. Although self.d carries some globals, it also carries
    some params that might be changed in the middle of a run. Do not define
    a capitalized global and a self.d key for the same parameter. Define one
    or the other but not both.

    Instead of this Params class, Openie6 stores all its parameters in a
    dictionary called `hparams` (hyperparameters) that is an attribute of
    the Model class.


    Possible choices for self.d:
    pid, task, action
    0. "", ""
    1. ex, train_test
       Conductor.train() followed by Conductor.test()
    2. ex, test  (appears twice in Openie6 readme)
        Conductor.test()
    3. ex, predict (appears twice in Openie6 readme)
        Conductor.predict()
    4. ex, resume
        Conductor.resume()
    5. cc, train_test
       Conductor.train() followed by Conductor.test()
    6. ex, splitpredict (appears twice in Openie6 readme)
       Conductor.splitpredict()

    self.task in ("ex", "cc"). SentenceAx uses ("ex", "cc") whereas Openie6
    uses ("oie", "conj") for self.task

    `action` is similar to Openie6.mode. self.action in ("train_test",
    "resume", "test", "predict", "splitpredict")


    Attributes
    ----------
    d: dict[str, Any]
    pid: int
        process ID, in {0, 1, ..., 6}
    action: str
    task: str

    """

    def __init__(self, pid):
        """

        Parameters
        ----------
        pid: int
        """
        self.pid = pid
        pid_to_pair = {
            0: ("", ""),
            1: ("ex", "train_test"),
            2: ("ex", "test"),
            3: ("ex", "predict"),
            4: ("ex", "resume"),
            5: ("cc", "train_test"),
            6: ("ex", "splitpredict")
        }

        self.task, self.action = pid_to_pair[pid]
        self.d = self.get_d()

    def describe_self(self):
        """
        This method pprints the dictionary self.d.

        Returns
        -------
        None

        """
        print("***************** new params")
        print(f"new params: "
            f"pid={str(self.pid)}, task='{self.task}', action='{self.action}'")
        print("params=")
        pprint(self.d)

    def get_d(self):
        """
        This is an internal method to be called only by __init__. The user
        should access the dictionary d as params.d, where params is the name
        of the Params instance.

        Note that at the end of this method, we define a default d
        dictionary, and then we merge the dominant d dictionary with the
        default one.

        The parameter values in this method come directly from the Openie6
        readme page. Note that that readme page repeats some (task, action)
        cases. In the case of repeats, we comment out all repeats except one.

        Returns
        -------
        dict[str, Any]

        """
        if not self.task:
            print("****self.task is empty")
        else:
            assert self.task in ["ex", "cc"]

        if not self.action:
            print("****self.action is empty")
        else:
            assert self.action in ["predict", "train_test", "splitpredict",
                                   "resume", "test"]

        ## Running self.model
        # this is repeated at begin and end of Openie6.readme.
        # elif self.task == "ex" and self.action == "splitpredict":
        #     self.d = {
        #         "cc_weights_fp": WEIGHTS_DIR +
        #                     "/cc_epoch=28_eval_acc=0.854.ckpt",
        #         "gpus": 1,
        #         # "inp": "sentences.txt",
        #         "action": "splitpredict",
        #         "num_extractions": 5,
        #         "ex_weights_fp": WEIGHTS_DIR +
        #                     "/ex_epoch=14_eval_acc=0.551_v0.ckpt",
        #         #"out": "predictions.txt",
        #         # "rescore_model":,
        #         # "rescoring":,
        #         "task": "ex"
        #     }

        ## Training self.model

        ### Warmup self.model
        # Training:
        if self.task == "ex" and self.action == "train_test":
            self.d = {
                "batch_size": 24,
                "num_epochs": 30,
                "gpus": 1,
                "num_iterative_layers": 2,
                "lr": 2E-5,
                "action": "train_test",
                "optimizer": "adamW",
                # "save": WEIGHTS_DIR + "/warmup_ex_model",
                "task": "ex"
            }

        # Testing:
        # this ex/test pair is a repeat. First one only differs
        # in save directory and batch size (16 before, 24 now)
        # elif self.task == "ex" and self.action == "test":
        #     self.d = {
        #         "batch_size": 24,
        #         "gpus": 1,
        #         "action": "test",
        #         "model_str": "bert-base-cased",
        #         # "save": WEIGHTS_DIR + "/warmup_ex_model",
        #         "task": "ex"
        #     }

        # Predicting
        # this ex/predict pair is a repeat. First one only differs
        # in save directory
        # elif self.task == "ex" and self.action == "predict":
        #     self.d = {
        #         "gpus": 1,
        #         # "inp": "sentences.txt",
        #         "action": "predict",
        #         "model_str": "bert-base-cased",
        #         #"out": "predictions.txt",
        #         # "save": WEIGHTS_DIR + "/warmup_ex_model",
        #         "task": "ex"
        #     }

        ### Constrained self.model
        # Training
        elif self.task == "ex" and self.action == "resume":
            # error in openie6 paper
            #         "lr": 5e-6, and "lr: 2e-5

            self.d = {
                "accumulate_grad_batches": 2,
                "batch_size": 16,
                "constraint_str": "posm_hvc_hvr_hve",
                "con_weight_str": "3_3_3_3",
                "num_epochs": 16,
                "gpus": 1,
                "gradient_clip_val": 1,
                "num_iterative_layers": 2,
                "lr": 2E-5,
                "action": "resume",
                "model_str": "bert-base-cased",
                "multi_opt": True,
                "optimizer": "adam",
                # "save": WEIGHTS_DIR + "/ex_model",
                "save_k": 3,
                "task": "ex",
                "val_check_interval": 0.1,
                "wreg": 1
            }
        # Testing
        elif self.task == "ex" and self.action == "test":
            self.d = {
                "batch_size": 16,
                "gpus": 1,
                "action": "test",
                "model_str": "bert-base-cased",
                # "save": WEIGHTS_DIR + "/ex_model",
                "task": "ex"
            }

        # Predicting
        elif self.task == "ex" and self.action == "predict":
            self.d = {
                "gpus": 1,
                # "inp": "sentences.txt",
                "action": "predict",
                "model_str": "bert-base-cased",
                # "out": "predictions.txt",
                # "save": WEIGHTS_DIR + "/ex_model",
                "task": "ex"
            }

        ### Running CCNode Analysis
        elif self.task == "cc" and self.action == "train_test":
            self.d = {
                "batch_size": 32,
                "num_epochs": 40,
                "gpus": 1,
                "num_iterative_layers": 2,
                "lr": 2E-5,
                "action": "train_test",
                "model_str": "bert-large-cased",
                "optimizer": "adamW",
                # "save": WEIGHTS_DIR + "/cc",
                "task": "cc"
            }

        ### Final self.model

        # Running
        # The splitpredict self.action was stated already at the beginning.
        # It is a repeat.
        elif self.task == "ex" and self.action == "splitpredict":
            self.d = {
                "cc_weights_fp":
                    WEIGHTS_DIR + "/cc_epoch=28_eval_acc=0.854.ckpt",
                "gpus": 1,
                # "inp": "carb_subset/data/carb_sentences.txt",
                "action": "splitpredict",
                # "num_extractions": EX_NUM_DEPTHS,
                "ex_weights_fp":
                    WEIGHTS_DIR + "/ex_epoch=14_eval_acc=0.551_v0.ckpt",
                # "out": WEIGHTS_DIR + "/results/final",
                # "rescore_model":
                # "rescoring:"
                "task": "ex"
            }
        else:
            self.d = {}
            print("****self.d is empty without defaults")

        default_d = \
            {
                "accumulate_grad_batches": 1,  # torch default is 1
                "batch_size": 32,
                "con_weight_str": "1",  # for multiple constraints "1_1_1"
                "dropout_fun": 0.0,
                "gpus": 1,
                "gradient_clip_val": 5,
                "lr": 2E-5,
                "model_str": "bert-base-cased",
                #"num_extractions": EX_NUM_DEPTHS,
                "num_iterative_layers": 2,
                "num_steps_per_epoch": None,
                "optimizer": "adamW",
                "refresh_cache": False,
                "save_k": 1,
                "split_only": False,
                "small_train": False,
                "val_check_interval": 1.0,
                "verbose": False,
                "wreg": 0
            }

        from sax_utils import merge_dicts
        self.d = merge_dicts(self.d, default_d)

        return self.d