from sax_globals import *

class Params:
    """
        Possible choices for self.d
        0. "", ""
        1. ex, train_test
           Conductor.train() followed by Conductor.test()
        2. ex, test  (appears twice)
            Conductor.test()
        3. ex, predict (appears twice)
            Conductor.predict()
        4. ex, resume
            Conductor.resume()
        5. cc, train_test
           Conductor.train() followed by Conductor.test()
        6. ex, splitpredict (appears twice)
           Conductor.splitpredict()



        I use "ex" instead of "oie" for self.task
        I use "cc" instead of "conj" for self.task

        choose self.task and self.mode before starting
        self.task in "ex", "cc"
        choose "ex" self.task if doing both "ex" and "cc". Choose "cc" self.task only
        when doing "cc" only
        self.mode in ("train_test", "resume", "test", "predict", "splitpredict")


        hparams = hyperparameters,
        fp = file path
        self.d = parameters dictionary


    Attributes
    ----------
    d: dict[str, Any]
    id: int
    mode: str
    task: str

    """
    def __init__(self, id):
        """

        Parameters
        ----------
        id: int
        """
        self.id = id
        id_to_pair = {
        0: ("", ""),
        1: ("ex", "train_test"),
        2: ("ex", "test"),
        3: ("ex", "predict"),
        4: ("ex", "resume"),
        5: ("cc", "train_test"),
        6: ("ex", "splitpredict")
        }
        
        self.task, self.mode = id_to_pair[id]
        self.d = self.get_d()

    def get_d(self):  
        if self.task == "":
            print("******************************self.task is empty")
        else:
            assert self.task in ["ex", "cc"]
            print("self.task=", self.task)
        
        if self.mode == "":
            print("******************************self.mode is empty")
        else:
            assert self.mode in ["predict", "train_test", "splitpredict",
                            "resume", "test"]
            print("self.mode=", self.mode)
        
        
        # Do not define capitalized global and self.d key for the same
        # parameter. Define one or the other but not both
        
        
            # define `self.d` in jupyter notebook before running any
            # subroutines that use it. The file `custom_self.d.txt` gives
            # some pointers on how to define a custom self.d.
        
    def get_d(self):
        ## Running self.model
        # this is repeated at begin and end.
        # elif self.task == "ex" and self.mode == "splitpredict":
        #     self.d = {
        #         "cc_weights_fp": WEIGHTS_DIR +
        #                     "/cc_model/epoch=28_eval_acc=0.854.ckpt",
        #         "gpus": 1,
        #         # "inp": "sentences.txt",
        #         "self.mode": "splitpredict",
        #         "num_extractions": 5,
        #         "ex_weights_fp": WEIGHTS_DIR +
        #                     "/ex_model/epoch=14_eval_acc=0.551_v0.ckpt",
        #         #"out": "predictions.txt",
        #         # "rescore_model": WEIGHTS_DIR + "/rescore_model",
        #         "rescoring": True,
        #         "self.task": "ex"
        #     }

        ## Training self.model

        ### Warmup self.model
        # Training:
        if self.task == "ex" and self.mode == "train_test":
            self.d = {
                "batch_size": 24,
                "epochs": 30,
                "gpus": 1,
                "iterative_layers": 2,
                "lr": 2E-5,
                "self.mode": "train_test",
                "self.model_str": "bert-base-cased",
                "optimizer": "adamW",
                # "save": WEIGHTS_DIR + "/warmup_ex_model",
                "self.task": "ex"
            }

        # Testing:
        # this ex/test pair is a repeat. First one only differs
        # in save directory and batch size (16 before, 24 now)
        # elif self.task == "ex" and self.mode == "test":
        #     self.d = {
        #         "batch_size": 24,
        #         "gpus": 1,
        #         "self.mode": "test",
        #         "self.model_str": "bert-base-cased",
        #         # "save": WEIGHTS_DIR + "/warmup_ex_model",
        #         "self.task": "ex"
        #     }

        # Predicting
        # this ex/predict pair is a repeat. First one only differs
        # in save directory
        # elif self.task == "ex" and self.mode == "predict":
        #     self.d = {
        #         "gpus": 1,
        #         # "inp": "sentences.txt",
        #         "self.mode": "predict",
        #         "self.model_str": "bert-base-cased",
        #         #"out": "predictions.txt",
        #         # "save": WEIGHTS_DIR + "/warmup_ex_model",
        #         "self.task": "ex"
        #     }

        ### Constrained self.model
        # Training
        elif self.task == "ex" and self.mode == "resume":
            # error in openie6 paper
            #         "lr": 5e-6, and "lr: 2e-5

            self.d = {
                "accumulate_grad_batches": 2,
                "batch_size": 16,
                "suggested_checkpoint_fp":
                    WEIGHTS_DIR + "/warmup_ex_model-epoch=13_eval_acc=0.544.ckpt",
                "constraint_str": "posm_hvc_hvr_hve",
                "con_weight_str": "3_3_3_3",
                "epochs": 16,
                "gpus": 1,
                "gradient_clip_val": 1,
                "iterative_layers": 2,
                "lr": 2E-5,
                "self.mode": "resume",
                "self.model_str": "bert-base-cased",
                "multi_opt": True,
                "optimizer": "adam",
                # "save": WEIGHTS_DIR + "/ex_model",
                "save_k": 3,
                "self.task": "ex",
                "val_check_interval": 0.1,
                "wreg": 1
            }
        # Testing
        elif self.task == "ex" and self.mode == "test":
            self.d = {
                "batch_size": 16,
                "gpus": 1,
                "self.mode": "test",
                "self.model_str": "bert-base-cased",
                # "save": WEIGHTS_DIR + "/ex_model",
                "self.task": "ex"
            }

        # Predicting
        elif self.task == "ex" and self.mode == "predict":
            self.d = {
                "gpus": 1,
                # "inp": "sentences.txt",
                "self.mode": "predict",
                "self.model_str": "bert-base-cased",
                # "out": "predictions.txt",
                # "save": WEIGHTS_DIR + "/ex_model",
                "self.task": "ex"
            }

        ### Running CCNode Analysis
        elif self.task == "cc" and self.mode == "train_test":
            self.d = {
                "batch_size": 32,
                "epochs": 40,
                "gpus": 1,
                "iterative_layers": 2,
                "lr": 2E-5,
                "self.mode": "train_test",
                "self.model_str": "bert-large-cased",
                "optimizer": "adamW",
                # "save": WEIGHTS_DIR + "/cc_model",
                "self.task": "cc"
            }

        ### Final self.model

        # Running
        # The splitpredict self.mode was stated already at the beginning.
        # It is a repeat.
        elif self.task == "ex" and self.mode == "splitpredict":
            self.d = {
                "cc_weights_fp":
                    WEIGHTS_DIR + "/cc_model-epoch=28_eval_acc=0.854.ckpt",
                "gpus": 1,
                # "inp": "carb_subset/data/carb_sentences.txt",
                "self.mode": "splitpredict",
                "num_extractions": MAX_EX_DEPTH,
                "ex_weights_fp":
                    WEIGHTS_DIR + "/ex_model/epoch=14_eval_acc=0.551_v0.ckpt",
                # "out": WEIGHTS_DIR + "/results/final",
                # "rescore_model": WEIGHTS_DIR + "/rescore_model",
                "rescoring": True,
                "self.task": "ex"
            }
        else:
            self.d = {}
            print("***********************self.d is empty")

        default_d = \
            {
                "batch_size": 32,
                "build_cache": True,
                "suggested_checkpoint_fp": "",
                "con_weight_str": "1",  # for multiple constraints "1_1_1"
                "dropout_fun": 0.0,
                "gpus": 1,
                "iterative_layers": 2,
                "lr": 2E-5,
                "self.model_str": "bert-base-cased",
                "num_extractions": MAX_EX_DEPTH,
                "optimizer": "adamW",
                "save_k": 1,
                "val_check_interval": 1.0,
                "wreg": 0
            }

        from sax_utils import merge_dicts
        self.d = merge_dicts(self.d, default_d)

        return self.d
    
    def num_depths(self):
        """

        Returns
        -------
        int

        """
        if self.task == "ex":
            return MAX_EX_DEPTH
        elif self.task == "cc":
            return MAX_CC_DEPTH
    
    def tag_to_ilabel(self):
        """

        Returns
        -------
        dict[str, int]

        """
        if self.task == "ex":
            return EXTAG_TO_ILABEL
        elif self.task == "cc":
            return CCTAG_TO_ILABEL
    
    def log_dir(self):
        """

        Returns
        -------
        str

        """
        if self.task == "ex":
            return WEIGHTS_DIR + '/ex_logs'
        elif self.task == "cc":
            return WEIGHTS_DIR + '/cc_logs'
