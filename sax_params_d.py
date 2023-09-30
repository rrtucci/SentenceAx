from sax_globals import *

TASK_MODE_d = {
0: ("", ""),
1: ("ex", "train_test"),
2: ("ex", "test"),
3: ("ex", "predict"),
4: ("ex", "resume"),
5: ("cc", "train_test"),
6: ("ex", "splitpredict")
}

TASK, MODE = TASK_MODE_d[2]


# Possible choices for PARAMS_D
# 0. "", ""
# 1. ex, train_test
#    Conductor.train() followed by Conductor.test()
# 2. ex, test  (appears twice)
#     Conductor.test()
# 3. ex, predict (appears twice)
#     Conductor.predict()
# 4. ex, resume
#     Conductor.resume()
# 5. cc, train_test
#    Conductor.train() followed by Conductor.test()
# 6. ex, splitpredict (appears twice)
#    Conductor.splitpredict()



# I use "ex" instead of "oie" for task
# I use "cc" instead of "conj" for task

# choose TASK and MODE before starting
# TASK in "ex", "cc"
# choose "ex" task if doing both "ex" and "cc". Choose "cc" task only
# when doing "cc" only
# MODE in ("train_test", "resume", "test", "predict", "splitpredict")

#
# hparams = hyperparameters,
# fp = file path
# params_d = parameters dictionary


if TASK == "":
    print("******************************TASK is empty")
else:
    assert TASK in ["ex", "cc"]
    print("TASK=", TASK)

if MODE == "":
    print("******************************MODE is empty")
else:
    assert MODE in ["predict", "train_test", "splitpredict",
                    "resume", "test"]
    print("MODE=", MODE)


# Do not define capitalized global and PARAMS_D key for the same
# parameter. Define one or the other but not both


    # define `PARAMS_D` in jupyter notebook before running any
    # subroutines that use it. The file `custom_params_d.txt` gives
    # some pointers on how to define a custom params_d.

def PARAMS_D():
    ## Running Model
    # this is repeated at begin and end.
    # elif TASK == "ex" and MODE == "splitpredict":
    #     PARAMS_D = {
    #         "cc_model_fp": WEIGHTS_DIR +
    #                     "/cc_model/epoch=28_eval_acc=0.854.ckpt",
    #         "gpus": 1,
    #         # "inp": "sentences.txt",
    #         "mode": "splitpredict",
    #         "num_extractions": 5,
    #         "ex_model_fp": WEIGHTS_DIR +
    #                     "/ex_model/epoch=14_eval_acc=0.551_v0.ckpt",
    #         #"out": "predictions.txt",
    #         # "rescore_model": WEIGHTS_DIR + "/rescore_model",
    #         "rescoring": True,
    #         "task": "ex"
    #     }

    ## Training Model

    ### Warmup Model
    # Training:
    if TASK == "ex" and MODE == "train_test":
        PARAMS_D = {
            "batch_size": 24,
            "epochs": 30,
            "gpus": 1,
            "iterative_layers": 2,
            "lr": 2E-5,
            "mode": "train_test",
            "model_str": "bert-base-cased",
            "optimizer": "adamW",
            # "save": WEIGHTS_DIR + "/warmup_ex_model",
            "task": "ex"
        }

    # Testing:
    # this ex/test pair is a repeat. First one only differs
    # in save directory and batch size (16 before, 24 now)
    # elif TASK == "ex" and MODE == "test":
    #     PARAMS_D = {
    #         "batch_size": 24,
    #         "gpus": 1,
    #         "mode": "test",
    #         "model_str": "bert-base-cased",
    #         # "save": WEIGHTS_DIR + "/warmup_ex_model",
    #         "task": "ex"
    #     }

    # Predicting
    # this ex/predict pair is a repeat. First one only differs
    # in save directory
    # elif TASK == "ex" and MODE == "predict":
    #     PARAMS_D = {
    #         "gpus": 1,
    #         # "inp": "sentences.txt",
    #         "mode": "predict",
    #         "model_str": "bert-base-cased",
    #         #"out": "predictions.txt",
    #         # "save": WEIGHTS_DIR + "/warmup_ex_model",
    #         "task": "ex"
    #     }

    ### Constrained Model
    # Training
    elif TASK == "ex" and MODE == "resume":
        # error in openie6 paper
        #         "lr": 5e-6, and "lr: 2e-5

        PARAMS_D = {
            "accumulate_grad_batches": 2,
            "batch_size": 16,
            "checkpoint_fp":
                WEIGHTS_DIR + "/warmup_ex_model-epoch=13_eval_acc=0.544.ckpt",
            "constraints": "posm_hvc_hvr_hve",
            "cweights": "3_3_3_3",
            "epochs": 16,
            "gpus": 1,
            "gradient_clip_val": 1,
            "iterative_layers": 2,
            "lr": 2E-5,
            "mode": "resume",
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
    elif TASK == "ex" and MODE == "test":
        PARAMS_D = {
            "batch_size": 16,
            "gpus": 1,
            "mode": "test",
            "model_str": "bert-base-cased",
            # "save": WEIGHTS_DIR + "/ex_model",
            "task": "ex"
        }

    # Predicting
    elif TASK == "ex" and MODE == "predict":
        PARAMS_D = {
            "gpus": 1,
            # "inp": "sentences.txt",
            "mode": "predict",
            "model_str": "bert-base-cased",
            # "out": "predictions.txt",
            # "save": WEIGHTS_DIR + "/ex_model",
            "task": "ex"
        }

    ### Running CCNode Analysis
    elif TASK == "cc" and MODE == "train_test":
        PARAMS_D = {
            "batch_size": 32,
            "epochs": 40,
            "gpus": 1,
            "iterative_layers": 2,
            "lr": 2E-5,
            "mode": "train_test",
            "model_str": "bert-large-cased",
            "optimizer": "adamW",
            # "save": WEIGHTS_DIR + "/cc_model",
            "task": "cc"
        }

    ### Final Model

    # Running
    # The splitpredict mode was stated already at the beginning.
    # It is a repeat.
    elif TASK == "ex" and MODE == "splitpredict":
        PARAMS_D = {
            "cc_model_fp":
                WEIGHTS_DIR + "/cc_model-epoch=28_eval_acc=0.854.ckpt",
            "gpus": 1,
            # "inp": "carb_subset/data/carb_sentences.txt",
            "mode": "splitpredict",
            "num_extractions": MAX_EX_DEPTH,
            "ex_model_fp":
                WEIGHTS_DIR + "/ex_model/epoch=14_eval_acc=0.551_v0.ckpt",
            # "out": WEIGHTS_DIR + "/results/final",
            # "rescore_model": WEIGHTS_DIR + "/rescore_model",
            "rescoring": True,
            "task": "ex"
        }
    else:
        PARAMS_D = {}
        print("***********************PARAMS_D is empty")

    DEFAULT_PARAMS_D = \
        {
            "batch_size": 32,
            "build_cache": True,
            "checkpoint_fp": "",
            "cweights": 1,
            "dropout_fun": 0.0,
            "gpus": 1,
            "iterative_layers": 2,
            "lr": 2E-5,
            "model_str": "bert-base-cased",
            "num_extractions": MAX_EX_DEPTH,
            "optimizer": "adamW",
            "save_k": 1,
            "val_check_interval": 1.0,
            "wreg": 0
        }

    from sax_utils import merge_dicts
    PARAMS_D = merge_dicts(PARAMS_D, DEFAULT_PARAMS_D)

    return PARAMS_D

def MAX_DEPTH():
    if TASK == "ex":
        return MAX_EX_DEPTH
    elif TASK == "cc":
        return MAX_CC_DEPTH

def TAG_TO_ILABEL():
    if TASK == "ex":
        return EXTAG_TO_ILABEL
    elif TASK == "cc":
        return CCTAG_TO_ILABEL

def LOG_DIR():
    if TASK == "ex":
        return WEIGHTS_DIR + '/ex_logs'
    elif TASK == "cc":
        return WEIGHTS_DIR + '/cc_logs'
