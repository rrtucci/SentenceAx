from sax_utils import *

# hparams = hyperparamters,
# fp = file path
# params_d = pamameters dictionary

EXT_SAMPLES_FP = "data/ext_samples.txt"
# file paths for training, tuning and testing cctags (cc=conjunction)
CCTAGS_TRAIN_FP = "data/cctags_train.txt"
# dev=development=validation=tuning
CCTAGS_TUNE_FP = "data/cctags_tune.txt"
CCTAGS_TEST_FP = "data/cctags_test.txt"

# file paths for training, tuning and testing extags (ex=extraction)
EXTAGS_FP = "data/extags_all.txt"
EXTAGS_TRAIN_FP = "data/extags_train.txt"
# dev=development=validation=tuning
EXTAGS_TUNE_FP = "data/extags_tune.txt"
EXTAGS_TEST_FP = "data/extags_test.txt"
BOS_TOKEN_ID = 101 # bos = begin of sentence
EOS_TOKEN_ID = 102 # eos = end of sentence

CACHE_DIR = 'data/pretrained_cache'
NUM_LABELS = 6
MAX_EXTRACTION_LENGTH = 5
UNUSED_TOKENS = ["[unused1]", "[unused2]", "[unused3]"]
UNUSED_TOKENS_STR = " " + " ".join(UNUSED_TOKENS)
# NUM_EMBEDDINGS = 100

EXTAG_TO_ILABEL = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                   'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}

CCTAG_TO_ILABEL = {'NONE': 0, 'CP': 1, 'CP_START': 2,
                   'CC': 3, 'SEP': 4, 'OTHERS': 5, }

LIGHT_VERBS = [
    "take", "have", "give", "do", "make", "has", "have",
    "be", "is", "were", "are", "was", "had", "being",
    "began", "am", "following", "having", "do",
    "does", "did", "started", "been", "became",
    "left", "help", "helped", "get", "keep",
    "think", "got", "gets", "include", "suggest",
    "used", "see", "consider", "means", "try",
    "start", "included", "lets", "say", "continued",
    "go", "includes", "becomes", "begins", "keeps",
    "begin", "starts", "said", "stop", "begin",
    "start", "continue", "say"]

# I use "ex" instead of "oie" for task
# I use "cc" instead of "conj" for task

# choose TASK and MODE before starting
# TASK in "ex", "cc"
# choose "ex" task if doing both "ex" and "cc". Choose "cc" task only
# when doing "cc" only
# MODE in ("predict", "train_test", "splitpredict", "resume", "test")

assert "TASK" in globals()
TASK = globals()["TASK"]
print('you\'ve entered TASK= "' + TASK + '"')

assert "MODE" in globals()
MODE = globals()["MODE"]
print('you\'ve entered MODE= "' + MODE + '"')

if TASK == "ex":
    TAG_TO_ILABEL = EXTAG_TO_ILABEL
    MAX_DEPTH = MAX_EXTRACTION_LENGTH
elif TASK == "cc":
    TAG_TO_ILABEL = CCTAG_TO_ILABEL
    MAX_DEPTH = 3
else:
    assert False

assert MODE in ["predict", "train_test", "splitpredict",
                "resume", "test"]

if "PARAMS_D" in globals():

    # define `PARAMS_D` in jupyter notebook before running any
    # subroutines that use it. The file `custom_params_d.txt` gives
    # some pointers on how to define a custom params_d.

## Running Model
elif TASK == "ex" and MODE == "splitpredict":
    PARAMS_D = none_dd({
        "conj_model": "models/conj_model/epoch=28_eval_acc=0.854.ckpt",
        "gpus": 1,
        "inp": "sentences.txt",
        "mode": "splitpredict",
        "num_extractions": 5,
        "oie_model": "models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt",
        "out": "predictions.txt",
        "rescore_model": "models/rescore_model",
        "rescoring": True,
        "task": "ex"
    })

## Training Model

### Warmup Model
# Training:
elif TASK == "ex" and MODE == "train_test":
    PARAMS_D = none_dd({
        "batch_size": 24,
        "epochs": 30,
        "gpus": 1,
        "iterative_layers": 2,
        "lr": 2E-5,
        "mode": "train_test",
        "model_str": "bert-base-cased",
        "optimizer": "adamW",
        "save": "models/warmup_oie_model",
        "task": "ex"
    })
    
# Testing:
elif TASK == "ex" and MODE == "test":
    PARAMS_D = none_dd({
        "batch_size": 24,
        "gpus": 1,
        "mode": "test",
        "model_str": "bert-base-cased",
        "save": "models/warmup_oie_model",
        "task": "ex"
    })

# Predicting
elif TASK == "ex" and MODE == "predict":
    PARAMS_D = none_dd({
        "gpus": 1,
        "inp": "sentences.txt",
        "mode": "predict",
        "model_str": "bert-base-cased",
        "out": "predictions.txt",
        "save": "models/warmup_oie_model",
        "task": "ex"
    })

### Constrained Model
# Training
elif TASK == "ex" and MODE == "resume":
    # error in openie6 paper
    #         "lr": 5e-6, and "lr: 2e-5
    
    PARAMS_D = none_dd({
        "accumulate_grad_batches": 2,
        "batch_size": 16,
        "checkpoint": "models/warmup_oie_model/epoch=13_eval_acc=0.544.ckpt",
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
        "save": "models/oie_model",
        "save_k": 3,
        "task": "ex",
        "val_check_interval": 0.1,
        "wreg": 1
    })
# Testing
elif TASK == "ex" and MODE == "test":
    PARAMS_D = none_dd({
        "batch_size": 16,
        "gpus": 1,
        "mode": "test",
        "model_str": "bert-base-cased",
        "save": "models/oie_model",
        "task": "ex"
    })

# Predicting
elif TASK == "ex" and MODE == "predict":
    PARAMS_D = none_dd({
        "gpus": 1,
        "inp": "sentences.txt",
        "mode": "predict",
        "model_str": "bert-base-cased",
        "out": "predictions.txt",
        "save": "models/oie_model",
        "task": "ex"
    })

### Running CCNode Analysis
elif TASK == "cc" and MODE == "train_test":
    PARAMS_D = none_dd({
        "batch_size": 32,
        "epochs": 40,
        "gpus": 1,
        "iterative_layers": 2,
        "lr": 2E-5,
        "mode": "train_test",
        "model_str": "bert-large-cased",
        "optimizer": "adamW",
        "save": "models/conj_model",
        "task": "cc"
    })

### Final Model

# Running
elif TASK == "ex" and MODE == "splipredict":
    PARAMS_D = none_dd({
        "conj_model": "models/conj_model/epoch=28_eval_acc=0.854.ckpt",
        "gpus": 1,
        "inp": "carb/data/carb_sentences.txt",
        "mode": "splitpredict",
        "num_extractions": 5,
        "oie_model": "models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt",
        "out": "models/results/final",
        "rescore_model": "models/rescore_model",
        "rescoring": True,
        "task": "ex"
    })

