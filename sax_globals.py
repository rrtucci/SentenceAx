"""


hparams = hyperparamters,
fp = file path
params_d = pamameters dictionary


"""
from sax_utils import none_dd, merge_dicts

# global paths

# # file paths for training, tuning and testing cctags (cc=conjunction)
# CCTAGS_TRAIN_FP = "input_data/cctags_train.txt"
# # dev=development=validation=tuning
# CCTAGS_TUNE_FP = "input_data/cctags_tune.txt"
# CCTAGS_TEST_FP = "input_data/cctags_test.txt"

CCTAGS_TRAIN_FP = 'input_data/openie-data/ptb-train.labels'
CCTAGS_TUNE_FP = 'input_data/openie-data/ptb-dev.labels'
CCTAGS_TEST_FP = 'input_data/openie-data/ptb-test.labels'


# # file paths for training, tuning and testing extags (ex=extraction)
# EXTAGS_FP = "input_data/extags_all.txt"
# EXTAGS_TRAIN_FP = "input_data/extags_train.txt"
# # dev=development=validation=tuning
# EXTAGS_TUNE_FP = "input_data/extags_tune.txt"
# EXTAGS_TEST_FP = "input_data/extags_test.txt"

EXTAGS_TRAIN_FP = 'input_data/openie-data/openie4_labels'
EXTAGS_TUNE_FP = 'input_data/carb-data/dev.txt'
EXTAGS_TEST_FP = 'input_data/carb-data/test.txt'

# sentences used for prediction
INP_FP = "carb_subset/data/carb_sentences.txt"

CACHE_DIR = 'input_data/pretrained_cache' # used by AutoModel and AutoTokenizer
WEIGHTS_DIR = "weights"
PREDICTIONS_DIR = "predictions"
RESCORE_DIR = "rescore"

QUOTES = "\"\'" #2
BRACKETS = "(){}[]<>" #8
SEPARATORS = ",:;&-" #5
ARITHMETICAL = "*|\/@#$%^+=~_" #13
ENDING = ".?!" #3
PUNCT_MARKS = QUOTES + BRACKETS + SEPARATORS + ARITHMETICAL + ENDING

UNUSED_TOKENS = ["[unused1]", "[unused2]", "[unused3]"]
UNUSED_TOKENS_STR = " " + " ".join(UNUSED_TOKENS)

DROPOUT = 0.0
NUM_EMBEDDINGS = 100
MAX_EXTRACTION_LENGTH = 5

BOS_LABEL = 101 # bos = begin of sentence
EOS_LABEL = 102 # eos = end of sentence
NUM_LABELS = 6
LABELLING_DIM = 300
EXTAG_TO_LABEL = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                   'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}
BASE_EXTAGS = EXTAG_TO_LABEL.keys()
LABEL_TO_EXTAG={0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
                 4: 'ARG2', 5: 'NONE'}

CCTAG_TO_LABEL = {'NONE': 0, 'CP': 1, 'CP_START': 2,
                   'CC': 3, 'SEP': 4, 'OTHERS': 5}
BASE_CCTAGS = CCTAG_TO_LABEL.keys()

# LIGHT_VERBS = [
#     "take", "have", "give", "do", "make", "has", "have",
#     "be", "is", "were", "are", "was", "had", "being",
#     "began", "am", "following", "having", "do",
#     "does", "did", "started", "been", "became",
#     "left", "help", "helped", "get", "keep",
#     "think", "got", "gets", "include", "suggest",
#     "used", "see", "consider", "means", "try",
#     "start", "included", "lets", "say", "continued",
#     "go", "includes", "becomes", "begins", "keeps",
#     "begin", "starts", "said", "stop", "begin",
#     "start", "continue", "say"]

# in alphabetical order, eliminated repeats begin(2), do(2), say(2), start(2)
LIGHT_VERBS = [
    "am", "are", "be", "became", "becomes", "begin",
    "began", "begins", "being", "continue",
    "continued", "consider", "do", "does", "did", "following",
    "get", "gets", "give", "go", "got", "had",
    "have", "have", "having", "help", "helped",
    "include", "included", "includes", "include", "is",
    "kept", "keep", "keeps", "left", "lets", "make", "means",
    "said", "say", "see", "start", "started",
    "starts", "stop", "suggest", "take", "think",
    "try", "used", "were", "was"
]

# 15 words in alphabetical order
UNBREAKABLE_WORDS = \
    ['addition', 'aggregate', 'amount', 'among', 'average',
     'between', 'center', 'equidistant', 'gross', 'mean',
     'median', 'middle', 'sum', 'total', 'value']

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
    TAG_TO_LABEL = EXTAG_TO_LABEL
    MAX_DEPTH = MAX_EXTRACTION_LENGTH
elif TASK == "cc":
    TAG_TO_LABEL = CCTAG_TO_LABEL
    MAX_DEPTH = 3
else:
    assert False

assert MODE in ["predict", "train_test", "splitpredict",
                "resume", "test"]

# Do not define capitalized global and PARAMS_D key for the same
# parameter. Define one or the other but not both

if "PARAMS_D" in globals():
    print("PARAM_D was defined prior to running sax_globals.py")

    # define `PARAMS_D` in jupyter notebook before running any
    # subroutines that use it. The file `custom_params_d.txt` gives
    # some pointers on how to define a custom params_d.

## Running Model
elif TASK == "ex" and MODE == "splitpredict":
    PARAMS_D = {
        "cc_model_fp": WEIGHTS_DIR + 
                    "/cc_model-epoch=28_eval_acc=0.854.ckpt",
        "gpus": 1,
        # "inp": "sentences.txt",
        "mode": "splitpredict",
        "num_extractions": 5,
        "ex_model_fp": WEIGHTS_DIR + 
                    "/ex_model-epoch=14_eval_acc=0.551_v0.ckpt",
        #"out": "predictions.txt",
        # "rescore_model": WEIGHTS_DIR + "/rescore_model",
        "rescoring": True,
        "task": "ex"
    }

## Training Model

### Warmup Model
# Training:
elif TASK == "ex" and MODE == "train_test":
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
elif TASK == "ex" and MODE == "test":
    PARAMS_D = {
        "batch_size": 24,
        "gpus": 1,
        "mode": "test",
        "model_str": "bert-base-cased",
        # "save": WEIGHTS_DIR + "/warmup_ex_model",
        "task": "ex"
    }

# Predicting
elif TASK == "ex" and MODE == "predict":
    PARAMS_D = {
        "gpus": 1,
        # "inp": "sentences.txt",
        "mode": "predict",
        "model_str": "bert-base-cased",
        #"out": "predictions.txt",
        # "save": WEIGHTS_DIR + "/warmup_ex_model",
        "task": "ex"
    }

### Constrained Model
# Training
elif TASK == "ex" and MODE == "resume":
    # error in openie6 paper
    #         "lr": 5e-6, and "lr: 2e-5
    
    PARAMS_D = {
        "accumulate_grad_batches": 2,
        "batch_size": 16,
        "checkpoint_fp": WEIGHTS_DIR + 
                      "/warmup_ex_model-epoch=13_eval_acc=0.544.ckpt",
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
        #"out": "predictions.txt",
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
# elif TASK == "ex" and MODE == "splitpredict":
#     PARAMS_D = {
#         "cc_model_fp": WEIGHTS_DIR + 
#                     "/cc_model-epoch=28_eval_acc=0.854.ckpt",
#         "gpus": 1,
#         # "inp": "carb_subset/data/carb_sentences.txt",
#         "mode": "splitpredict",
#         "num_extractions": 5,
#         "ex_model_fp": WEIGHTS_DIR + 
#                     "/ex_model-epoch=14_eval_acc=0.551_v0.ckpt",
#         # "out": WEIGHTS_DIR + "/results/final",
#         # "rescore_model": WEIGHTS_DIR + "/rescore_model",
#         "rescoring": True,
#         "task": "ex"
#     }
else:
    assert False


DEFAULT_PARAMS_D=\
{
    "batch_size": 32,
    "checkpoint_fp": "",
    "cweights": 1,
    "dropout": 0.0,
    "gpus": 1,
    "iterative_layers": 2,
    "lr": 2E-5,
    "model_str": "bert-base-cased",
    "num_extractions": 5,
    "optimizer": "adamW",
    "save_k": 1,
    "val_check_interval": 1.0,
    "wreg": 0
}
PARAMS_D = merge_dicts(PARAMS_D, DEFAULT_PARAMS_D)