"""
IMPORTANT:
Must import this file with star
from sax_globals import *
This will not do:
import sax_globals

hparams = hyperparameters,
fp = file path
params_d = parameters dictionary


"""

# global paths

# # file paths for training, tuning and testing cctags (cc=conjunction)
# CCTAGS_TRAIN_FP = "input_data/cctags_train.txt"
# # dev=development=validation=tuning
# CCTAGS_TUNE_FP = "input_data/cctags_tune.txt"
# CCTAGS_TEST_FP = "input_data/cctags_test.txt"

# # file paths for training, tuning and testing extags (ex=extraction)
# EXTAGS_TRAIN_FP = "input_data/extags_train.txt"
# # dev=development=validation=tuning
# EXTAGS_TUNE_FP = "input_data/extags_tune.txt"
# EXTAGS_TEST_FP = "input_data/extags_test.txt"

INPUT_DIR = "input_data"
TTT_CACHE_DIR = INPUT_DIR + '/pretrained_cache'
WEIGHTS_DIR = "weights"
PRED_DIR = "predictions"
CC_METRIC_STORAGE_DIR = "cc_metric_storage"

CCTAGS_TRAIN_FP = 'input_data/openie-data/ptb-train.labels'
CCTAGS_TUNE_FP = 'input_data/openie-data/ptb-dev.labels'
CCTAGS_TEST_FP = 'input_data/openie-data/ptb-test.labels'
CC_FIN_WEIGHTS_FP = WEIGHTS_DIR + '/cc.txt'

EXTAGS_TRAIN_FP = INPUT_DIR + '/openie-data/openie4_labels'
EXTAGS_TUNE_FP = INPUT_DIR + '/carb-data/dev.txt'
EXTAGS_TEST_FP = INPUT_DIR + '/carb-data/test.txt'
EX_FIN_WEIGHTS_FP = WEIGHTS_DIR + '/ex.txt'

PRED_IN_FP = PRED_DIR + "/carb_sentences"
PRED_OUT_FP = PRED_DIR + "/out.txt"
PRED_ALLEN_OUT_FP = PRED_DIR + "/allen_out.txt"
RE_ALLEN_IN_FP = PRED_ALLEN_OUT_FP
RE_ALLEN_OUT_FP = PRED_DIR + "/re_allen_out.txt"

QUOTES = "\"\'"  # 2
BRACKETS = "(){}[]<>"  # 8
SEPARATORS = ",:;&-"  # 5
ARITHMETICAL = "*|\/@#$%^+=~_"  # 13
ENDING = ".?!"  # 3
PUNCT_MARKS = QUOTES + BRACKETS + SEPARATORS + ARITHMETICAL + ENDING

UNUSED_TOKENS = ["[unused1]", "[unused2]", "[unused3]"]
UNUSED_TOKENS_STR = " " + " ".join(UNUSED_TOKENS)

DROPOUT = 0.0
EX_NUM_DEPTHS = 5
CC_NUM_DEPTHS = 3

PAD_ICODE = 0  # ipad
NUM_ICODES = 100
BOS_ICODE = 101  # bos = begin of sentence
EOS_ICODE = 102  # eos = end of sentence

NUM_ILABELS = 6
ILABELLING_DIM = 300

EXTAG_TO_ILABEL = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                   'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}
BASE_EXTAGS = EXTAG_TO_ILABEL.keys()
ILABEL_TO_EXTAG = {0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
                   4: 'ARG2', 5: 'NONE'}

CCTAG_TO_ILABEL = {'NONE': 0, 'CP': 1, 'CP_START': 2,
                   'CC': 3, 'SEP': 4, 'OTHERS': 5}
BASE_CCTAGS = CCTAG_TO_ILABEL.keys()
ILABEL_TO_CCTAG = {0: 'NONE', 1: 'CP', 2: 'CP_START',
                   3: 'CC', 4: 'SEP', 5: 'OTHERS'}

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
