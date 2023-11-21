"""
SentenceAx global variables

IMPORTANT:
Must import this file with star
`from sax_globals import *`
This will not do:
`import sax_globals`

fp = file path

osent = original sentence
osentL = osent + UNUSED_TOKENS_STR
L = Long
osent2 = either osent or osentL

ttt = train, tune, test
tuning=dev=development=validation

"""

LINE_SEPARATOR = "[%@!]"
SEED = 777

#"ex" stands for extraction
# "cc" stands for coordinating conjunction
# There are only 7 cc's. They are called the "fanboys",
# because they start with the letters f-a-n-b-o-y-s
FANBOYS = ["for" , "and" , "nor", "but", "or", "yet", "so"]


USE_POS_INFO = True
CC_METRIC_SAVE = True

INPUT_DIR = "input_data"
CACHE_DIR = 'cache'
WEIGHTS_DIR = "weights"
PREDICTING_DIR = "predicting"
LOGS_DIR = "logs"
CC_METRIC_STORAGE_DIR = "cc_metric_storage"
VAL_OUT_DIR = "val_outputs"

CCTAGS_TRAIN_FP = 'input_data/openie-data/ptb-train.labels'
# CCTAGS_TRAIN_FP = 'tests/small_cctags.txt' # small file for warmup run

CCTAGS_TUNE_FP = 'input_data/openie-data/ptb-dev.labels'
CCTAGS_TEST_FP = 'input_data/openie-data/ptb-test.labels'

# for Openie6, this is "models/conj_model/epoch=28_eval_acc=0.854.ckpt"
CC_BEST_WEIGHTS_FP = WEIGHTS_DIR + '/cc.txt' # to be changed 

EXTAGS_TRAIN_FP = INPUT_DIR + '/openie-data/openie4_labels'
# EXTAGS_TRAIN_FP = "tests/small_extags.txt" # small file for warmup run

# IMPORTANT: dev.txt and test.txt are extag files with single ex that only
# contains NONE extags. The actual extags are obtained by ExMetric from
# benchmark files. Don't change the dev.txt or test.txt files to something
# else because ExMetric is hard wired to expect them.
EXTAGS_TUNE_FP = INPUT_DIR + "/carb-data/dev.txt"
EXTAGS_TEST_FP = INPUT_DIR + "/carb-data/test.txt"

# for Openie6, this is "models/warmup_oie_model/epoch=13_eval_acc=0.544.ckpt"
EX_BEST_WEIGHTS_FP = WEIGHTS_DIR + '/ex.txt' # to be changed

PRED_IN_FP = PREDICTING_DIR + "/carb_sentences"

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
MAX_NUM_OSENTL_WORDS = 100

PAD_ICODE = 0  # ipad
SEP_ICODE = 100
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

# in alphabetical order, eliminated repeats begin(2X->1X), 
# do(2X->1X), say(2X->1X), start(2X->1X)
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
