# hparams=hyperparamters, fp = file path


EXT_SAMPLES_FPATH = "data/ext_samples.txt"
# file paths for training, tuning and testing cctags (cc=conjunction)
CCTAGS_TRAIN_FPATH = "data/cctags_train.txt"
#dev=development=validation=tuning
CCTAGS_TUNE_FPATH=  "data/cctags_tune.txt"
CCTAGS_TEST_FPATH= "data/cctags_test.txt"

# file paths for training, tuning and testing extags (ex=extraction)
EXTAGS_FPATH = "data/extags_all.txt"
EXTAGS_TRAIN_FPATH = "data/extags_train.txt"
#dev=development=validation=tuning
EXTAGS_TUNE_FPATH=  "data/extags_tune.txt"
EXTAGS_TEST_FPATH= "data/extags_test.txt"


BOS_TOKEN_ID = 101 # bos=begin of sentence
EOS_TOKEN_ID = 102 # eos=end of sentence
#MODEL_STR =

MAX_EXTRACTION_LENGTH = 5

# optimization arguments
NUM_EPOCHS=24
BATCH_SIZE=32
SEED=777
LR=2E-5  # lr=learning rate
OTHER_LR=1E-3
OPTIMIZER='adamw'

# data arguments
# model arguments
# bert-large-cased-whole-word-masking, bert-large-cased, bert-base-cased
MODEL_STR='bert-base-cased'
DROPOUT=0.0
OPTIM_ADAM=True
OPTIM_LSTM=True
OPTIM_ADAM_LSTM=True
ITERATIVE_LAYERS=2
LABELLING_DIM=300
NUM_EXTRACTIONS=5
KEEP_ALL_PREDICTIONS=True
OIE_SPLIT=True
NO_LT=True
WRITE_ALLENNLP=True
WRITE_ASYNC=True

# constraints
WREG=0
CWEIGHTS='1'
MULTI_OPT=True


UNUSED_TOKENS = ["[unused1]", "[unused2]", "[unused3]"]

