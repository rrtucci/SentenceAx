from dict_utils import *

# HPARAMS = hyperparamters, fp = file path

EXT_SAMPLES_FPATH = "data/ext_samples.txt"
# file paths for training, tuning and testing cctags (cc=conjunction)
CCTAGS_TRAIN_FPATH = "data/cctags_train.txt"
# dev=development=validation=tuning
CCTAGS_TUNE_FPATH = "data/cctags_tune.txt"
CCTAGS_TEST_FPATH = "data/cctags_test.txt"

# file paths for training, tuning and testing extags (ex=extraction)
EXTAGS_FPATH = "data/extags_all.txt"
EXTAGS_TRAIN_FPATH = "data/extags_train.txt"
# dev=development=validation=tuning
EXTAGS_TUNE_FPATH = "data/extags_tune.txt"
EXTAGS_TEST_FPATH = "data/extags_test.txt"
BOS_TOKEN_ID = 101 # bos = begin of sentence
EOS_TOKEN_ID = 102 # eos = end of sentence

META_DATA_VOCAB = None
CACHE_DIR = 'data/pretrained_cache'
NUM_LABELS = 6
MAX_EXTRACTION_LENGTH = 5
UNUSED_TOKENS = ["[unused1]", "[unused2]", "[unused3]"]
UNUSED_TOKENS_STR = " " + " ".join(UNUSED_TOKENS)

# I use "cc" instead of "oie" for task
# I use "cc" instead of "conj" for task

# choose current TASK and MODE here
# TASK in "ex", "cc", "custom1", custom2", etc
# MODE in ("predict", "train_test", "splitpredict", "resume", "test", )
TASK = "ex"
MODE = "splipredict"

## Running Model

if TASK == "ex" and MODE == "splipredict":
    HPARAMS = {
        "conj_model": "models/conj_model/epoch=28_eval_acc=0.854.ckpt",
        "gpus": 1,
        "inp": "sentences.txt",
        "mode": "splipredict",
        "num_extractions": 5,
        "oie_model": "models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt",
        "out": "predictions.txt",
        "rescore_model": "models/rescore_model",
        "rescoring": True,
        "task": "ex"
    }
## Training Model

### Warmup Model
# Training:
elif TASK == "ex" and MODE == "train_test":
    HPARAMS = {
        "batch_size": 24,
        "epochs": 30,
        "gpus": 1,
        "iterative_layers": 2,
        "lr": "2e-05",
        "mode": "train_test",
        "model_str": "bert-base-cased",
        "optimizer": "adamW",
        "save": "models/warmup_oie_model",
        "task": "ex"
    }
# Testing:
elif TASK == "ex" and MODE == "test":
    HPARAMS = {
        "batch_size": 24,
        "gpus": 1,
        "mode": "test",
        "model_str": "bert-base-cased",
        "save": "models/warmup_oie_model",
        "task": "ex"
    }

# Predicting
elif TASK == "ex" and MODE == "predict":
    HPARAMS = {
        "gpus": 1,
        "inp": "sentences.txt",
        "mode": "predict",
        "model_str": "bert-base-cased",
        "out": "predictions.txt",
        "save": "models/warmup_oie_model",
        "task": "ex"
    }
### Constrained Model
# Training
elif TASK == "ex" and MODE == "resume":
    HPARAMS = {
        "accumulate_grad_batches": 2,
        "batch_size": 16,
        "checkpoint": "models/warmup_oie_model/epoch=13_eval_acc=0.544.ckpt",
        "constraints": "posm_hvc_hvr_hve",
        "cweights": "3_3_3_3",
        "epochs": 16,
        "gpus": 1,
        "gradient_clip_val": 1,
        "iterative_layers": 2,
        "lr": "2e-5",
        "lr": "5e-06",
        "mode": "resume",
        "model_str": "bert-base-cased",
        "multi_opt": True,
        "optimizer": "adam",
        "save": "models/oie_model",
        "save_k": 3,
        "task": "ex",
        "val_check_interval": "0.1",
        "wreg": 1
    }
# Testing
elif TASK == "ex" and MODE == "test":
    HPARAMS = {
        "batch_size": 16,
        "gpus": 1,
        "mode": "test",
        "model_str": "bert-base-cased",
        "save": "models/oie_model",
        "task": "ex"
    }
# Predicting
elif TASK == "ex" and MODE == "predict":
    HPARAMS = {
        "gpus": 1,
        "inp": "sentences.txt",
        "mode": "predict",
        "model_str": "bert-base-cased",
        "out": "predictions.txt",
        "save": "models/oie_model",
        "task": "ex"
    }
### Running Coordination Analysis
elif TASK == "cc" and MODE == "train_test":
    HPARAMS = {
        "batch_size": 32,
        "epochs": 40,
        "gpus": 1,
        "iterative_layers": 2,
        "lr": "2e-05",
        "mode": "train_test",
        "model_str": "bert-large-cased",
        "optimizer": "adamW",
        "save": "models/conj_model",
        "task": "cc"
    }

### Final Model

# Running
elif TASK == "ex" and MODE == "splipredict":
    HPARAMS = {
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
    }
elif TASK == "custom1":
    # change dictionary values to custom ones
    # define TASK = "custom1", "custom2", etc.
    HPARAMS_LONG = {
        "accumulate_grad_batches": None,
        "batch_size": None,
        "bos_token_id": BOS_TOKEN_ID,
        "build_cache": None,
        "checkpoint": None,
        "conj_model": None,
        "constraints": None,
        "cweights": None,
        "debug": None,
        "dev_fp": None,
        "dropout": None,
        "eos_token_id": EOS_TOKEN_ID,
        "epochs": None,
        "gpus": None,
        "gradient_clip_val": None,
        "inp": None,
        "iterative_layers": 2,
        "labelling_dim": None,
        "lr": None,  # lr = learning rate
        "max_steps": None,
        "mode": None,
        "model_str": None,
        "multi_opt": None,
        "no_lt": None, # no local time
        "num_extractions": None,
        "num_sanity_val_steps": None,
        "num_tpu_cores": None,
        "oie_model": None,
        "optimizer": None,
        "out": None,
        "predict_fp": None,
        "rescore_model": None,
        "rescoring": None,
        "save": None,
        "save_k": None,
        "split_fp": None,
        "task": None,
        "test_fp": None,
        "track_grad_norm": None,
        "train_fp": None,
        "train_percent_check": None,
        "type": None,
        "use_tpu": None,
        "val_check_interval": None,
        "wreg": None,
        "write_allennlp": None,
        "write_async": None
    }

    # eliminate params that are None, in case they are reset to
    # default values by pytorch lightning.
    HPARAMS = {key: value for key, value in HPARAMS_LONG.items()
               if value is not None}
