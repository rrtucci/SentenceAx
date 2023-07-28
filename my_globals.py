from dict_utils import *
# hparams=hyperparamters, fp = file path

hparams_long = {
    "accumulate_grad_batches": None,
    "batch_size": 32,
    "bos_token_id": 101,  # bos":begin of sentence
    "build_cache": None,
    "checkpoint": "data/save",
    "conj_model": None,
    "constraints": None,
    "cweights": None,
    "debug": None,
    "dev_fp": None,
    "dropout": 0.5,
    "eos_token_id": 102,  # eos":end of sentence
    "epochs": 24,
    "gpus": None,
    "gradient_clip_val": 5,
    "inp": None,
    "iterative_layers": 2,
    "labelling_dim": 300,
    "lr": 2E-5,  # lr":learning rate
    "max_steps": None,
    "mode": None, # "train", "test", "predict", "splitpredict"
    "model_str": None,
    "multi_opt": None,
    "no_lt": None,
    "num_extractions": 5,
    "num_sanity_val_steps": None,
    "num_tpu_cores": None,
    "oie_model": None,
    "optimizer": 'adamw',
    "out": None,
    "predict_fp": None,
    "rescore_model": None,
    "rescoring": None,
    "save": "data/save",
    "save_k": None,
    "split_fp": None,
    "task": None, # "cc", "ex"
    "test_fp": None,
    "track_grad_norm": None,
    "train_fp": None,
    "train_percent_check": None,
    "type": None,
    "use_tpu": None,
    "val_check_interval": None,
    "wreg": 0,
    "write_allennlp": None,
    "write_async": None
}

hparams= {key: value for key, value in hparams_long.items()
           if value is not None}
Hparams = ClassFromDict(hparams)

META_DATA_VOCAB = None
MAX_EXTRACTION_LENGTH = 5
UNUSED_TOKENS = ["[unused1]", "[unused2]", "[unused3]"]

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
