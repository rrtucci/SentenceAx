task = "oie"
mode = "splipredtict"

## Running Model

if task == "oie" and mode == "splipredtict":
    hparams = {
        "conj_model": "models/conj_model/epoch=28_eval_acc=0.854.ckpt",
        "gpus": 1,
        "inp": "sentences.txt",
        "mode": "splipredtict",
        "num_extractions": 5,
        "oie_model": "models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt",
        "out": "predictions.txt",
        "rescore_model": "models/rescore_model",
        "rescoring": True,
        "task": "oie"
    }
## Training Model

### Warmup Model
# Training:
if task == "oie" and mode == "train_test":
    hparams = {
        "batch_size": 24,
        "epochs": 30,
        "gpus": 1,
        "iterative_layers": 2,
        "lr": "2e-05",
        "mode": "train_test",
        "model_str": "bert-base-cased",
        "optimizer": "adamW",
        "save": "models/warmup_oie_model",
        "task": "oie"
    }
# Testing:
if task == "oie" and mode == "test":
    hparams = {
        "batch_size": 24,
        "gpus": 1,
        "mode": "test",
        "model_str": "bert-base-cased",
        "save": "models/warmup_oie_model",
        "task": "oie"
    }

# Predicting
if task == "oie" and mode == "predtict":
    hparams = {
        "gpus": 1,
        "inp": "sentences.txt",
        "mode": "predict",
        "model_str": "bert-base-cased",
        "out": "predictions.txt",
        "save": "models/warmup_oie_model",
        "task": "oie"
    }
### Constrained Model
# Training
if task == "oie" and mode == "resume":
    hparams = {
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
        "task": "oie",
        "val_check_interval": "0.1",
        "wreg": 1
    }
# Testing
if task == "oie" and mode == "test":
    hparams = {
        "batch_size": 16,
        "gpus": 1,
        "mode": "test",
        "model_str": "bert-base-cased",
        "save": "models/oie_model",
        "task": "oie"
    }
# Predicting
if task == "oie" and mode == "predtict":
    hparams = {
        "gpus": 1,
        "inp": "sentences.txt",
        "mode": "predict",
        "model_str": "bert-base-cased",
        "out": "predictions.txt",
        "save": "models/oie_model",
        "task": "oie"
    }
### Running Coordination Analysis
if task == "conj" and mode == "train_test":
    hparams = {
        "batch_size": 32,
        "epochs": 40,
        "gpus": 1,
        "iterative_layers": 2,
        "lr": "2e-05",
        "mode": "train_test",
        "model_str": "bert-large-cased",
        "optimizer": "adamW",
        "save": "models/conj_model",
        "task": "conj"
    }

### Final Model

# Running
if task == "oie" and mode == "splipredtict":
    hparams = {
        "conj_model": "models/conj_model/epoch=28_eval_acc=0.854.ckpt",
        "gpus": 1,
        "inp": "carb/data/carb_sentences.txt",
        "mode": "splitpredict",
        "num_extractions": 5,
        "oie_model": "models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt",
        "out": "models/results/final",
        "rescore_model": "models/rescore_model",
        "rescoring": True,
        "task": "oie"
    }
