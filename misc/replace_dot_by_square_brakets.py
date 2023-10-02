"""

For a given set S of files, this script replaces inside every file in S,
terms like "params_d.word" by "params_d["word"]".

"""
import os
from copy import copy

file_paths = ["SaxDataLoader.py",
              "ModelDataLoader_old.py",
              "MConductor.py"]

word_d =\
{
    "accumulate_grad_batches": (2, ),
    "batch_size": (32, 16, 24, ),
    "checkpoint": ("models/warmup_oie_model/epoch=13_eval_acc=0.544.ckpt", ),
    "conj_model": ("models/conj_model/epoch=28_eval_acc=0.854.ckpt", ),
    "constraints": ("posm_hvc_hvr_hve", ),
    "con_weights": ("3_3_3_3", ),
    "epochs": (30, 16, 40, ),
    "gpus": (1, ),
    "gradient_clip_val": (1, ),
    "inp": ("carb/data/carb_sentences.txt", "sentences.txt", ),
    "iterative_layers": (2, ),
    "lr": ("2e-5", "2e-05", "5e-06", ),
    "mode": ("train_test", "resume", "splitpredict", "test", "predict", ),
    "model_str": ("bert-large-cased", "bert-base-cased", ),
    "multi_opt": (True, ),
    "num_extractions": (5, ),
    "oie_model": ("models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt", ),
    "optimizer": ("adam", "adamW", ),
    "out": ("predictions.txt", "models/results/final", ),
    "rescore_model": ("models/rescore_model", ),
    "rescoring": (True, ),
    "save": ("models/oie_model", "models/conj_model", "models/warmup_oie_model", ),
    "save_k": (3, ),
    "task": ("conj", "oie", ),
    "val_check_interval": ("0.1", ),
    "wreg": (1, ),
}

additional_dict = {
    "build_cache": None,
    "dev_fp": None,
    "predict_fp": None,
    "split_fp": None,
    "test_fp": None,
    "train_fp": None,
    "type": None,
    "write_allennlp": None
}

def replace_in_file(file_path):
    with open(file_path, mode='r', encoding="utf-8") as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        new_line = copy(line)
        stripped = line.strip()
        if len(stripped) > 0 and stripped[0] != "#":
            # print("cdfgb", line)
            # important:
            # must make changes in reverse alphabetical order
            # so replace "model" before "mode"
            words = sorted(list(word_d.keys()) +
                           list(additional_dict.keys()),
                           reverse=True)
            for word in words:
                new_line = new_line.replace('params_d.' + word,
                                    'params_d["' + word + '"]')

        new_lines.append(new_line)

    with open(file_path, mode='w', encoding="utf-8") as file:
        file.writelines(new_lines)


for file_path in file_paths:
    if os.path.exists(file_path):
        print("bnmgccccccccccccccccc", file_path)
        replace_in_file(file_path)
        print(f"Modified: {file_path}")
    else:
        print(f"File not found: {file_path}")
