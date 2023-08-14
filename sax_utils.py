from collections import defaultdict
import random
import numpy as np
import torch
import nltk
from sax_globals import *
from math import floor

class ClassFromDict(dict):
    """
    dot instead of [] notation access to dictionary attributes.
    Nice to know but won't use.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def update_dict(dict, new_dict, add_new_keys=True):
    for key in dict:
        if key in new_dict: # overlapping keys
            dict[key] = new_dict[key]
    if add_new_keys:
        for key in new_dict:
            if key not in dict: # new keys not in dict yet
                dict[key] = new_dict[key]


def get_words(sent, algo=1):
    """
    get_words("") = []

    Parameters
    ----------
    sent

    Returns
    -------

    """
    if algo==1:
        return nltk.word_tokenize(sent)
    elif algo==2:
        if sent:
            li = sent.strip().split()
            li0 = []
            for word in li:
                if word[-1] in PUNCT_MARKS and \
                        word[-1] not in QUOTES:
                    li0.append(word[:-1])
                    li0.append(word[-1])
                else:
                    li0.append(word)
            return li0
        else:
            return []
    else:
        assert False

def none_dd(di):
    # dd = default dictionary
    return defaultdict(lambda: None, di)

def set_seed(seed):
    # Be warned that even with all these seeds,
    # complete reproducibility cannot be guaranteed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


def get_num_ttt_sents(num_sents, ttt_fractions):
    assert abs(sum(ttt_fractions) - 1) < 1e-8
    num_train_sents = floor(ttt_fractions[0] * num_sents)
    num_tune_sents = floor(ttt_fractions[1] * num_sents)
    num_test_sents = floor(ttt_fractions[2] * num_sents)
    num_extra_sents = num_sents - num_train_sents - \
                      num_tune_sents - num_test_sents
    num_train_sents += num_extra_sents
    return num_train_sents, num_tune_sents, num_test_sents


if __name__ == "__main__":
    def main():
        h = {"x": 5, "y": 3}
        H = ClassFromDict(h)
        print(H.x)  # Output: 5
        print(H.y)  # Output: 3
        H.y = 5
        print(H.y, h["y"]) # output 5,3

        def F(x, y):
            return x + y

        print(F(**h))  # Output: 8

    main()