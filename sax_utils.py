from collections import defaultdict
import random
import numpy as np
import torch
import spacy
import nltk
from Params import *
from math import floor
from copy import copy


class ClassFromDict(dict):
    """
    dot instead of [] notation access to dictionary attributes.
    Nice to know but won't use.

    Attributes
    ----------

    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def merge_dicts(dominant_d, default_d):
    """

    Parameters
    ----------
    dominant_d: dict
    default_d: dict

    Returns
    -------
    dict

    """
    new_dict = copy(default_d)
    for key in dominant_d:
        new_dict[key] = dominant_d[key]
    return new_dict


def get_words(sent, algo="nltk"):
    """
    note: get_words("") = []

    Openie6 and SentenceAx start off from an Allen file (AF). We will
    assume that in an AF, the sentences have punctuation marks like commas
    and periods with blank space before and after. Hence, using `get_words(
    )` with the "ss" algo will be sufficient for most purposes.

    nlkt and spacy both split '[unused1]' into '[', 'unused1', ']' so if
    want POS to split it also, use nlkt or spacy.

    Spacy slow for tokenizing only.


    Parameters
    ----------
    sent: str
    algo: str

    Returns
    -------
    list[str]

    """
    if algo == "ss":
        return sent.strip().split()
    elif algo == "ss+":
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
    elif algo == "spacy":
        # slow for just tokenizing
        nlp = spacy.load("en_core_web_sm")
        if "[unused" in sent:
            doc = nlp(undoL(sent))
            return [tok.text for tok in doc] + UNUSED_TOKENS
        else:
            doc = nlp(sent)
            return [tok.text for tok in doc]

    elif algo == "nltk":
        if "[unused" in sent:
            return nltk.word_tokenize(undoL(sent)) + UNUSED_TOKENS
        else:
            return nltk.word_tokenize(sent)
    else:
        assert False


def none_dd(di):
    """

    Parameters
    ----------
    di: dict

    Returns
    -------
    defaultdict

    """
    # dd = default dictionary
    return defaultdict(lambda: None, di)


def set_seed(seed):
    """

    Parameters
    ----------
    seed: int

    Returns
    -------
    None

    """
    # Be warned that even with all these seeds,
    # complete reproducibility cannot be guaranteed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_ttt_sents(num_sents, ttt_fractions):
    """

    Parameters
    ----------
    num_sents: int
    ttt_fractions: list(float)

    Returns
    -------
    int, int, int

    """
    assert abs(sum(ttt_fractions) - 1) < 1e-8
    num_train_sents = floor(ttt_fractions[0] * num_sents)
    num_tune_sents = floor(ttt_fractions[1] * num_sents)
    num_test_sents = floor(ttt_fractions[2] * num_sents)
    num_extra_sents = \
        num_sents - num_train_sents - num_tune_sents - num_test_sents
    num_train_sents += num_extra_sents
    assert num_sents == num_train_sents + num_tune_sents + num_test_sents
    # print("nnmk", num_train_sents, num_tune_sents, num_test_sents)
    return num_train_sents, num_tune_sents, num_test_sents


def undoL(x):
    """

    Parameters
    ----------
    x: str|list[str]

    Returns
    -------
    str|list[str]

    """
    if type(x) == str:
        return x.split("[unused")[0].strip()
    else:
        return [a.split("[unused")[0].strip() for a in x]


def redoL(x):
    """

    Parameters
    ----------
    x: str|list[str]

    Returns
    -------
    str|list[str]
    """
    if type(x) == str:
        return x + UNUSED_TOKENS_STR
    else:
        return [a + UNUSED_TOKENS_STR for a in x]


# Don't use, even if the inner dimmension of lll_ex_ilabel
# does not agree with the number of words in osent2
# def unL_lll(lll_ex_ilabel):
#     return [[l_ilabel[:-3] for l_ilabel in ll_ilabel]
#             for ll_ilabel in lll_ex_ilabel]

def use_ascii_quotes(line):
    """

    Parameters
    ----------
    line: str

    Returns
    -------
    str

    """
    return line.replace('’', '\'').replace('”', '\'\'')


def replace_in_list(l_x, x, new_x):
    """

    Parameters
    ----------
    l_x:  list[Any]
    x: Any
    new_x: Any

    Returns
    -------
    None

    """
    assert l_x.count(x) == 1
    k = l_x.index(x)
    l_x[k] = new_x


def sax_sniffer(name, osent2_to_exs, lll_ex_ilabel):
    """

    Parameters
    ----------
    name: str
    osent2_to_exs: dict[str, list[SaxExtraction]]
    lll_ex_ilabel: list[list[list[int]]]

    Returns
    -------
    None

    """
    print(name + " sniffer")
    for sam, (osent2, exs) in enumerate(osent2_to_exs.items()):
        if "Philip Russell" in osent2:
            print(lll_ex_ilabel[sam])
            for ex in exs:
                print(ex.arg1, ex.rel, ex.arg2)


def carb_sniffer(name, osent2_to_exs):
    """

    Parameters
    ----------
    name: str
    osent2_to_exs: dict[str, list[Extraction]]

    Returns
    -------
    None

    """
    print(name + " sniffer")
    for sam, (osent2, exs) in enumerate(osent2_to_exs.items()):
        if "Philip Russell" in osent2:
            for ex in exs:
                print(ex.pred, ex.args)


def Ten(lista):
    """

    Parameters
    ----------
    lista: list[Any]

    Returns
    -------
    torch.Tensor

    """
    assert type(lista) == list
    return torch.Tensor(lista)


def Li(tensor):
    """

    Parameters
    ----------
    tensor: torch.Tensor

    Returns
    -------
    list[Any]

    """
    assert type(tensor) == torch.Tensor
    return tensor.tolist()


def add_key_to_target_d(key, fix_d, target_d):
    """

    Parameters
    ----------
    key: Any
    fix_d: dict[Any, Any]
    target_d: dict[Any, Any]

    Returns
    -------
    dict[Any, Any]

    """
    if fix_d:
        if fix_d[key] not in target_d:
            target_d[fix_d[key]] = []
    else:
        if key not in target_d:
            target_d[key] = []

def add_key_value_pair_to_target_d(key, value, fix_d, target_d):
    """

    Parameters
    ----------
    key: Any
    value: Any
    fix_d: dict[Any, Any]
    target_d: dict[Any, Any]

    Returns
    -------
    dict[Any, Any]

    """
    if fix_d:
        if value not in target_d[fix_d[key]]:
            target_d[fix_d[key]].append(value)
    else:
        if value not in target_d[key]:
            target_d[key].append(value)


if __name__ == "__main__":
    def main1():
        h = {"x": 5, "y": 3}
        H = ClassFromDict(h)
        print(H.x)  # Output: 5
        print(H.y)  # Output: 3
        H.y = 5
        print(H.y, h["y"])  # output 5,3

        def F(x, y):
            return x + y

        print(F(**h))  # Output: 8


    def main2():
        l_x = [1, 2, 3, 4]
        replace_in_list(l_x, 3, 33)
        print("l_x=", l_x)


    def main3():
        sent1 = 'This is a great quote: "To be, or not to be".'
        sent2 = 'This is a great quote : " To be, or not to be [unused1] " . '
        print(get_words(sent1))
        print(get_words(sent2))


    main1()
    main2()
    main3()

