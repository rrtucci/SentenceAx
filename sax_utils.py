from collections import defaultdict
import random
import numpy as np
import torch
import spacy
import nltk
from Params import *
from math import floor
from copy import copy
import pkg_resources as pkg


class DotDict(dict):
    """
    dot instead of [] notation access to dictionary attributes.
    Openie6 uses this but we won't.

    This is sort of the inverse of to_dict(). DotDict() creates a class from
    a dictionary and to_dict() creates a dictionary from a class.

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
    if not default_d:
        default_d = {}
    new_dict = copy(default_d)
    for key in dominant_d:
        new_dict[key] = dominant_d[key]
    return new_dict


def get_tag_to_ilabel(task):
    """

    Parameters
    ----------
    task: str

    Returns
    -------
    dict[str, int]

    """
    if task == "ex":
        tag_to_ilabel = EXTAG_TO_ILABEL
    elif task == "cc":
        tag_to_ilabel = CCTAG_TO_ILABEL
    else:
        assert False
    return tag_to_ilabel


def get_task_logs_dir(task):
    """
    Parameters
    ----------
    task: str

    Returns
    -------
    str

    """
    if task == "ex":
        tdir = LOGS_DIR + '/ex'
    elif task == "cc":
        tdir = LOGS_DIR + '/cc'
    else:
        assert False
    return tdir


def get_num_depths(task):
    """
    Parameters
    ----------
    task: str

    Returns
    -------
    int

    """
    if task == "ex":
        x = EX_NUM_DEPTHS
    elif task == "cc":
        x = CC_NUM_DEPTHS
    else:
        assert False
    return x


def has_puntuation(str0,
                   ignored_chs="",
                   verbose=False):
    """

    Parameters
    ----------
    str0: str
    ignored_chs: str
    verbose: bool

    Returns
    -------
    bool

    """
    for ch in str0:
        if ch in PUNCT_MARKS and ch not in ignored_chs:
            if verbose:
                print(ch)
            return True
    return False


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
    x: str|list[str]|dict[str, Any]

    Returns
    -------
    str|list[str]|dict[str, Any]

    """
    if type(x) == str:
        return x.split("[unused")[0].strip()
    elif type(x) == list:
        return [a.split("[unused")[0].strip() for a in x]
    elif type(x) == dict:
        return {key.split("[unused")[0].strip():value
                for key, value in x.items()}
    else:
        assert False


def redoL(x):
    """

    Parameters
    ----------
    x: str|list[str]|dict[str, Any]

    Returns
    -------
    str|list[str]|dict[str, Any]
    """
    if type(x) == str:
        return x + UNUSED_TOKENS_STR
    elif type(x) == list:
        return [a + UNUSED_TOKENS_STR for a in x]
    elif type(x) == dict:
        return {key + UNUSED_TOKENS_STR: value
                for key, value in x.items()}
    else:
        assert False


# Don't use, even if the inner dimmension of lll_ilabel
# does not agree with the number of words in osent2
# def unL_lll(lll_ilabel):
#     return [[l_ilabel[:-3] for l_ilabel in ll_ilabel]
#             for ll_ilabel in lll_ilabel]

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


def sax_sniffer(name, osent2_to_exs, lll_ilabel):
    """

    Parameters
    ----------
    name: str
    osent2_to_exs: dict[str, list[SaxExtraction]]
    lll_ilabel: list[list[list[int]]]

    Returns
    -------
    None

    """
    print(name + " sniffer")
    for sam, (osent2, exs) in enumerate(osent2_to_exs.items()):
        if "Philip Russell" in osent2:
            print(lll_ilabel[sam])
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


def to_dict(class_obj):
    """

    Parameters
    ----------
    class_obj: Any

    Returns
    -------
    dict[str, Any]

    """
    attributes_dict = {}
    for attr_name in dir(class_obj):
        attr = getattr(class_obj, attr_name)
        if not callable(attr) and not attr_name.startswith('__'):
            attributes_dict[attr_name] = attr
    return attributes_dict


def print_list(list_name, li):
    """
    describe list

    Parameters
    ----------
    list_name: str
    li: list[Any, Any]

    Returns
    -------
    None

    """
    print(list_name + " " + str(len(li)) + "\n" + str(li))


def print_tensor(tensor_name, ten):
    """

    Parameters
    ----------
    tensor_name: str
    ten: torch.tensor

    Returns
    -------
    None

    """
    print(tensor_name + " " + str(ten.shape) + "\n" + str(ten))


def check_module_version(module_name, lub_version):
    """

    Parameters
    ----------
    module_name: str
    lub_version: str

    Returns
    -------
    None

    """
    try:
        module_version = pkg.get_distribution(module_name).version
        if pkg.parse_version(module_version) >= \
                pkg.parse_version(lub_version):
            print(f"{module_name} version is {module_version} "
                  f"so it is >= {lub_version} as required.")
        else:
            print(f"{module_name} version is {module_version}. "
                  f" Version >= {lub_version} is required.")
            assert False
    except pkg.DistributionNotFound:
        print(f"{module_name} is not installed.")


if __name__ == "__main__":
    def main1():
        h = {"x": 5, "y": 3}
        H = DotDict(h)
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


    def main4():
        print(has_puntuation("NONE NONE\n"))


    def main5():
        class Simple:
            def __init__(self):
                self.a = 3
                self.b = [6, 9]

        simp = Simple()
        print(to_dict(simp))


    main1()
    main2()
    main3()
    main4()
    main5()
