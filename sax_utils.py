"""

The purpose of this file is to gather together those global (i.e., static)
methods used by SentenceAx that seem too general to belong to any one of
its classes.

"""

from collections import defaultdict
import random
import numpy as np
import torch
import nltk
from Params import *
from math import floor
from copy import copy
import pkg_resources as pkg
from unidecode import unidecode
import os


class DotDict(dict):
    """
    This class provides dot (.) instead of square bracket ([]) access to
    dictionary attributes. Openie6 uses this but SentenceAx doesn't.

    This is sort of the inverse of to_dict() defined elsewhere in this file.
    DotDict() creates a class instance from a dictionary and to_dict()
    creates a dictionary from a class instance.

    Attributes
    ----------

    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def merge_dicts(dominant_d, default_d):
    """
    This method returns a new dictionary which is the result of merging a
    dominant dictionary `dominant_d` with a default dictionary `default_d`.

    Parameters
    ----------
    dominant_d: dict[str, Any]
    default_d: dict[str, Any]

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
    This method returns the tag_to_ilabel dictionary for the task `task`,
    where task in ["ex", "cc"].


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
    This method returns the logs_directory for the task `task`, where task
    in ["ex", "cc"].



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
    This method returns the number_of_depths for the task `task`, where task
    in ["ex", "cc"].

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
    This method returns True iff the string `str0` contains characters that
    are punctuation marks (for example, an underscore "_").

    Parameters
    ----------
    str0: str
    ignored_chs: str
        ignored characters, all presented as one string. e.g., "!?,"
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
    This method splits a sentence into words (some punctuation marks like
    periods and commas are considered words.)

    nlkt and spacy both split '[unused1]' into '[', 'unused1', ']' so first
    remove UNUSED_TOKENS_STR, split into words, and finally add
    UNUSED_TOKENS to result.

    note: get_words("") = []

    Spacy is slow compared to nlkt if used only for tokenizing into words.
    Hence, SentenceAx will use only nlkt for this.


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
    # elif algo == "spacy":
    #     # slow for just tokenizing
    #     nlp = spacy.load("en_core_web_sm")
    #     if "[unused" in sent:
    #         doc = nlp(undoL(sent))
    #         return [tok.text for tok in doc] + UNUSED_TOKENS
    #     else:
    #         doc = nlp(sent)
    #         return [tok.text for tok in doc]

    elif algo == "nltk":
        if "[unused" in sent:
            return nltk.word_tokenize(undoL(sent)) + UNUSED_TOKENS
        else:
            return nltk.word_tokenize(sent)
    else:
        assert False


def set_seed(seed):
    """
    similar to Openie6.model.set_seed()
    
    This method sets a panoply of seeds to `seed`.

    Be forewarned that even with all these seeds set, complete
    reproducibility cannot be guaranteed.

    Parameters
    ----------
    seed: int

    Returns
    -------
    None

    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_num_ttt_sents(num_sents, ttt_fractions):
    """
    Given an int `num_sents`, and a list of fractions `ttt_fractions` equal
    to [f0, f1, f2] such that f0+f1+f2=1, this method returns a triple of
    integers (x, y, z) such that x+y+z = num_sents, and x \approx
    num_sents*f_0, y \approx num_sents*f1, z \approx num_sents*f2.
    

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
    This method works if x is a str, list[str] or  dict[str, Any]. If x is a
    str, it returns the same string, with the tail UNUSED_TOKENS_STR
    removed, if it has this tail to begin with. If x is a list[str] (or a
    dict[str, Any]), it applies undoL() to each list item (or dictionary key).
    
    undoL() and redoL() are inverse operations.

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
        return {key.split("[unused")[0].strip(): value
                for key, value in x.items()}
    else:
        assert False


def redoL(x):
    """
    This method works if x is a str, list[str] or  dict[str, Any]. If x is a
    str, it returns the same string, with a tail UNUSED_TOKENS_STR added,
    if it did not have this tail to begin with. If x is a list[str] (or a
    dict[ str, Any]), it applies redoL() to each list item (or dictionary key).
    
    undoL() and redoL() are inverse operations.

    Parameters
    ----------
    x: str|list[str]|dict[str, Any]

    Returns
    -------
    str|list[str]|dict[str, Any]
    """
    if type(x) == str:
        return undoL(x) + UNUSED_TOKENS_STR
    elif type(x) == list:
        return [undoL(a) + UNUSED_TOKENS_STR for a in x]
    elif type(x) == dict:
        return {undoL(key) + UNUSED_TOKENS_STR: value
                for key, value in x.items()}
    else:
        assert False


def get_ascii(x):
    """
    This method takes as input a string or list of strings x, with possibly
    non-ascii characters ( utf-8). It returns a new string or list of
    strings which is the same as the old one, except that non-ascii
    characters have been converted to their nearest ascii counterparts. For
    example, curly quotes will be converted to straight ones.

    Parameters
    ----------
    x: str | list[str]

    Returns
    -------
    str | list[str]

    """
    if type(x) == str:
        return unidecode(x)
    else:
        return [unidecode(str0) for str0 in x]


def replace_in_list(l_x, x, new_x):
    """
    This method checks that `x` occurs only once in list `l_x`. It returns a
    new list wherein `x` has been replaced by `new_x`.

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
    This method was used for debugging.

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
    This method was used for debugging.

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
    This method takes as input a list (or list[list[ or list[list[list[)
    `lista`). It converts `lista` to a torch.Tensor, which it then returns.

    Ten() and Li() are sort of inverses of each other, except that in
    general a list[list[ cannot be converted to a torch.Tensor, unless it is
    padded first.

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
    This method takes as input a torch.Tensor `tensor`. It converts `tensor`
    to a list (or list[list[ or list[list[list[) which it then returns.

    Ten() and Li() are sort of inverses of each other, except that in
    general a list[list[ cannot be converted to a torch.Tensor, unless it is
    padded first.


    Parameters
    ----------
    tensor: torch.Tensor

    Returns
    -------
    list[Any]

    """
    assert type(tensor) == torch.Tensor
    return tensor.tolist()


def add_key_to_this_d(key, grow_d, this_d):
    """
    This method returns a dictionary after adding to it a key.

    This method is used in Model.

    Parameters
    ----------
    key: Any
    grow_d: dict[Any, Any]
    this_d: dict[Any, Any]

    Returns
    -------
    dict[Any, Any]

    """
    if grow_d:
        if grow_d[key] not in this_d:
            this_d[grow_d[key]] = []
    else:
        if key not in this_d:
            this_d[key] = []


def add_key_value_pair_to_this_d(key, value, grow_d, this_d):
    """
    This method returns a dictionary after adding to it a key-value pair.

    This method is used in Model.

    Parameters
    ----------
    key: Any
    value: Any
    grow_d: dict[Any, Any]
    this_d: dict[Any, Any]

    Returns
    -------
    dict[Any, Any]

    """
    if grow_d:
        if value not in this_d[grow_d[key]]:
            this_d[grow_d[key]].append(value)
    else:
        if value not in this_d[key]:
            this_d[key].append(value)


def to_dict(class_obj):
    """
    This method takes as input an object (instance) of a class, and it
    returns a dictionary with (key, value) = (class attribute name, value of
    that attribute).

    This is sort of the inverse of DotDict() defined elsewhere in this file.
    DotDict() creates a class instance from a dictionary and to_dict()
    creates a dictionary from a class instance.

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
    This method prints the name of a list, its length and, in the next line,
    its values.

    Parameters
    ----------
    list_name: str
    li: list[Any, Any]

    Returns
    -------
    None

    """
    print(list_name + ", len=" + str(len(li)) + "\n" + str(li))


def print_tensor(tensor_name, ten):
    """
    This method prints the name of a tensor, its shape and, in the next
    line, its values. Only edge values are printed if the tensor is too big.

    Parameters
    ----------
    tensor_name: str
    ten: torch.tensor

    Returns
    -------
    None

    """
    print(tensor_name + ", shape=" + str(ten.shape) + "\n" + str(ten))


def check_module_version(module_name, lub_version):
    """
    This method checks that the version of the module named `module_name` is
    greater or equal to `lub_version`. (lub=least upper bound)

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


def is_valid_label_list(labels, task, label_type):
    """
    This method checks that a cctags or extags or ilabels list named
    `labels` satisfies certain minimal requirements (for example, that it
    have an ARG1 and a REL for extags).

    task in ["ex", "cc"]

    label_type in ["ilabels", "tags"]

    Parameters
    ----------
    labels: list[str]|list[int]
    task: str
    label_type: str

    Returns
    -------
    bool

    """
    assert task in ["ex", "cc"]
    assert label_type in ["ilabels", "tags"]
    valid = False
    if task == "ex":
        if label_type == "ilabels":
            # 'ARG1': 1, 'REL': 2
            valid = (1 in labels and 2 in labels)
        elif label_type == "tags":
            valid = ("ARG1" in labels and "REL" in labels)
    elif task == "cc":
        if label_type == "ilabels":
            # 'CC': 3
            valid = (3 in labels)
        elif label_type == "tags":
            valid = ("CC" in labels)

    return valid


def get_omit_exless_flag(task, ttt):
    """
    This method returns True iff we want to omit exless samples (i.e.,
    samples with no extractions).

    For task="ex", the dev.txt and test.txt are extag files with single ex
    that only contains NONE extags, so do not omit exless samples for those,
    or will omit entire file. The input files for predicting have no exs at
    all, so don't omit exless samples for those either. In all other cases,
    do omit the exless samples.

    Parameters
    ----------
    task: str
    ttt: str

    Returns
    -------

    """
    assert task in ["ex", "cc"]
    assert ttt in ["train", "tune", "test"]
    if task == "ex" and ttt in ["tune", "test"]:
        return False
    return True


def delete_all_files_with_given_ending(dir_fp, ending):
    """
    This method deletes all files in the directory with path `dir_fp` whose
    names end in the string `ending`.


    Parameters
    ----------
    dir_fp: str
    ending: str

    Returns
    -------
    None

    """
    try:
        fnames = os.listdir(dir_fp)
        for fname in fnames:
            fpath = os.path.join(dir_fp, fname)
            if fname.endswith(ending):
                os.remove(fpath)
    except Exception as e:
        print(f"An error occurred: {e}")


def round_dict_values(di, precision=4):
    """
    This method rounds the values of a dict[str, float] to precision
    `precision`.

    Parameters
    ----------
    di: dict[str, Any]
    precision: int

    Returns
    -------
    dict[str, float]

    """
    return {key: round(float(di[key]), precision) for key in di.keys()}


def get_train_tags_fp(task, small=False):
    """
    This method returns the file path to the training dataset.

    Parameters
    ----------
    task: str
    small: bool
        True iff desire a small training dataset for warmup and debugging
        purposes

    Returns
    -------
    str

    """
    if task == "ex":
        if small:
            fp = SMALL_TRAIN_EXTAGS_FP
        else:
            fp = TRAIN_EXTAGS_FP
    elif task == "cc":
        if small:
            fp = SMALL_TRAIN_CCTAGS_FP
        else:
            fp = TRAIN_CCTAGS_FP
    else:
        assert False
    return fp


def write_l_sample_str(l_sample_str,
                       out_fp,
                       appended,
                       numbered):
    """
    This method writes a file at `out_fp`. The file is an enumerated list of
    the strings in the list `l_sample_str`. The file precedes each sample
    string by a line consisting of SAMPLE_SEPARATOR and the enumeration number
    (or no number if numbered=False)

    Parameters
    ----------
    l_sample_str: list[str]
    out_fp: str
    appended: bool
        open() mode is either "w" or "a"
    numbered: bool

    Returns
    -------
    None

    """
    fmode = "a" if appended else "w"
    with open(out_fp, fmode) as f:
        num_sam = len(l_sample_str)
        for k in range(num_sam):
            num_str = ""
            if numbered:
                num_str = str(k + 1)
            f.write(SAMPLE_SEPARATOR + num_str + "\n" +
                    l_sample_str[k].strip() + "\n")


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


    def main6():
        curly_quotes = "‘magic’"
        str0 = "``abc" + curly_quotes
        print(get_ascii(str0))
        print(get_words(str0))
        print(get_words(get_ascii(str0)))


    # main1()
    # main2()
    # main3()
    # main4()
    # main5()
    main6()
