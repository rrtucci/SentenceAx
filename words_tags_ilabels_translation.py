"""
EXTAG_TO_ILABEL = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                   'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}
BASE_EXTAGS = EXTAG_TO_ILABEL.keys()
ILABEL_TO_EXTAG={0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
                 4: 'ARG2', 5: 'NONE'}

CCTAG_TO_ILABEL = {'NONE': 0, 'CP': 1, 'CP_START': 2,
                   'CC': 3, 'SEP': 4, 'OTHERS': 5}
BASE_CCTAGS = CCTAG_TO_ILABEL.keys()
ILABEL_TO_CCTAG = {0: 'NONE', 1: 'CP', 2: 'CP_START',
                   3: 'CC', 4:'SEP', 5: 'OTHERS'}

CHAIN OF CONVERSIONS
words->extags, cctags->ilabels->extags->words
"""
from sax_globals import *
from sax_utils import *
from SAXExtraction import *


def trans_words_to_extags(ex, set_extags):
    if set_extags:
        ex.set_extags()
    return ex.extags


def trans_words_to_cctags():



def trans_extags_to_ilabels(extags):
    ilabels = []
    for extag in extags:
        ilabels.append(EXTAG_TO_ILABEL[extag])
    return ilabels


def trans_cctags_to_ilabels(cctags):
    ilabels = []
    for cctag in cctags:
        ilabels.append(CCTAG_TO_ILABEL[cctag])
    return ilabels


def trans_ilabels_to_cctags(ilabels, all_words):
    cctags = []
    for ilabel in ilabels:
        cctags.append(ILABEL_TO_CCTAG(ilabel))
    return cctags


def trans_ilabels_to_extags(ilabels):
    extags = []
    for ilabel in ilabels:
        extags.append(ILABEL_TO_EXTAG(ilabel))

    return extags


def trans_cctags_to_words(cctags, orig_sentL):
    all_words = get_words(orig_sentL)
    max_len = len(all_words)
    cc_words = []
    for k, cctag in enumerate(cctags):
        # cctags may be padded
        if k < max_len and cctag not in ["CP_START", "NONE"]:
            cc_words.append(all_words[k])
    return cc_words


def trans_extags_to_words(extags, orig_sentL):
    """
    inferred from Openie6 data_processing.label_is_of_relations()

    Parameters
    ----------
    extags
    orig_sentL

    Returns
    -------

    """
    all_words = get_words(orig_sentL)
    max_len = len(all_words)
    l_arg1 = []
    l_rel = []
    l_arg2 = []
    for k, extag in enumerate(extags):
        # extags may be padded
        if k < max_len:
            if extag == "ARG1":
                l_arg1.append(all_words[k])
            elif extag ==  "REL":
                l_rel.append(all_words[k])
            elif extag == "ARG2":
                l_arg2.append(all_words[k])
            if l_rel[-1] == "[unused1]":
                l_rel = ["[is]"] + l_rel[:-1]
            elif l_rel[-1] == "[unused2]":
                l_rel = ["[is]"] + l_rel[:-1] + ["[of]"]
            elif l_rel[-1] == "[unused3]":
                l_rel = ["[is]"] + l_rel[:-1] + ["[from]"]

        return l_arg1 + l_rel + l_arg2
