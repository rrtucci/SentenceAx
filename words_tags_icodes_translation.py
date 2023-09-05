"""
EXTAG_TO_ICODE = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                   'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}
BASE_EXTAGS = EXTAG_TO_ICODE.keys()
ICODE_TO_EXTAG={0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
                 4: 'ARG2', 5: 'NONE'}

CCTAG_TO_ICODE = {'NONE': 0, 'CP': 1, 'CP_START': 2,
                   'CC': 3, 'SEP': 4, 'OTHERS': 5}
BASE_CCTAGS = CCTAG_TO_ICODE.keys()
ICODE_TO_CCTAG = {0: 'NONE', 1: 'CP', 2: 'CP_START',
                   3: 'CC', 4:'SEP', 5: 'OTHERS'}

* extags (openie-data/openie4_labels) *.labels has no [unused1], *_labels does
have [unused]
Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE


* cctags (openie-data/ptb-train.labels)
Bell , based in Los Angeles , makes and distributes electronic , computer and building products .
NONE NONE NONE NONE NONE NONE NONE CP_START CC CP CP_START SEP CP CC CP NONE NONE
NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE
NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE NONE


CHAIN OF CONVERSIONS
words->extags, cctags->icodes->extags->words
"""
from sax_globals import *
from sax_utils import *
from SaxExtraction import *
from CCTree import *

"""

translate_words_to_extags and translate_words_to_cctags are the hardest. All 
the others are trivial.


"""


def translate_words_to_extags(ex, set_extags):
    if set_extags:
        ex.set_extags()
    return ex.extags


def translate_words_to_cctags(orig_sent, ll_icode):
    """
    CCTree icodes not same as those provided by AutoEncoder.

    This method is surely wrong, but a good first stab.

    Openie6 not very clear about this.


    Parameters
    ----------
    orig_sent
    ll_icode

    Returns
    -------

    """
    words = get_words(orig_sent)
    cctree = CCTree(orig_sent, ll_icode)
    l_spanned_locs = cctree.l_spanned_locs
    max_depth = len(l_spanned_locs)
    nodes = cctree.ccnodes
    depth_to_cclocs = {}
    depth_to_seplocs = {}
    for depth in range(max_depth):
        seplocs = []
        cclocs = []
        for node in nodes:
            if node.depth == depth:
                seplocs += node.seplocs
                cclocs.append(node.ccloc)
        depth_to_seplocs[depth] = seplocs
        depth_to_cclocs[depth] = cclocs
    l_cctags = [["NONE"]*len(words)]
    for depth, locs in enumerate(l_spanned_locs):
        cctags = l_cctags[depth]
        for k, loc in enumerate(sorted(locs)):
            if k==0:
                cctags[loc] = "CP_START"
            else:
                cctags[loc] = "CP"
            if loc in depth_to_cclocs[depth]:
                cctags[loc] = "CC"
            if loc in depth_to_seplocs[depth]:
                cctags[loc] = "SEP"
    return l_cctags





def translate_extags_to_icodes(extags):
    icodes = []
    for extag in extags:
        icodes.append(EXTAG_TO_ICODE[extag])
    return icodes


def translate_cctags_to_icodes(cctags):
    icodes = []
    for cctag in cctags:
        icodes.append(CCTAG_TO_ICODE[cctag])
    return icodes


def translate_icodes_to_cctags(icodes):
    """
    Openie6 seems to use CCTree to go from l_icodes to l_cctags (see
    metric.get_coords()). However, I believe the l_icodes used by CCTree,
    and the ones in this function are different.

    Parameters
    ----------
    icodes

    Returns
    -------

    """

    cctags = []
    for icode in icodes:
        cctags.append(ICODE_TO_CCTAG(icode))
    return cctags


def translate_icodes_to_extags(icodes):
    extags = []
    for icode in icodes:
        extags.append(ICODE_TO_EXTAG(icode))

    return extags


def translate_cctags_to_words(cctags, orig_sentL):
    all_words = get_words(orig_sentL)
    max_len = len(all_words)
    cc_words = []
    for k, cctag in enumerate(cctags):
        # cctags may be padded
        if k < max_len and cctag not in "NONE":
            cc_words.append(all_words[k])
    return cc_words


def translate_extags_to_words(extags, orig_sentL):
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
            elif extag in ["ARG2", "LOC", "TIME"]:
                l_arg2.append(all_words[k])
            if l_rel[-1] == "[unused1]":
                l_rel = ["[is]"] + l_rel[:-1]
            elif l_rel[-1] == "[unused2]":
                l_rel = ["[is]"] + l_rel[:-1] + ["[of]"]
            elif l_rel[-1] == "[unused3]":
                l_rel = ["[is]"] + l_rel[:-1] + ["[from]"]

        return l_arg1 + l_rel + l_arg2
