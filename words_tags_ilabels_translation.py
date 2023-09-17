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
words->extags, cctags->ilabels->extags->words
"""
from sax_globals import *
from sax_utils import *
from SaxExtraction import *
from CCTree import *

"""

translate_words_to_extags and translate_words_to_cctags are the hardest. All 
the others are trivial.


"""


def translate_words_to_extags(ex):
    if not ex.extags_are_set:
        ex.set_extags()
    return ex.extags


def translate_words_to_cctags(ll_ilabel, orig_sentL):
    """
    CCTree ilabels not same as those provided by AutoEncoder.

    This method is surely wrong, but a good first stab.

    Openie6 not very clear about this.


    Parameters
    ----------
    orig_sent
    ll_ilabel

    Returns
    -------

    """
    osent = orig_sentL.split("[unused1]")[0].strip()
    osent_words = get_words(osent)
    cctree = CCTree(osent, ll_ilabel)
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
    l_cctags = [["NONE"] * len(osent_words)]
    for depth, locs in enumerate(l_spanned_locs):
        cctags = l_cctags[depth]
        for k, loc in enumerate(sorted(locs)):
            if k == 0:
                cctags[loc] = "CP_START"
            else:
                cctags[loc] = "CP"
            if loc in depth_to_cclocs[depth]:
                cctags[loc] = "CC"
            if loc in depth_to_seplocs[depth]:
                cctags[loc] = "SEP"
    return l_cctags


def translate_extags_to_ilabels(extags):
    ilabels = []
    for extag in extags:
        ilabels.append(EXTAG_TO_ILABEL[extag])
    return ilabels


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
    orig_sentL_words = get_words(orig_sentL)
    max_len = len(orig_sentL_words)
    l_arg1 = []
    l_rel = []
    l_arg2 = []
    for k, extag in enumerate(extags):
        # extags may be padded
        if k < max_len:
            if extag == "ARG1":
                l_arg1.append(orig_sentL_words[k])
            elif extag == "REL":
                l_rel.append(orig_sentL_words[k])
            elif extag in ["ARG2", "LOC", "TIME"]:
                l_arg2.append(orig_sentL_words[k])
    if l_rel[-1] == "[unused1]":
        l_rel = ["[is]"] + l_rel[:-1]
    elif l_rel[-1] == "[unused2]":
        l_rel = ["[is]"] + l_rel[:-1] + ["[of]"]
    elif l_rel[-1] == "[unused3]":
        l_rel = ["[is]"] + l_rel[:-1] + ["[from]"]

    return l_arg1 + l_rel + l_arg2


def translate_cctags_to_ilabels(cctags):
    ilabels = []
    for cctag in cctags:
        ilabels.append(CCTAG_TO_ILABEL[cctag])
    return ilabels


def translate_cctags_to_words(cctags, orig_sentL):
    orig_senL_words = get_words(orig_sentL)
    max_len = len(orig_senL_words)
    cc_words = []
    for k, cctag in enumerate(cctags):
        # cctags may be padded
        if k < max_len and cctag not in "NONE":
            cc_words.append(orig_senL_words[k])
    return cc_words


def translate_ilabels_to_cctags(ilabels):
    """
    Openie6 seems to use CCTree to go from l_ilabels to l_cctags (see
    metric.get_coords()). However, I believe the l_ilabels used by CCTree,
    and the ones in this function are different.

    Parameters
    ----------
    ilabels

    Returns
    -------

    """

    cctags = []
    for ilabel in ilabels:
        cctags.append(ILABEL_TO_CCTAG(ilabel))
    return cctags


def translate_ilabels_to_extags(ilabels):
    extags = []
    for ilabel in ilabels:
        extags.append(ILABEL_TO_EXTAG(ilabel))

    return extags


def translate_ilabels_to_words_via_extags(ilabels, orig_sentL):
    extags = translate_ilabels_to_extags(ilabels)
    return translate_extags_to_words(extags, orig_sentL)


def translate_ilabels_to_words_via_cctags(ilabels, orig_sentL):
    cctags = translate_ilabels_to_cctags(ilabels)
    return translate_cctags_to_words(cctags, orig_sentL)


def file_translate_tags_to_ilabels(tag_type,
                                   in_fp,
                                   out_fp):
    with open(in_fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(out_fp, "w", encoding="utf-8") as f:
        for line in lines:
            if tag_type == "ex":
                if "NONE" not in line and "REL" not in line:
                    f.write(line.strip() + "\n")
                else:
                    ilabels = translate_extags_to_ilabels(get_words(line))
                    ilabels = [str(x) for x in ilabels]
                    f.write(" ".join(ilabels) + "\n")
            elif tag_type == "cc":
                if "NONE" not in line and "CC" not in line:
                    f.write(line.strip() + "\n")
                else:
                    ilabels = translate_cctags_to_ilabels(get_words(line))
                    ilabels = [str(x) for x in ilabels]
                    f.write(" ".join(ilabels) + "\n")
            else:
                assert False


def file_translate_ilabels_to_tags(tag_type,
                                   in_fp,
                                   out_fp):
    with open(in_fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(out_fp, "w", encoding="utf-8") as f:
        for line in lines:
            words= get_words(line)
            if all([word.isdigit() for word in words[0:5]]):
                ilabels = [int(x) for x in words()]
                if tag_type == "ex":
                    tags = translate_ilabels_to_extags(ilabels)
                elif tag_type == "cc":
                    tags = translate_ilabels_to_cctags(ilabels)
                else:
                    assert  False
                f.write(" ".join(tags) + "\n")


if __name__ == "__main__":
    from AllenTool import *


    def main1():
        at = AllenTool("testing_files/one_sample_allen.tsv")
        # print("llkm", at.osentL_to_exs)
        orig_sentL = list(at.osentL_to_exs.keys())[0]
        exs = at.osentL_to_exs[orig_sentL]
        for ex in exs:
            extags = translate_words_to_extags(ex)
            ilabels = translate_extags_to_ilabels(extags)
            words = translate_extags_to_words(extags, orig_sentL)
            print()
            print(" ".join(extags))
            print(" ".join(str(k) for k in ilabels))
            print(" ".join(words))
            print(ex.get_simple_sent())


    main1()
