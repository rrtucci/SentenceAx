"""

The purpose of this file is to gather together a bunch of global methods
that take as input a list, and return as output a new list, where the input
and output lists are lists of words, tags (cctags or extags), or ilabels (in
range(0:6)).

The first 2 methods: translate_words_to_extags() and 
translate_words_to_cctags(), are the hardest. All the other methos in this 
fiile are fairly trivial.

Recall the following global variables declared in file sax_globals.py.

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

The file `misc/openie6-input-file-formats.txt` in SentenceAx contains
examples of the various types of file formats used by Openie6. The lines in
`misc/openie6-input-file-formats.txt`, when reduced to words with get_words(
line), are typical of the input lists for the methods defined in this file.

"""
from Params import *
from utils_gen import *
from SaxExtraction import *
from CCTree import *


def translate_words_to_extags(ex):
    """
    Translate the words get_words(ex.simple_sent()) in an SaxExtraction `ex`
    to a list[ str] of extags. This is hard, but all the hard work is done
    within SaxExtraction.

    Parameters
    ----------
    ex: SaxExtraction

    Returns
    -------
    list[str]

    """
    if not ex.extags_are_set:
        ex.set_extags()
    return ex.extags


def translate_words_to_cctags(words):
    """
    Translate all the words in a simple sentence extracted from osent,
    to a cctags list.

    Openie6 not very clear about this so we leave it blank for now.

    Parameters
    ----------
    words: list[str]

    Returns
    -------
    list[str]
    """
    return words


def translate_extags_to_ilabels(extags):
    """
    Obvious what method does from its name.

    Parameters
    ----------
    extags: list[str]

    Returns
    -------
    list[int]

    """
    ilabels = []
    for extag in extags:
        ilabels.append(EXTAG_TO_ILABEL[extag])
    return ilabels


def translate_extags_to_words(extags, orig_sentL):
    """
    inferred from Openie6 data_processing.label_is_of_relations()
    
    Obvious what method does from its name.

    Parameters
    ----------
    extags: list[str]
    orig_sentL: str

    Returns
    -------
    list[str]

    """
    orig_sentL_words = get_words(orig_sentL)
    max_len = len(orig_sentL_words)
    arg1_words = []
    rel_words = []
    arg2_words = []
    for k, extag in enumerate(extags):
        # extags may be padded
        if k < max_len:
            if extag == "ARG1":
                arg1_words.append(orig_sentL_words[k])
            elif extag == "REL":
                rel_words.append(orig_sentL_words[k])
            elif extag in ["ARG2", "LOC", "TIME"]:
                arg2_words.append(orig_sentL_words[k])
    if rel_words[-1] == "[unused1]":
        rel_words = ["[is]"] + rel_words[:-1]
    elif rel_words[-1] == "[unused2]":
        rel_words = ["[is]"] + rel_words[:-1] + ["[of]"]
    elif rel_words[-1] == "[unused3]":
        rel_words = ["[is]"] + rel_words[:-1] + ["[from]"]

    return arg1_words + rel_words + arg2_words


def translate_cctags_to_ilabels(cctags):
    """
    Obvious what method does from its name.

    Parameters
    ----------
    cctags: list[str]

    Returns
    -------
    list[int]

    """
    ilabels = []
    for cctag in cctags:
        ilabels.append(CCTAG_TO_ILABEL[cctag])
    return ilabels


def translate_cctags_to_words(cctags, orig_sentL):
    """
    Obvious what method does from its name.

    Parameters
    ----------
    cctags: list[str]
    orig_sentL: str

    Returns
    -------
    list[str]

    """
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
    
    Obvious what method does from its name.

    Parameters
    ----------
    ilabels: list[int]

    Returns
    -------
    list[str]

    """

    cctags = []
    for ilabel in ilabels:
        cctags.append(ILABEL_TO_CCTAG(ilabel))
    return cctags


def translate_ilabels_to_extags(ilabels):
    """
    Obvious what method does from its name.

    Parameters
    ----------
    ilabels: list[int]

    Returns
    -------
    list[str]

    """
    extags = []
    for ilabel in ilabels:
        extags.append(ILABEL_TO_EXTAG(ilabel))

    return extags


def translate_ilabels_to_words(ilabels, orig_sentL, route="ex"):
    """
    This method translates ilabels->extags->words if route="ex",
    or ilabels->cctags->words if route="cc".

    IMPORTANT: One would think that both routes should give the same result
    because 0->NONE->"" and (non-zero int)->??->word for both routes. But
    that is not the case because the cctags files have much fewer non-NONE
    cctags than the extags files.

    Parameters
    ----------
    ilabels: list[int]
    orig_sentL: str
    route: str

    Returns
    -------
    list[str]

    """
    if route == "ex":
        extags = translate_ilabels_to_extags(ilabels)
        return translate_extags_to_words(extags, orig_sentL)
    elif route == "cc":
        cctags = translate_ilabels_to_cctags(ilabels)
        return translate_cctags_to_words(cctags, orig_sentL)
    else:
        assert False


def file_translate_tags_to_ilabels(tag_type,
                                   in_fp,
                                   out_fp):
    """
    This method reads a tags file at `in_fp` and writes an ilabels file at 
    `out_fp`. The tags file is an extags file (if tag_type=="ex") or a
    cctatgs file (if tag_type=="cc")`.
    
    file_translate_tags_to_ilabels() and file_translate_ilabels_to_tags() 
    perform inverse operations.

    Parameters
    ----------
    tag_type: str
    in_fp: str
    out_fp: str

    Returns
    -------
    None

    """
    with open(in_fp, "r", encoding="utf-8") as f:
        lines = get_ascii(f.readlines())
    with open(out_fp, "w") as f:
        for line in lines:
            if not line or SAMPLE_SEPARATOR in line:
                continue
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
    """
    This method reads an ilabels file at `in_fp` and writes a tags file at
    `out_fp`. The tags file is an extags file (if tag_type=="ex") or a
    cctatgs file (if tag_type=="cc")`.
    
    file_translate_tags_to_ilabels() and file_translate_ilabels_to_tags() 
    perform inverse operations.

    Parameters
    ----------
    tag_type: str
    in_fp: str
    out_fp: str

    Returns
    -------
    None

    """
    with open(in_fp, "r", encoding="utf-8") as f:
        lines = get_ascii(f.readlines())
    with open(out_fp, "w") as f:
        for line in lines:
            if not line or SAMPLE_SEPARATOR in line:
                continue
            words = get_words(line)
            if all([word.isdigit() for word in words[0:5]]):
                ilabels = [int(x) for x in words]
                if tag_type == "ex":
                    tags = translate_ilabels_to_extags(ilabels)
                elif tag_type == "cc":
                    tags = translate_ilabels_to_cctags(ilabels)
                else:
                    assert False
                f.write(" ".join(tags) + "\n")


def file_translate_tags_to_words(tag_type,
                                 in_fp,
                                 out_fp):
    """
    This method reads a tags file at `in_fp` and writes a words file at
    `out_fp`. The tags file is an extags file (if tag_type=="ex") or a
    cctatgs file (if tag_type=="cc")`.


    Parameters
    ----------
    tag_type: str
    in_fp: str
    out_fp: str

    Returns
    -------
    None

    """
    with open(in_fp, "r", encoding="utf-8") as f:
        lines = get_ascii(f.readlines())
    with open(out_fp, "w") as f:
        for line in lines:
            if not line or SAMPLE_SEPARATOR in line:
                continue
            if tag_type == "ex":
                if "NONE" not in line and "REL" not in line:
                    f.write(line.strip() + "\n")
                    osentL = redoL(line)
                else:
                    words = translate_extags_to_words(get_words(line),
                                                      osentL)
                    f.write(" ".join(words) + "\n")
            elif tag_type == "cc":
                if "NONE" not in line and "CC" not in line:
                    f.write(line.strip() + "\n")
                    osentL = redoL(line)
                else:
                    words = translate_cctags_to_words(get_words(line),
                                                      osentL)
                    f.write(" ".join(words) + "\n")
            else:
                assert False


if __name__ == "__main__":
    from AllenTool import *


    def main1():
        at = AllenTool("tests/one_sample_allen.tsv")
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
