"""

The purpose of this file is to bring together various static methods that
are used by SentenceAx for reading and writing various types of files.

"""

from Params import *
from sax_utils import *


def write_tags_file(out_fp,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens,
                    ll_confidence=None):
    """
    This method writes an extags or a cctags file. It works for both. It is
    called internally by both write_extags_file() and write_cctags_file().

    Parameters
    ----------
    out_fp: str
    l_orig_sent: list[str]
    ll_tags: list[list[list[str]]]
    with_unused_tokens: bool
    ll_confidence: list[[list[float]]

    Returns
    -------
    None

    """
    with open(out_fp, "w") as f:
        num_samples = len(l_orig_sent)
        for sam in range(num_samples):
            f.write(str(sam + 1) + "." + "\n")

            if with_unused_tokens:
                orig_sentL = l_orig_sent[sam] + UNUSED_TOKENS_STR
                f.write(orig_sentL + "\n")
            else:
                f.write(l_orig_sent[sam])
                for depth in range(len(ll_tags[0])):
                    end_str = "\n"
                    if ll_confidence:
                        end_str = "(" + ll_confidence[sam][depth] + ")"
                    f.write(ll_tags[sam][depth] + end_str)


def write_extags_file(out_fp,
                      l_orig_sent,
                      ll_tags,
                      ll_confidence=None):
    """
    This method writes an extags file.

    Parameters
    ----------
    out_fp: str
    l_orig_sent: list[str]
    ll_tags: list[list[list[str]]]
    ll_confidence: list[list[float]]

    Returns
    -------
    None

    """
    write_tags_file(out_fp,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens=True,
                    ll_confidence=ll_confidence)


def write_cctags_file(out_fp,
                      l_orig_sent,
                      ll_tags,
                      ll_confidence=None):
    """
    This method writes a cctags file.

    Parameters
    ----------
    out_fp: str
    l_orig_sent: list[str]
    ll_tags: list[list[list[str]]]
    ll_confidence: list[list[float]]

    Returns
    -------
    None

    """
    write_tags_file(out_fp,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens=False,
                    ll_confidence=ll_confidence)


def load_sub_osent2_to_osent2(in_fp, word_tokenize=False):
    """
    similar to Openie6.data_processing.load_conj_mapping()

    This method is never used by SentenceAx or Openie6.

    This method returns:
        if word_tokenize==False
            sub_osent2_to_osent2, similar to Openie6.mapping.
        if word_tokenize==True
            sent_to_words, similar to Openie6.conj_mapping.

    Parameters
    ----------
    in_fp: str
    word_tokenize: bool


    Returns
    -------
    dict[str, str]|dict[str, list[str]]

    """
    sub_osent2_to_osent2 = {}
    sent_to_words = {}
    with open(in_fp, "r", encoding="utf-8") as f:
        content = get_ascii(f.read())
        fixed_sent = ''
        for sample in content.split('\n\n'):
            for i, line in enumerate(sample.strip('\n').split('\n')):
                if i == 0:
                    fixed_sent = line
                else:
                    if not word_tokenize:
                        sub_osent2_to_osent2[line] = fixed_sent
                    else:
                        sent_to_words[line] = get_words(fixed_sent)
    if not word_tokenize:
        return sub_osent2_to_osent2
    else:
        return sent_to_words
