"""

The purpose of this file is to bring together various static methods that
are used by SentenceAx for reading and writing various types of files.

"""

from Params import *
from sax_utils import *


def write_tags_file(path,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens,
                    ll_confi=None):
    """
    This method writes an extags or a cctags file. It works for both. It is
    called internally by both write_extags_file() and write_cctags_file().

    Parameters
    ----------
    path: str
    l_orig_sent: list[str]
    ll_tags: list[list[list[str]]]
    with_unused_tokens: bool
    ll_confi: list[[list[float]]

    Returns
    -------
    None

    """
    with open(path, "w") as f:
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
                    if ll_confi:
                        end_str = "(" + ll_confi[sam][depth] + ")"
                    f.write(ll_tags[sam][depth] + end_str)


def write_extags_file(path,
                      l_orig_sent,
                      ll_tags,
                      ll_confi=None):
    """
    This method writes an extags file.

    Parameters
    ----------
    path: str
    l_orig_sent: list[str]
    ll_tags: list[list[list[str]]]
    ll_confi: list[list[float]]

    Returns
    -------
    None

    """
    write_tags_file(path,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens=True,
                    ll_confi=ll_confi)


def write_cctags_file(path,
                      l_orig_sent,
                      ll_tags,
                      ll_confi=None):
    """
    This method writes a cctags file.

    Parameters
    ----------
    path: str
    l_orig_sent: list[str]
    ll_tags: list[list[list[str]]]
    ll_confi: list[list[float]]

    Returns
    -------
    None

    """
    write_tags_file(path,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens=False,
                    ll_confi=ll_confi)


def load_sent_to_sent(in_fp, word_tokenize=False):
    """
    similar to Openie6.data_processing.load_conj_mapping()

    This method is never used by SentenceAx or Openie6.

    This method returns:
        if word_tokenize==False
            sent_to_sent, similar to Openie6.mapping.
        if word_tokenize==True
            sent_to_words, similar to Openie6.conj_mapping.

    Parameters
    ----------
    in_fp: str
    word_tokenize: bool


    Returns
    -------
    dict[str, str]

    """
    sent_to_sent = {}
    with open(in_fp, "r") as f:
        content = f.read()
        fixed_sent = ''
        for sample in content.split('\n\n'):
            for i, line in enumerate(sample.strip('\n').split('\n')):
                if i == 0:
                    fixed_sent = line
                else:
                    if not word_tokenize:
                        sent_to_sent[line] = fixed_sent
                    else:
                        sent_to_sent[line] = get_words(fixed_sent)
    return sent_to_sent
