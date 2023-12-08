from sax_utils import *
import re


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


def process_l_sample_str(l_sample_str,
                         ll_cc_word=None):
    """
    This method is similar to part of Openie6.run.splitpredict()

    l_sample_str ~ Openie6.example_sentences
    ll_cc_word ~ Openie6.all_conj_words

    l_osentL ~ Openie6.orig_sentences
    l_split_sentL ~ Openie6.sentences
    sub_osent_to_osent ~ Openie6.mapping
    osent_to_words ~ Openie6.conj_word_mapping

    If you are trying to understand the meaning of Openie6.mapping and
    Openie6.conj_word_mapping, this is where they are filled from scratch.

    This method is a really productive workhorse. It takes a list of
    sample strings `l_sample_str` and it returns 4 things:

    l_osentL: list[str]
        This is a list of osentL (original sentences, with unused token
        str)
    l_split_sentL: list[str]
        This is a list of sents produced by split but before extract. In
        case the osent yielded no split sents, the osent is included
        instead. If there are split sents, the osent is not included.
    sub_osent_to_osent: dict[str, str]
        This maps a bunch of sub osent (i.e., either extractions or the
        original sent) to the original sent.
    osent_to_words: dict[str, list[str]]
        This maps osent to some words. This corresponds to
        Openie6.conj_words_mapping, which Openie6 fills but never uses.
        We include it in SentenceAx only for the sake of completeness.


    Parameters
    ----------
    l_sample_str: list[str]
    ll_cc_word: list[list[str]] | None

    Returns
    -------
        list[str], list[str], dict[str, str], dict[str, list[str]]
            l_osentL, l_split_sentL, sub_osent_to_osent, osent_to_words

    """
    l_osentL = []
    l_split_sentL = []
    sub_osent_to_osent = {}
    osent_to_words = {}
    for sample_id, sample_str in enumerate(l_sample_str):
        l_sent = sample_str.strip().split("\n")
        sent0L = redoL(l_sent[0])
        sent0 = undoL(sent0L)
        # print("nnnmjk**********", sent0L)
        if len(l_sent) == 1:
            l_osentL.append(sent0L)
            sub_osent_to_osent[sent0] = sent0
            if ll_cc_word:
                # model.osent_to_words is filled but never used
                osent_to_words[sent0] = \
                    ll_cc_word[sample_id]
            l_split_sentL.append(sent0L)
        # len(l_sent) > 1
        else:
            l_osentL.append(sent0L)
            # IMP: we omit this on purpose
            # l_split_sentL.append(sent0L)
            if ll_cc_word:
                # model.osent_to_words is filled but never used
                osent_to_words[sent0] = \
                    ll_cc_word[sample_id]
            for sent in l_sent[1:]:
                sent = undoL(sent)
                if sent not in sub_osent_to_osent.keys():
                    sub_osent_to_osent[sent] = sent0
                    l_split_sentL.append(redoL(sent))

    return l_osentL, l_split_sentL, sub_osent_to_osent, osent_to_words


def rebuild_l_sample_str(l_sample_str,
                         l_osentL,
                         sub_osent_to_osent):
    """
    This method takes a sample_str list `l_sample_str` as input and
    returns a "rebuilt" sample_str list `l_sample_str_new`. The new
    sample_str list has fewer (len(l_osentL)) items than the old one.
    The first line of each item in l_sample_str_new` is an item from
    `l_osentL`.

    Parameters
    ----------
    l_sample_str: list[str]
    l_osentL: list[str]
    sub_osent_to_osent: dict[str, str]

    Returns
    -------
    list[str]

    """
    l_sample_str_new = [""] * len(l_osentL)
    for isam, osentL in enumerate(l_osentL):
        osent = undoL(osentL)
        l_sample_str_new[isam] = osentL + "\n"
        for sample_str in l_sample_str:
            l_sent = sample_str.strip().split("\n")
            sub_osent0 = undoL(l_sent[0])
            # print_list("nnnvbg l_sent", l_sent)
            if sub_osent_to_osent[sub_osent0] == osent:
                if len(l_sent) == 1:
                    l_sample_str_new[isam] += sub_osent0 + "\n"
                else:
                    for sent in l_sent[1:]:
                        l_sample_str_new[isam] += undoL(sent) + "\n"
    return l_sample_str_new


def write_predictions_in_original_order(pred_in_fp,
                                        unsorted_fp,
                                        sorted_fp):
    """
    This method reads the file `pred_in_fp` with the sentences we want
    to split. It also reads an ssent (simple sentences) file
    `unsorted_fp` with a split. Each sample in `unsorted_fp` is assumed
    to be separated by a line with the SAMPLE_SEPARATOR str.

    After reading these 2 files, the method writes a file `sorted_fp`
    with the same samples as `unsorted_fp`, but in the original order
    given by the osents in `pred_in_fp`.

    If the strings `unsorted_fp` and `sorted_fp` are the same, the
    unsorted file is overwritten.


    Parameters
    ----------
    pred_in_fp: str
    unsorted_fp: str
    sorted_fp: str

    Returns
    -------
    None

    """

    with open(pred_in_fp, "r", encoding="utf-8") as f:
        l_osent = get_ascii(f.readlines())

    with open(unsorted_fp, "r", encoding="utf-8") as f:
        unsorted_content = get_ascii(f.read())

    sep = SAMPLE_SEPARATOR
    unsorted_content = unsorted_content.strip().strip(sep)
    l_unsorted_sample_str = unsorted_content.split(sep)
    osent_to_sample_str = {}
    for sample_str in l_unsorted_sample_str:
        l_sent = undoL(sample_str.strip().split("\n"))
        osent_to_sample_str[undoL(l_sent[0]).strip()] = sample_str

    l_sorted_sample_str = []
    for osent in l_osent:
        osent = osent.strip()
        if osent in osent_to_sample_str.keys():
            l_sorted_sample_str.append(osent_to_sample_str[osent])
        else:
            print("This sentence not in pred_in_fp:\n" + osent)
            assert False

    write_l_sample_str(l_sorted_sample_str,
                       sorted_fp,
                       appended=False,
                       numbered=True)


def read_l_sample_str(in_fp, numbered):
    """

    Parameters
    ----------
    in_fp: str
    numbered: bool

    Returns
    -------
    list[str]

    """
    with open(in_fp, "r") as f:
        content = f.read()
    if numbered:
        pattern = re.compile(SAMPLE_SEPARATOR + r'\d+\n')
        content = re.sub(pattern,
                         repl=SAMPLE_SEPARATOR + "\n",
                         string=content)
    content = content.strip().strip(SAMPLE_SEPARATOR).strip()
    l_sample_str = content.split(SAMPLE_SEPARATOR + "\n")
    return l_sample_str


def prune_l_sample_str(l_sample_str):
    """
    This method a list of sample_str by a new list of sample_str. The new
    list of sample_str the same as the one, exact that in each
    sample, ssents that equal the osent have been removed and ssents that
    are repeats have been replaced by a single copy of the ssent.

    Parameters
    ----------
    l_sample_str: str

    Returns
    -------
    None

    """
    l_sample_str_new = [""]*len(l_sample_str)
    for isam, sample_str in enumerate(l_sample_str):
        l_sent = sample_str.strip().split("\n")
        sent0L = l_sent[0]
        sent0 = undoL(l_sent[0])
        l_sample_str_new[isam] += sent0L + "\n"
        l_sent_new = []
        for sent in l_sent[1:]:
            if sent != sent0 and \
                    sent not in l_sent_new:
                l_sent_new.append(sent)
                l_sample_str_new[isam] += sent + "\n"

    return l_sample_str_new
