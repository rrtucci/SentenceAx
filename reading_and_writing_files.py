from Params import *


def write_tags_file(path,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens,
                    ll_confi=None):
    """

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


def load_sent_to_sent(in_fp):
    """
    similar to Openie6.data_processing.load_conj_mapping() Our
    sent_to_sent is similar to Openie6 mapping and conj_mapping. This
    method works equally well for ExMetric.sent_to_sent and
    CCMetric.sent_to_words

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
                    sent_to_sent[line] = fixed_sent
    return sent_to_sent
