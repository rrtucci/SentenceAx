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
