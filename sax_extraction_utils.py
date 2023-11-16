"""

The purpose of this file is to gather together a bunch of global methods
that are used in class SaxExtraction. We could just as well have made these
methods staticmethods inside class SaxExtraction.

"""

import difflib
import re


def count_sub_reps(sub, full):
    """
    similar to Openie6.data_processing.seq_in_seq()

    This method counts the number of times the string `" ".join(sub)`
    appears in the string `" ".join(full)`.

    rep = repetitions
    ["apple", "banana", "cherry"].count("cherry") # output 1
    'dog is in dog house'.count('dog') # output 2

    str(["dog", "pet"]) # output "['dog', 'pet']"
    the reason for the [1, -1] in Openie6 is to exclude '[' and ']'
    return str(full)[1:-1].count(str(sub)[1:-1])

    Parameters
    ----------
    sub: list[str]
    full: list[str]

    Returns
    -------
    int

    """
    return " ".join(full).count(" ".join(sub))


def sub_exists(sub, full, start_loc):
    """
    similar to Openie6.data_processing.starts_with()

    This method returns True iff a " ".join(sub) is part of " ".join(full)
    and the first word of sub is at position `start_loc` of full.

    Parameters
    ----------
    sub: list[str]
    full: list[str]
    start_loc: int

    Returns
    -------
    bool

    """
    # similar to Openie6.data_processing.starts_with()
    if len(full) - 1 < start_loc + len(sub) - 1:
        return False
    return all([sub[i] == full[start_loc + i] for i in range(len(sub))])


def has_2_matches(matches):
    """
    This method returns True iff `matches` contains 2 matches. The second
    match is the useless one of size zero.

    See misc/SequenceMatcher-examples.py for examples of
    difflib.SequenceMatcher usage

    Parameters
    ----------
    matches: list[Match]
    """
    return len(matches) == 2 and \
        matches[0].a == 0 and \
        matches[1].a - matches[0].a == matches[0].size and \
        matches[1].size == 0


def has_gt_2_matches(matches):
    """
    This is similar to a code snippet from Openie6.data_processing.label_arg()

    This method returns True iff `matches` contains > 2 matches that are
    contiguous, with matches[0].a == 0

    See misc/SequenceMatcher-examples.py for examples of
    difflib.SequenceMatcher usage

    Parameters
    ----------
    matches: list[Match]

    Returns
    -------
    bool
    """
    # matches[-1].a - matches[-2].a == matches[-2].size
    # is just li[i] when i=len(matches)-1
    li = [matches[i].a == matches[i - 1].a + matches[i - 1].size
          for i in range(1, len(matches))]
    cond1 = len(matches) > 2
    cond2 = matches[0].a == 0
    cond3 = all(li)

    return cond1 and cond2 and cond3


def get_matches(list0, list1):
    """
    This method finds matching blocks in two lists `list0` and `list1`.

    See misc/SequenceMatcher-examples.py for examples of
    difflib.SequenceMatcher usage

    Parameters
    ----------
    list0: Sequence
    list1: Sequence

    Returns
    -------
    list[Match]

    """
    return difflib.SequenceMatcher(None, list0, list1). \
        get_matching_blocks()


def find_xlist_item_that_minimizes_cost_fun(xlist, cost_fun):
    """
    This method finds the item in the list `xlist` that minimizes the
    function `cost_fun`. The method returns the tuple

    argmin_{x \in xlist} cost_fun(x), min_{x \in xlist} cost_fun(x)

    Parameters
    ----------
    xlist: list
    cost_fun: function

    Returns
    -------
    Any, Any

    """
    y0 = cost_fun(1E4)
    x0 = -1
    for x in xlist:
        y = cost_fun(x)
        if y < y0:
            y0 = y
            x0 = x

    return x0, y0
