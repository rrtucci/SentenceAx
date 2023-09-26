import difflib
from carb_subset.oie_readers.extraction import Extraction
import re


def count_sub_reps(sub, full):
    """
    similar to Openie6.data_processing.seq_in_seq()

    Parameters
    ----------
    sub: list
    full: list

    Returns
    -------
    int

    """
    # rep = repetitions
    # ["apple", "banana", "cherry"].count("cherry") # output 1
    # 'dog is in dog house'.count('dog') # output 2

    # str(["dog", "pet"]) # output "['dog', 'pet']"
    # the reason for the [1, -1] is to exclude '[' and ']'
    #  return str(full)[1:-1].count(str(sub)[1:-1])
    return " ".join(full).count(" ".join(sub))


def sub_exists(sub, full, start_loc):
    """

    Parameters
    ----------
    sub: list[Any]
    full: list[Any]
    start_loc: int

    Returns
    -------
    bool

    """
    # similar to Openie6.data_processing.starts_with()

    return all([sub[i] == full[start_loc + i] for i in range(len(sub))])


def has_2_matches(matches):
    """
    > sm = difflib.SequenceMatcher(None, a='ACT', b='ACTGACT')
    > sm.get_matching_blocks()
    [Match(a=0, b=0, size=3), Match(a=3, b=7, size=0)]

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
    len(matches) > 2 and
    matches[0].a == 0 and
    all(matches[i].a == matches[i-1].a + matches[i-1].size
    for i in range(1, len(matches)-1)) and
    matches[-2].a + matches[-2].size == matches[-1].a

    # matches[-1].a - matches[-2].a == matches[-2].size
    # is just li[i] when i=len(matches)-1

    Parameters
    ----------
    matches: list[Match]

    Returns
    -------
    bool
    """
    li = [matches[i].a - matches[i - 1].a == matches[i - 1].size
          for i in range(1, len(matches))]
    return len(matches) > 2 and \
        matches[0].a == 0 and \
        all(li) and \
        matches[len(matches) - 1] == 0


def get_matches(list0, list1):
    """
    > sm = difflib.SequenceMatcher(None, a='ACT', b='ACTGACT')
    > sm.get_matching_blocks()
    [Match(a=0, b=0, size=3), Match(a=3, b=7, size=0)]

    Parameters
    ----------
    list0: list
    list1: list

    Returns
    -------
    list[Match]

    """
    return difflib.SequenceMatcher(None, list0, list1). \
        get_matching_blocks()


def find_xlist_item_that_minimizes_cost_fun(xlist, cost_fun):
    """

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
