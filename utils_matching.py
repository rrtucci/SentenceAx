"""

This file contains string matching methods that build upon the difflib
python package. These methods are used in the class SaxExtraction.

Ref. on difflib.SequenceMatcher:
https://stackoverflow.com/questions/35517353/how-does-pythons-sequencematcher-work

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
    contiguous in a, with matches[0].a == 0. The last match is the useless
    one of size zero.

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
    # print("cond1, cond2, cond3", cond1, cond2, cond3)

    return cond1 and cond2 and cond3


def get_matches(list0, list1):
    """
    This method finds matching blocks in two lists `list0` and `list1`.

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


def print_matches(a, b):
    """
    This method prints and returns the matches obtained via
    difflib.SequenceMatcher.

    Parameters
    ----------
    a: Sequence
    b: Sequence

    Returns
    -------
    list[Match]

    """
    print()
    print("a=", a)
    print("b=", b)
    matches = get_matches(a, b)
    print(matches)
    return matches


if __name__ == "__main__":
    def main1():
        matches = print_matches(a='ACT', b='ACTGACT')
        print("has 2 matches?", has_2_matches(matches))

        # output [Match(a=0, b=0, size=3), Match(a=3, b=7, size=0)]

        # note that ACT appears twice in b but only first match is given.

        # Match(a=0, b=0, size=3): This indicates that a matching block of
        # size 3 starts at index 0 in sequence a and index 0 in sequence b.
        #
        # Match(a=3, b=7, size=0): This indicates that there is a matching
        # block of size 0 starting at index 3 in sequence a and index 7 in
        # sequence b. This "useless" zero sized block at the end is always
        # given.

        # a and b stand for sequences in the constructor
        # difflib.SequenceMatcher(None, a, b), but for sequence locations in
        # the output of get_matching_blocks()

        print_matches(a='abcdef', b='acdef')

        matches = \
            print_matches(
                a=['apple', 'banana', 'orange', 'kiwi'],
                b=['apple', 'grape', 'banana', 'orange', 'kiwi'])
        print("has > 2 matches?", has_gt_2_matches(matches))


    main1()
