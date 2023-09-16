import difflib
from SaxExtraction import *
from carb_subset.oie_readers.extraction import Extraction
import re


def count_sub_reps(sub, full):
    """
    similar to Openie6.data_processing.seq_in_seq()

    Parameters
    ----------
    sub
    full

    Returns
    -------

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
    similar to Openie6.data_processing.starts_with()


    """
    return all(sub[i] == full[start_loc + i] for i in range(0, len(sub)))


def has_2_matches(matches):
    """
    > sm = difflib.SequenceMatcher(None, a='ACT', b='ACTGACT')
    > sm.get_matching_blocks()
    [Match(a=0, b=0, size=3), Match(a=3, b=7, size=0)]
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
    """
    return difflib.SequenceMatcher(None, list0, list1). \
        get_matching_blocks()

def find_xlist_item_that_minimizes_cost_fun(xlist, cost_fun):
    y0 = cost_fun(1E4)
    x0 = -1
    for x in xlist:
        y = cost_fun(x)
        if y < y0:
            y0 = y
            x0 = x

    return x0, y0


def get_extraction(ex_ilabels, orig_sentL, score):
    """
    similar to Openie6.model.process_extraction()

    ILABEL_TO_EXTAG={
        0: 'NONE',
        1: 'ARG1',
        2: 'REL',
        3: 'ARG2',
        4: 'ARG2',
        5: 'NONE'
    }


    Parameters
    ----------
    ex_labels:
    orig_sentL
    score

    Returns
    -------

    """
    ex_ilabels = ex_ilabels.to_list()  # change from torch tensor to list

    l_rel = []
    l_arg1 = []
    l_arg2 = []
    # l_loc_time=[]
    # l_args = []
    rel_case = 0
    for i, word in enumerate(get_words(orig_sentL)):
        if '[unused' in word:
            if ex_ilabels[i] == 2:  # REL
                rel_case = int(
                    re.search('\[unused(.*)\]', word).group(1)
                )  # this returns either 1, 2 or 3
            continue
        if ex_ilabels[i] == 0:  # NONE
            pass
        elif ex_ilabels[i] == 1:  # ARG1
            l_arg1.append(word)
        elif ex_ilabels[i] == 2:  # REL
            l_rel.append(word)
        elif ex_ilabels[i] == 3:  # ARG2
            l_arg2.append(word)
        elif ex_ilabels[i] == 4:  # ARG2
            # l_loc_time.append(word)
            l_arg2.append(word)
        else:
            assert False

    rel = ' '.join(l_rel).strip()
    if rel_case == 1:
        rel = 'is ' + rel
    elif rel_case == 2:
        rel = 'is ' + rel + ' of'
    elif rel_case == 3:
        rel = 'is ' + rel + ' from'

    arg1 = ' '.join(l_arg1).strip()
    arg2 = ' '.join(l_arg2).strip()

    # args = ' '.join(l_args).strip()
    # loc_time = ' '.join(l_loc_time).strip()
    # if not self.params_d["no_lt"]: # no_lt = no loc time
    #     arg2 = (arg2 + ' ' + loc_time + ' ' + args).strip()

    extraction = SaxExtraction(orig_sentL,
                               arg1,
                               rel,
                               arg2,
                               score=score)

    return extraction