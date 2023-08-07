import difflib
from collections import defaultdict

class ClassFromDict(dict):
    """
    dot instead of [] notation access to dictionary attributes.
    Nice to know but won't use.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def update_dict(dict, new_dict, add_new_keys=True):
    for key in dict:
        if key in new_dict: # overlapping keys
            dict[key] = new_dict[key]
    if add_new_keys:
        for key in new_dict:
            if key not in dict: # new keys not in dict yet
                dict[key] = new_dict[key]


def get_words(ztz):
    # get_words("") = []
    if ztz:
        li = ztz.strip().split()
    else:
        li = []
    return li


def count_sub_reps(sub, full):  # formerly seq_in_seq
    # rep = repetitions
    # ["apple", "banana", "cherry"].count("cherry") # output 1
    # 'dog is in dog house'.count('dog') # output 2

    # str(["dog", "pet"]) # output "['dog', 'pet']"
    # the reason for the [1, -1] is to exclude '[' and ']'
    #  return str(full)[1:-1].count(str(sub)[1:-1])
    return " ".join(full).count(" ".join(sub))


def sub_exists(sub, full, start_loc):  # formerly starts_with
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
    xlist_s = sorted(xlist)
    assert xlist_s[0] >=0
    y0 = cost_fun(1E4)
    x0 = -1
    for x in xlist_s:
        y = cost_fun(x)
        if y < y0:
            y0 = y
            x0 = x

    return x0, y0

def none_dd(di):
    # dd = default dictionary
    return defaultdict(lambda: None, di)

if __name__ == "__main__":
    def main():
        h = {"x": 5, "y": 3}
        H = ClassFromDict(h)
        print(H.x)  # Output: 5
        print(H.y)  # Output: 3
        H.y = 5
        print(H.y, h["y"]) # output 5,3

        def F(x, y):
            return x + y

        print(F(**h))  # Output: 8

    main()