import difflib
from sax_globals import *


# important
# carb has its own version of Extraction,
# so call ours SaxExtraction, SAx=SentenceAx
# from carb_subset.oie_readers.extraction import Extraction


# extag = extraction tag
# pred=predicate same as rel=relation
class SaxExtraction():
    def __init__(self,
                 orig_sent="",  # original sentence, before extractions
                 # or adding unused tokens
                 arg1="",
                 rel="",
                 arg2="",
                 confidence=None):
        """
        assume only one: arg1, rel, arg2, time_arg, loc_arg
        assume args list is empty
        """

        self.confidence = confidence
        sent = orig_sent + UNUSED_TOKENS_STR
        self.sent_pair = (sent, get_words(sent))
        self.arg1_pair = (arg1, get_words(arg1))
        self.rel_pair = (rel, get_words(rel))
        self.arg2_pair = (arg2, get_words(arg2))
        self.time_arg_pair = ("", [])
        self.loc_arg_pair = ("", [])

        self.base_extags = ["NONE", "ARG1", "REL", "ARG2", "TIME", "LOC"]
        self.sent_extags = ["NONE"] * len(self.sent_pair[1])
        self.base_extag_is_assigned = {extag_name: False
                                       for extag_name in self.base_extags}

    def add_arg1(self, arg1):
        self.arg1_pair = (arg1, get_words(arg1))

    def add_rel(self, rel):
        self.rel_pair = (rel, get_words(rel))

    def add_arg2(self, arg2):
        self.arg2_pair = (arg2, get_words(arg2))

    def add_time_arg(self, str0):
        self.time_arg_pair = (str0, get_words(str0))

    def add_loc_arg(self, str0):
        self.loc_arg_pair = (str0, get_words(str0))

    def to_string(self):
        li = [self.arg1_pair[1],
              self.rel_pair[1],
              self.arg2_pair[1],
              self.loc_arg_pair[1],
              self.time_arg_pair[1]]
        li = [x for x in li if x]
        return " ".join(li)

    def set_is_extagged_to_true(self, extag_name):
        assert extag_name in self.base_extags
        self.base_extag_is_assigned[extag_name] = True

    def set_extags_of_2_matches(self, matches, extag_name):
        assert extag_name in self.base_extags
        assert has_2_matches(matches)
        m0 = matches[0]
        self.sent_extags[m0.b: m0.b + m0.size] = [extag_name] * m0.size
        self.set_is_extagged_to_true(extag_name)

    def set_extags_of_gt_2_matches(self, matches, extag_name):
        assert extag_name in self.base_extags
        assert has_gt_2_matches(matches)
        self.set_is_extagged_to_true(extag_name)
        for m in matches:
            self.sent_extags[m.b: m.b + m.size] = \
                [extag_name] * m.size

    def set_extags_of_arg2(self):  # formerly label_arg2()

        li_2lt = self.arg2_pair[1] + self.loc_arg_pair[1] + \
                 self.time_arg_pair[1]
        li_2tl = self.arg2_pair[1] + self.time_arg_pair[1] + \
                 self.loc_arg_pair[1]
        li_2t = self.arg2_pair[1] + self.time_arg_pair[1]
        li_2l = self.arg2_pair[1] + self.loc_arg_pair[1]
        li_2 = self.arg2_pair[1]
        with_2_lists = [li_2lt, li_2tl, li_2t, li_2l, li_2]

        li_tl = self.time_arg_pair[1] + self.loc_arg_pair[1]
        li_lt = self.loc_arg_pair[1] + self.time_arg_pair[1]
        li_t = self.time_arg_pair[1]
        li_l = self.loc_arg_pair[1]
        without_2_lists = [li_tl, li_lt, li_t, li_l]

        if len(self.arg2_pair[1]) != 0:
            for li in with_2_lists:
                if count_sub_reps(li, self.sent_pair[1]) == 1:
                    matches = get_matches(li, self.sent_pair[1])
                    self.set_extags_of_2_matches(matches, "ARG2")
                    return
        else:  # len(self.arg2_pair[1]) == 0
            for li in without_2_lists:
                if count_sub_reps(li, self.sent_pair[1]) == 1:
                    matches = get_matches(li, self.sent_pair[1])
                    self.set_extags_of_2_matches(matches, "ARG2")
                    return
        # if everything else fails, still
        # set this flag true
        self.arg2_extagged = True

    def set_extags_of_arg1_or_rel(self, arg_name):
        if arg_name == "arg1":
            arg_words = self.arg1_pair[1]
        elif arg_name == "rel":
            arg_words = self.rel_pair[1]
        else:
            assert False
        if count_sub_reps(arg_words, self.sent_pair[1]) == 1:
            matches = get_matches(arg_words, self.sent_pair[1])
            self.set_extags_of_2_matches(matches, arg_name.upper())

        elif count_sub_reps(arg_words, self.sent_pair[1]) == 0:
            # sub doesn't fit in one piece into full
            # but still possible it exists in fractured form
            matches = get_matches(arg_words, self.sent_pair[1])
            if has_gt_2_matches(matches):
                self.set_extags_of_gt_2_matches(matches, arg_name.upper())

    def set_extags_for_unused_num(self, unused_num):
        assert unused_num in [1, 2, 3]
        unused_str = '[unused' + unused_num + ']'
        # this equals -3, -2, -1 for 1, 2, 3
        unused_loc = -4 + unused_num
        if unused_num in [2, 3]:
            last_rel_loc = -1
        elif unused_num == 1:
            last_rel_loc = len(self.rel_pair[1])
        else:
            assert False
        # [1:-1] when self.rel_pair[0][0] = "[is]"
        # and self.rel_pair[0][-1] = "of" or 'from"
        # [1: ] when self.rel_pair[0][0] = "[is]"
        # and self.rel_pair[0][-1] is anything
        # that doesn't start with "["
        if count_sub_reps(self.rel_pair[1][1:last_rel_loc],
                          self.sent_pair[1]) == 1:
            matches = get_matches(
                self.rel_pair[1][1:last_rel_loc], self.sent_pair[1])
            self.set_extags_of_2_matches(matches, "REL")
            assert self.sent_pair[1][unused_loc] == unused_str
            self.sent_extags[unused_loc] = 'REL'
        elif len(self.rel_pair[1]) > 2 and \
                count_sub_reps(self.rel_pair[1][1:last_rel_loc],
                               self.sent_pair[1]) == 0:
            matches = get_matches(
                self.rel_pair[1][1:last_rel_loc], self.sent_pair[1])
            # sub doesn't fit in one piece into full
            # but still possible it exists in fractured form
            if has_gt_2_matches(matches):
                self.set_extags_of_gt_2_matches(matches, "REL")
                assert self.sent_pair[1][unused_loc] == unused_str
                # if sent starts with "[is]" and ends with
                # anything, "[of]", or "[from]"
                # then switch the extag at sent positions of
                # [unused1], [unused2], [unused3]
                # from NONE to REL
                self.sent_extags[unused_loc] = 'REL'


    def set_extags_of_IS_OF_FROM(self):
        # sent can have implicit "is", "of", "from"
        # inserted in by hand and indicated by "[is]", "[of]" , "[from]"
        # Note that this is different from explicit "is", "of", "from"
        rel_is_extagged = self.base_extag_is_assigned["rel"]
        if (not rel_is_extagged) and len(self.rel_pair[1]) > 0:
            # IS
            if self.rel_pair[0] == '[is]':
                self.set_is_extagged_to_true("REL")
                assert self.sent_pair[1][-3] == '[unused1]'
                self.sent_extags[-3] = 'REL'
            # IS-OF
            elif self.rel_pair[1][0] == '[is]' and \
                    self.rel_pair[1][-1] == '[of]':
                self.set_extags_for_unused_num(2)
            # IS  FROM
            elif self.rel_pair[1][0] == '[is]' and \
                    self.rel_pair[1][-1] == '[from]':
                self.set_extags_for_unused_num(3)
            # IS
            elif self.rel_pair[1][0] == '[is]' and\
                    len(self.rel_pair[1]) > 1:
                assert self.rel_pair[1][-1].startswith('[') == ""
                self.set_extags_for_unused_num(1)


    def set_extags_of_arg1_if_repeated(self):
        rel_is_extagged = self.base_extag_is_assigned["REL"]
        arg1_is_extagged = self.base_extag_is_assigned["ARG1"]

        if rel_is_extagged and \
                arg1_is_extagged == False and \
                count_sub_reps(self.arg1_pair[1], self.sent_pair[1]) > 1:
            starting_locs = [j for j in
                             range(len(self.sent_pair[1])) if
                             sub_exists(self.arg1_pair[1],
                                        self.sent_pair[1], j)]
            assert len(starting_locs) > 1

            min_dist = int(1E8)
            if 'REL' in self.sent_extags:
                rel_loc = self.sent_extags.index('REL')
                final_loc = -1

                for loc in starting_locs:
                    dist = abs(rel_loc - loc)
                    if dist < min_dist:
                        min_dist = dist
                        final_loc = loc

                assert self.arg1_pair[1] == self.sent_pair[1][
                                            final_loc: final_loc + len(
                                                self.arg1_pair[1])]
                self.set_is_extagged_to_true("ARG1")
                self.sent_extags[
                final_loc: final_loc + len(self.arg1_pair[1])] = \
                    ['ARG1'] * len(self.arg1_pair[1])
            else:
                assert False

    def set_extags_of_rel_if_repeated(self):
        arg1_is_extagged = self.base_extag_is_assigned["ARG1"]
        arg2_is_extagged = self.base_extag_is_assigned["ARG2"]
        rel_is_extagged = self.base_extag_is_assigned["REL"]

        if arg1_is_extagged and arg2_is_extagged and \
                rel_is_extagged == "" and \
                len(self.rel_pair[1]) > 0:
            rt = None
            if count_sub_reps(self.rel_pair[1], self.sent_pair[1]) > 1:
                rt = self.rel_pair[1]
            elif self.rel_pair[1][0] == '[is]' and \
                    count_sub_reps(self.rel_pair[1][1:],
                                   self.sent_pair[1]) > 1:
                rt = self.rel_pair[1][1:]
            elif self.rel_pair[1][0] == '[is]' and \
                    self.rel_pair[1][-1].startswith('[') and \
                    count_sub_reps(self.rel_pair[1][1:-1],
                                   self.sent_pair[1]) > 1:
                rt = self.rel_pair[1][1:-1]

            if rt:
                starting_locs = [j for j in range(len(self.sent_pair[1]))
                                 if sub_exists(rt, self.sent_pair[1], j)]
                assert len(starting_locs) > 1

                min_dist = int(1e8)
                if 'ARG1' in self.sent_extags and \
                        (self.arg2_pair[0] == "" or
                         'ARG2' in self.sent_extags):
                    arg1_loc = self.sent_extags.index('ARG1')
                    if self.arg2_pair[0] == "":
                        final_loc = -1
                        for loc in starting_locs:
                            dist = abs(arg1_loc - loc)
                            if dist < min_dist:
                                min_dist = dist
                                final_loc = loc

                        assert rt == \
                               self.sent_pair[1][
                               final_loc: final_loc + len(rt)]
                        self.set_is_extagged_to_true("REL")
                        self.sent_extags[final_loc: final_loc + len(rt)] = \
                            ['REL'] * len(rt)

                    else:
                        arg2_loc = self.sent_extags.index('ARG2')
                        final_loc = -1
                        for loc in starting_locs:
                            dist = abs(arg1_loc - loc) + abs(arg2_loc - loc)
                            if dist < min_dist:
                                min_dist = dist
                                final_loc = loc

                        assert rt == \
                               self.sent_pair[1][
                               final_loc: final_loc + len(rt)]
                        self.set_is_extagged_to_true('REL')
                        self.sent_extags[final_loc: final_loc + len(rt)] = \
                            ['REL'] * len(rt)

    def set_extags_of_loc_or_time(self, arg_name):
        if arg_name == "time":
            pair = self.time_arg_pair
        elif arg_name == "loc":
            pair = self.loc_arg_pair
        else:
            assert False
        matches = get_matches(pair[1], self.sent_pair[1])
        self.set_extags_of_2_matches(matches, arg_name.upper())

    def set_extags_of_all(self):
        self.set_extags_of_arg2()
        self.set_extags_of_arg1_or_rel("arg1")
        self.set_extags_of_arg1_or_rel("rel")
        self.set_extags_of_IS_OF_FROM()
        self.set_extags_of_arg1_if_repeated()
        self.set_extags_of_rel_if_repeated()
        self.set_extags_of_loc_or_time("loc")
        self.set_extags_of_loc_or_time("time")

    def is_in_list(self, ex_list):
        str = ' '.join(self.args) + ' ' + self.rel_pair[1]
        for ex in ex_list:
            if str == ' '.join(ex.args) + ' ' + ex.rel:
                return True
        return False


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
