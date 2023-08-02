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
                 sent="",  # original sentence, before extractions.
                 arg1="",
                 rel="",
                 arg2="",
                 confidence=None):
        """
        assume only one: arg1, rel, arg2, time_arg, loc_arg
        assume args list is empty
        """

        self.confidence = confidence

        self.sent_pair = (sent, get_words(sent))
        sent_plus = sent + UNUSED_TOKENS_STR
        self.sent_plus_pair = (sent_plus, get_words(sent_plus))
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

    def set_extags_of_arg2(self): # formerly label_arg2()

        li_2lt= self.arg2_pair[1] + self.loc_arg_pair[1] +\
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
                if count_subs(li, self.sent_plus_pair[1]) == 1:
                    matches = get_matches(li, self.sent_plus_pair[1])
                    self.set_extags_of_2_matches(matches, "ARG2")
                    return
        else: # len(self.arg2_pair[1]) == 0
            for li in without_2_lists:
                if count_subs(li, self.sent_plus_pair[1]) == 1:
                    matches = get_matches(li, self.sent_plus_pair[1])
                    self.set_extags_of_2_matches(matches, "ARG2")
                    return
        # if everything else fails, still
        # set this flag true
        self.arg2_extagged = True


    def set_extags_of_arg1_or_rel(self, arg_name):
        if arg_name == "arg1":
            arg_words = getattr(self, "arg1_words")
        elif arg_name == "rel":
            arg_words = getattr(self, "rel_words")
        else:
            assert False
        if count_subs(arg_words, self.sent_plus_pair[1]) == 1:
            matches = get_matches(arg_words, self.sent_plus_pair[1])
            assert has_2_matches(matches)
            self.set_is_extagged_to_true(arg_name.upper())
            m0 = matches[0]
            self.sent_extags[m0.b: m0.b + m0.size] = \
                [arg_name.upper()] * m0.size


        elif count_subs(arg_words, self.sent_plus_pair[1]) == 0:
            matches = get_matches(arg_words, self.sent_plus_pair[1])
            if has_gt_2_matches(matches):
                self.set_is_extagged_to_true(arg_name.upper())
                for m in matches:
                    self.sent_extags[m.b: m.b + m.size] = \
                        [arg_name.upper()] * m.size

    def set_extags_of_is_of_from(self):

        rel_is_extagged = self.base_extag_is_assigned["rel"]
        if rel_is_extagged == "" and len(self.rel_pair[1]) > 0:
            if self.rel_pair[0] == '[is]':
                self.set_is_extagged_to_true("REL")
                assert self.sent_plus_pair[1][-3] == '[unused1]'
                self.sent_extags[-3] = 'REL'

            elif self.rel_pair[1][0] == '[is]' and \
                    self.rel_pair[1][-1] == '[of]':
                if len(self.rel_pair[1]) > 2 and \
                        count_subs(self.rel_pair[1][1:-1],
                                   self.sent_plus_pair[1]) \
                        == 1:
                    matches = get_matches(self.rel_pair[1][1:-1],
                                          self.sent_plus_pair[1])
                    self.set_extags_of_2_matches(matches, "REL")
                    rel_is_extagged = True
                    assert self.sent_plus_pair[1][-2] == '[unused2]'
                    self.sent_extags[-2] = 'REL'

                elif len(self.rel_pair[1]) > 2 and \
                        count_subs(self.rel_pair[1][1:-1],
                                   self.sent_plus_pair[1]) \
                        == 0:
                    matches = get_matches(self.rel_pair[1][1:-1],
                                          self.sent_plus_pair[1])
                    if has_gt_2_matches(matches):
                        self.set_is_extagged_to_true("REL")
                        for m in matches:
                            self.sent_extags[m.b: m.b + m.size] = \
                                ["REL"] * m.size
                        assert self.sent_plus_pair[1][-2] == '[unused2]'
                        self.sent_extags[-2] = 'REL'

            elif self.rel_pair[1][0] == '[is]' and \
                    self.rel_pair[1][-1] == '[from]':
                if len(self.rel_pair[1]) > 2 and \
                        count_subs(self.rel_pair[1][1:-1],
                                   self.sent_plus_pair[1]) \
                        == 1:
                    matches = get_matches(self.rel_pair[1][1:-1],
                                          self.sent_plus_pair[1])
                    self.set_extags_of_2_matches(matches, "REL")
                    rel_is_extagged = True
                    assert self.sent_plus_pair[1][-1] == '[unused3]'
                    self.sent_extags[-1] = 'REL'

                elif len(self.rel_pair[1]) > 2 and \
                        count_subs(self.rel_pair[1][1:-1],
                                   self.sent_plus_pair[1]) \
                        == 0:
                    matches = get_matches(self.rel_pair[1][1:-1],
                                          self.sent_plus_pair[1])
                    if has_gt_2_matches(matches):
                        rel_is_extagged = True
                        for m in matches:
                            self.sent_extags[m.b: m.b + m.size] = \
                                ["REL"] * m.size
                        assert self.sent_plus_pair[1][-1] == '[unused3]'
                        self.sent_extags[-1] = 'REL'

            elif self.rel_pair[1][0] == '[is]' and len(self.rel_pair[1]) > 1:
                assert self.rel_pair[1][-1].startswith('[') == ""
                if count_subs(self.rel_pair[1][1:],
                              self.sent_plus_pair[1]) == 1:
                    matches = get_matches(
                        self.rel_pair[1][1:], self.sent_plus_pair[1])
                    self.set_extags_of_2_matches(matches, "REL")
                    self.set_is_extagged_to_true("REL")
                    assert self.sent_plus_pair[1][-3] == '[unused1]'
                    self.sent_extags[-3] = 'REL'

                elif len(self.rel_pair[1]) > 2 and \
                        count_subs(self.rel_pair[1][1:],
                                   self.sent_plus_pair[1]) == 0:
                    matches = get_matches(
                        self.rel_pair[1][1:-1], self.sent_plus_pair[1])
                    if has_gt_2_matches(matches):
                        self.set_is_extagged_to_true("REL")
                        for m in matches:
                            self.sent_extags[m.b: m.b + m.size] = \
                                ["REL"] * m.size
                        assert self.sent_plus_pair[1][-3] == '[unused1]'
                        self.sent_extags[-3] = 'REL'

    def set_extags_of_multiple_arg1(self):
        rel_is_extagged = self.base_extag_is_assigned["REL"]
        arg1_is_extagged = self.base_extag_is_assigned["ARG1"]

        if rel_is_extagged and \
                arg1_is_extagged == "" and \
                count_subs(self.arg1_pair[1], self.sent_plus_pair[1]) > 1:
            starting_locs = [j for j in
                             range(len(self.sent_plus_pair[1])) if
                             sub_exists(self.arg1_pair[1],
                                        self.sent_plus_pair[1], j)]
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

                assert self.arg1_pair[1] == self.sent_plus_pair[1][
                                            final_loc: final_loc + len(
                                                self.arg1_pair[1])]
                self.set_is_extagged_to_true("ARG1")
                self.sent_extags[
                final_loc: final_loc + len(self.arg1_pair[1])] = \
                    ['ARG1'] * len(self.arg1_pair[1])
            else:
                assert False

    def set_extags_of_multiple_rel(self):
        arg1_is_extagged = self.base_extag_is_assigned["ARG1"]
        arg2_is_extagged = self.base_extag_is_assigned["ARG2"]
        rel_is_extagged = self.base_extag_is_assigned["REL"]

        if arg1_is_extagged and arg2_is_extagged and \
                rel_is_extagged == "" and \
                len(self.rel_pair[1]) > 0:
            rt = None
            if count_subs(self.rel_pair[1], self.sent_plus_pair[1]) > 1:
                rt = self.rel_pair[1]
            elif self.rel_pair[1][0] == '[is]' and \
                    count_subs(self.rel_pair[1][1:],
                               self.sent_plus_pair[1]) > 1:
                rt = self.rel_pair[1][1:]
            elif self.rel_pair[1][0] == '[is]' and \
                    self.rel_pair[1][-1].startswith('[') and \
                    count_subs(self.rel_pair[1][1:-1],
                               self.sent_plus_pair[1]) > 1:
                rt = self.rel_pair[1][1:-1]

            if rt:
                starting_locs = [j for j in range(len(self.sent_plus_pair[1]))
                                 if sub_exists(rt, self.sent_plus_pair[1], j)]
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
                               self.sent_plus_pair[1][
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
                               self.sent_plus_pair[1][
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
        self.set_extags_of_is_of_from()
        self.set_extags_of_multiple_arg1()
        self.set_extags_of_multiple_rel()
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


def count_subs(sub, full):  # formerly seq_in_seq
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
    return difflib.SequenceMatcher(None, list0, list1).\
        get_matching_blocks()
