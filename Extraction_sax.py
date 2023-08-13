from sax_globals import *
from sax_utils import *
from extraction_utils import *


# important
# carb has its own Extraction class at
# carb_subset.oie_readers.extraction
# call ours Extraction_sax
#sax = sentence ax


# extag = extraction tag
# pred=predicate same as rel=relation
class Extraction_sax():
    def __init__(self,
                 ex_sent,  # extracted sentence, without
                 # unused tokens but may have [is], [of], [from] tokens
                 arg1="",
                 rel="",
                 arg2="",
                 confidence=None):
        """
        formerly data_processing.py


        assume only one: arg1, rel, arg2, time_arg, loc_arg
        assume args list is empty
        """

        self.confidence = confidence
        sent = ex_sent + UNUSED_TOKENS_STR
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

    def get_str(self):
        li = [self.arg1_pair[1],
              self.rel_pair[1],
              self.arg2_pair[1],
              self.loc_arg_pair[1],
              self.time_arg_pair[1]]
        li = [x for x in li if x]
        return " ".join(li)

    def set_is_extagged_flag_to_true(self, extag_name):
        assert extag_name in self.base_extags
        self.base_extag_is_assigned[extag_name] = True

    def set_extags_of_2_matches(self, matches, extag_name):
        assert extag_name in self.base_extags
        assert has_2_matches(matches)
        m0 = matches[0]
        self.sent_extags[m0.b: m0.b + m0.size] = [extag_name] * m0.size
        self.set_is_extagged_flag_to_true(extag_name)

    def set_extags_of_gt_2_matches(self, matches, extag_name):
        assert extag_name in self.base_extags
        assert has_gt_2_matches(matches)
        self.set_is_extagged_flag_to_true(extag_name)
        for m in matches:
            self.sent_extags[m.b: m.b + m.size] = \
                [extag_name] * m.size

    def set_extags_of_arg2(self):
        """
        formerly data_processing.label_arg2()


        Returns
        -------

        """

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
        """
        formerly data_processing.label_arg(),


        Parameters
        ----------
        arg_name

        Returns
        -------

        """
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
        """
        formerly data_processing.label_is_of_relations()

        Returns
        -------

        """
        # sent can have implicit "is", "of", "from"
        # inserted in by hand and indicated by "[is]", "[of]" , "[from]"
        # Note that this is different from explicit "is", "of", "from"
        rel_is_extagged = self.base_extag_is_assigned["rel"]
        if (not rel_is_extagged) and len(self.rel_pair[1]) > 0:
            # IS
            if self.rel_pair[0] == '[is]':
                self.set_is_extagged_flag_to_true("REL")
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


    def set_extags_of_repeated_arg1(self):
        """
        formerly data_processing.label_multiple_arg1()


        Returns
        -------

        """
        rel_is_extagged = self.base_extag_is_assigned["REL"]
        arg1_is_extagged = self.base_extag_is_assigned["ARG1"]

        if rel_is_extagged and \
                (not arg1_is_extagged) and \
                count_sub_reps(self.arg1_pair[1], self.sent_pair[1]) > 1:
            start_locs = [start_loc for start_loc in
                             range(len(self.sent_pair[1])) if
                             sub_exists(self.arg1_pair[1],
                                        self.sent_pair[1], start_loc)]
            assert len(start_locs) > 1

            if 'REL' in self.sent_extags:
                # li.index(x) gives first occurrence of x
                rel_loc = self.sent_extags.index('REL')

                xlist = start_locs
                cost_fun = lambda x: abs(rel_loc - x)
                loc0, cost0 = \
                    find_xlist_item_that_minimizes_cost_fun(xlist, cost_fun)
                assert self.arg1_pair[1] == self.sent_pair[1][
                       loc0: loc0 + len(self.arg1_pair[1])]
                self.set_is_extagged_flag_to_true("ARG1")
                # only extag the first occurrence of arg1
                self.sent_extags[
                loc0: loc0 + len(self.arg1_pair[1])] = \
                    ['ARG1'] * len(self.arg1_pair[1])
            else: # 'REL" is not in extags
                assert False

    def set_extags_of_repeated_rel(self):
        """
        formerly data_processing.label_multiple_rel()


        Returns
        -------

        """
        arg1_is_extagged = self.base_extag_is_assigned["ARG1"]
        arg2_is_extagged = self.base_extag_is_assigned["ARG2"]
        rel_is_extagged = self.base_extag_is_assigned["REL"]

        if arg1_is_extagged and arg2_is_extagged and \
                (not rel_is_extagged) and \
                len(self.rel_pair[1]) > 0:
            rel_words = []
            if count_sub_reps(self.rel_pair[1], self.sent_pair[1]) > 1:
                rel_words = self.rel_pair[1]
            elif self.rel_pair[1][0] == '[is]' and \
                    count_sub_reps(self.rel_pair[1][1:],
                                   self.sent_pair[1]) > 1:
                rel_words = self.rel_pair[1][1:]
            elif self.rel_pair[1][0] == '[is]' and \
                    self.rel_pair[1][-1].startswith('[') and \
                    count_sub_reps(self.rel_pair[1][1:-1],
                                   self.sent_pair[1]) > 1:
                rel_words = self.rel_pair[1][1:-1]

            if rel_words:
                start_locs =\
                    [start_loc for start_loc in range(len(self.sent_pair[1]))
                        if sub_exists(rel_words, self.sent_pair[1], start_loc)]
                assert len(start_locs) > 1
                arg2_condition = self.arg2_pair[0] == "" or \
                       'ARG2' in self.sent_extags
                if 'ARG1' in self.sent_extags and arg2_condition:
                    arg1_loc = self.sent_extags.index('ARG1')

                    if self.arg2_pair[0] == "":
                        xlist = start_locs
                        cost_fun = lambda x: abs(arg1_loc - x)
                        loc0, cost0 = \
                            find_xlist_item_that_minimizes_cost_fun(xlist,
                                                                    cost_fun)

                        assert rel_words == \
                               self.sent_pair[1][
                               loc0: loc0 + len(rel_words)]
                        self.set_is_extagged_flag_to_true("REL")
                        self.sent_extags[loc0: loc0 + len(rel_words)] = \
                            ['REL'] * len(rel_words)

                    else: # self.arg2_pair[0] non-empty
                        arg2_loc = self.sent_extags.index('ARG2')
                        xlist = start_locs
                        # this cost function has as minimum
                        # abs(arg1_loc - arg2_loc). This minimum is achieved
                        # by any x in the interval [arg1_loc, arg2_loc]
                        cost_fun =\
                            lambda x: abs(arg1_loc - x) + abs(arg2_loc - x)
                        loc0, cost0 = \
                            find_xlist_item_that_minimizes_cost_fun(xlist,
                                                                    cost_fun)

                        assert rel_words == \
                               self.sent_pair[1][
                               loc0: loc0 + len(rel_words)]
                        self.set_is_extagged_flag_to_true('REL')
                        self.sent_extags[loc0: loc0 + len(rel_words)] = \
                            ['REL'] * len(rel_words)

    def set_extags_of_loc_or_time(self, arg_name):
        """
        formerly data_processing.label_time(),
        formerly data_processing.label_loc()




        Parameters
        ----------
        arg_name

        Returns
        -------

        """
        if arg_name == "time":
            pair = self.time_arg_pair
        elif arg_name == "loc":
            pair = self.loc_arg_pair
        else:
            assert False
        matches = get_matches(pair[1], self.sent_pair[1])
        if has_2_matches(matches):
            self.set_extags_of_2_matches(matches, arg_name.upper())

    def set_extags_of_all(self):
        self.set_extags_of_arg2()
        self.set_extags_of_arg1_or_rel("arg1")
        self.set_extags_of_arg1_or_rel("rel")
        self.set_extags_of_IS_OF_FROM()
        self.set_extags_of_repeated_arg1()
        self.set_extags_of_repeated_rel()
        self.set_extags_of_loc_or_time("loc")
        self.set_extags_of_loc_or_time("time")
