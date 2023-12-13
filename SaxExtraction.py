from Params import *
from utils_gen import *
from utils_matching import *
import numpy as np
from carb_subset.oie_readers.extraction import Extraction


class SaxExtraction:
    """
    similar to Openie6.data_processing.py
    
    Important: Carb has its own extraction class called `Extraction` at
    carb_subset.oie_readers.extraction. To distinguish theirs from ours,
    we call ours `SaxExtraction`. sax = Sentence Ax.

    ex= extraction
    rel=relation

    strings: `arg1, rel, arg2, time_arg, loc_arg`
    `args` is a list[str] (assume empty).
    arg1 is the subject of a sentence.
    rel is usually a verb and adverbs.
    If there is any time_arg or loc_arg string, add it to arg2.

    orig_sent = osent = original (before splitting) sentence
    L = Long
    osentL = osent + UNUSED_TOKENS_STR
    osent2 = osent or osentL

    We allow the following addition to rel, of symbols that don't appear in 
    osent:

    unused_case=1: `[is]` at the beginning of rel
    unused_case=2: `[is]` at beginning and `[of]` at end of rel
    unused_case=3: `[is]` at beginning and `[from]` at end of rel


    Attributes
    ----------
    _arg1: str
    _arg2: str
    _loc_arg: str
    _orig_sentL: str
    _rel: str
    _time_arg: str
    arg1: str
    arg1_words: list[str]
    arg2: str
    arg2_is_assigned: bool
    arg2_words: list[str]
    confidence: float
    extags: list[str]
    extags_are_set: bool
    is_assigned_d: dict[str, bool]
    loc_arg: str
    loc_arg_words: str
    orig_sentL: str
    rel_words: list[str]
    time_arg_words: list[str]

    """

    def __init__(self,
                 orig_sentL,
                 arg1="",
                 rel="",
                 arg2="",
                 confidence=None):
        """

        Parameters
        ----------
        orig_sentL: str
        arg1: str
        rel: str
        arg2: str
        confidence: float
        """

        self.confidence = confidence

        self._orig_sentL = orig_sentL
        self.orig_sentL_words = get_words(orig_sentL)

        self._arg1 = arg1
        self.arg1_words = get_words(arg1)

        self._rel = rel
        self.rel_words = get_words(rel)

        self._arg2 = arg2
        self.arg2_words = get_words(arg2)

        self._time_arg = ""
        self.time_arg_words = []

        self._loc_arg = ""
        self.loc_arg_words = []

        self.arg2_is_assigned = False

        self.extags = ["NONE"] * len(self.orig_sentL_words)
        self.is_assigned_d = {extag_name: False
                              for extag_name in BASE_EXTAGS}
        self.extags_are_set = False

    @property
    def orig_sentL(self):
        """

        Returns
        -------
        str
        """
        return self._orig_sentL

    @orig_sentL.setter
    def orig_sentL(self, value):
        """

        Parameters
        ----------
        value: str

        Returns
        -------
        None

        """
        self.orig_sentL = value
        self.orig_sentL_words = get_words(value)

    @property
    def arg1(self):
        """

        Returns
        -------
        str

        """
        return self._arg1

    @arg1.setter
    def arg1(self, value):
        """

        Parameters
        ----------
        value: str

        Returns
        -------
        None

        """
        self.arg1 = value
        self.arg1_words = get_words(value)

    @property
    def rel(self):
        """

        Returns
        -------
        str

        """
        return self._rel

    @rel.setter
    def rel(self, value):
        """

        Parameters
        ----------
        value: str

        Returns
        -------
        None

        """
        self.rel = value
        self.rel_words = get_words(value)

    @property
    def arg2(self):
        """

        Returns
        -------
        str

        """
        return self._arg2

    @arg2.setter
    def arg2(self, value):
        """

        Parameters
        ----------
        value: str

        Returns
        -------
        None

        """
        self.arg2 = value
        self.arg2_words = get_words(value)

    @property
    def time_arg(self):
        """

        Returns
        -------
        str

        """
        return self._time_arg

    @time_arg.setter
    def time_arg(self, value):
        """

        Parameters
        ----------
        value: str

        Returns
        -------
        None

        """
        self.time_arg = value
        self.time_arg_words = get_words(value)

    @property
    def loc_arg(self):
        """

        Returns
        -------
        str

        """
        return self._loc_arg

    @loc_arg.setter
    def loc_arg(self, value):
        """

        Parameters
        ----------
        value: str

        Returns
        -------
        None

        """
        self.loc_arg = value
        self.loc_arg_words = get_words(value)

    def __eq__(self, other_ex):
        """
        This method defines `ex1=ex2` for 2 SaxExtractions `ex1` and `ex2`.

        Parameters
        ----------
        other_ex: SaxExtraction

        Returns
        -------
        bool

        """
        return self.get_simple_sent() == other_ex.get_simple_sent()

    def get_simple_sent(self):
        """
        This method returns a simple sentence concocted with arg1, rel,
        arg2, loc_arg and time_arg.

        Returns
        -------
        str

        """
        li = [self.arg1,
              self.rel,
              self.arg2,
              self.loc_arg,
              self.time_arg]
        li = [x for x in li if x]
        str0 = " ".join(li)
        # print("hjki", str0)
        return str0

    def set_the_is_assigned_flag_to_true(self, extag_name):
        """
        All this method does is to set self.is_assigned_d[extag_name] equal
        to True.

        Parameters
        ----------
        extag_name: str

        Returns
        -------
        None

        """
        assert extag_name in BASE_EXTAGS
        self.is_assigned_d[extag_name] = True

    def set_extags_of_2_matches(self, matches, extag_name):
        """
        This method is based on the method
        sax_extraction_utils.has_2_matches().

        The two methods set_extags_of_2_matches() and
        set_extags_of_gt_2_matches() are building blocks that are called
        internally by the other methods whose names start with
        `set_extags_to_*`.

        Parameters
        ----------
        matches: list[Match]
        extag_name: str

        Returns
        -------
        None

        """
        assert extag_name in BASE_EXTAGS
        if has_2_matches(matches):
            m0 = matches[0]
            self.extags[m0.b: m0.b + m0.size] = [extag_name] * m0.size
            self.set_the_is_assigned_flag_to_true(extag_name)

    def set_extags_of_gt_2_matches(self, matches, extag_name):
        """
        This method is based on the method
        sax_extraction_utils.has_gt_2_matches().

        The two methods set_extags_of_2_matches() and 
        set_extags_of_gt_2_matches() are building blocks that are called 
        internally by the other methods whose names start with 
        `set_extags_to_*`.

        Parameters
        ----------
        matches: list[Match]
        extag_name: str

        Returns
        -------
        None

        """
        assert extag_name in BASE_EXTAGS
        if has_gt_2_matches(matches):
            self.set_the_is_assigned_flag_to_true(extag_name)
            for m in matches:
                self.extags[m.b: m.b + m.size] = \
                    [extag_name] * m.size

    def set_extags_to_ARG2(self):
        """
        similar to Openie6.data_processing.label_arg2()

        This method's only goal in life is to be called by self.set_extags(
        ). The method returns nothing. It just changes the class attribute
        self.extags. That attribute is initialized when the class is created
        to a list of NONE's. This method changes some of those NONE's to
        to ARG2.

        Setting extags to ARG2 differs from setting them to ARG1 and REL in
        that ARG2 must also include the legacy time_arg and loc_arg.

        Returns
        -------
        None

        """

        li_2lt = \
            self.arg2_words + self.loc_arg_words + self.time_arg_words
        li_2tl = \
            self.arg2_words + self.time_arg_words + self.loc_arg_words
        li_2t = self.arg2_words + self.time_arg_words
        li_2l = self.arg2_words + self.loc_arg_words
        li_2 = self.arg2_words
        with_2_lists = [li_2lt, li_2tl, li_2t, li_2l, li_2]

        li_lt = self.loc_arg_words + self.time_arg_words
        li_tl = self.time_arg_words + self.loc_arg_words
        li_t = self.time_arg_words
        li_l = self.loc_arg_words
        without_2_lists = [li_lt, li_tl, li_t, li_l]

        if len(self.arg2_words) != 0:
            for li in with_2_lists:
                if count_sub_reps(li, self.orig_sentL_words) == 1:
                    matches = get_matches(li, self.orig_sentL_words)
                    self.set_extags_of_2_matches(matches, "ARG2")
                    return
        else:  # len(self.arg2_words) == 0
            for li in without_2_lists:
                if count_sub_reps(li, self.orig_sentL_words) == 1:
                    matches = get_matches(li, self.orig_sentL_words)
                    self.set_extags_of_2_matches(matches, "ARG2")
                    return
        # if everything else fails, still
        # set this flag true
        self.arg2_is_assigned = True

    def set_extags_to_ARG1_or_REL(self, arg_name):
        """
        similar to Openie6.data_processing.label_arg().

        This method's only goal in life is to be called by self.set_extags(
        ). The method returns nothing. It just changes the class attribute
        self.extags. That attribute is initialized when the class is created
        to a list of NONE's. This method changes some of those NONE's to
        ARG1 or REL.

        Two sub cases are considered:

        1. a sub occurs unfractured within full. In this case we call
        set_extags_of_2_matches()

        2. a sub does not occur unfractured within full, but it does occur 
        in fractured form. In this case we call set_extags_of_gt_2_matches()


        Parameters
        ----------
        arg_name: str

        Returns
        -------
        None

        """
        if arg_name == "arg1":
            arg_words = self.arg1_words
        elif arg_name == "rel":
            arg_words = self.rel_words
        else:
            assert False
        if count_sub_reps(arg_words, self.orig_sentL_words) == 1:
            matches = get_matches(arg_words, self.orig_sentL_words)
            self.set_extags_of_2_matches(matches, arg_name.upper())

        elif count_sub_reps(arg_words, self.orig_sentL_words) == 0:
            # sub doesn't fit in one piece into full
            # but still possible it exists in fractured form
            matches = get_matches(arg_words, self.orig_sentL_words)
            if has_gt_2_matches(matches):
                self.set_extags_of_gt_2_matches(matches, arg_name.upper())

    def set_extags_to_REL_for_unused_case(self, unused_case):
        """
        This method's only goal in life is to be called by self.set_extags(
        ). The method returns nothing. It just changes the class attribute
        self.extags. That attribute is initialized when the class is created
        to a list of NONE's. This method changes some of those NONE's to REL
        for one of the 3 possible unused cases.
        
        Since set_extags_to_REL_or_ARG1() also changes extags to REL, 
        how do the 2 methods differ? The 3 unused cases considered by this 
        method include [is] [of] [from] tokens in rel that are not in osent, 
        so these unused cases are always bypassed by 
        set_extags_to_REL_or_ARG1().

        unused_case=1: rel has a [is] as the first word. Set the extag
        corresponding to the [unused1] token to REL. Set extags
        corresponding to rel[1:] to REL.

        unused_case=2: rel has an [is] as the first word and an [of] as the
        last. Set the extag corresponding to the [unused2] token to REL. Set
        extags corresponding to rel[1:-1] to REL.

        unused_case=3: rel has an [is] as the first word and a [from] as the
        last. Set the extag corresponding to the [unused3] token to REL. Set
        extags corresponding to rel[1:-1] to REL.
        
        Two sub cases are considered for each unused_case:

        1. a sub occurs unfractured within full. In this case we call
        set_extags_of_2_matches()

        2. a sub does not occur unfractured within full, but it does occur 
        in fractured form. In this case we call set_extags_of_gt_2_matches()


        Parameters
        ----------
        unused_case: int

        Returns
        -------
        None

        """
        assert unused_case in [1, 2, 3]
        unused_str = f'[unused{unused_case}]'
        # this equals -3, -2, -1 for 1, 2, 3
        unused_loc = -4 + unused_case

        # [1:-1] when self.rel[0] = "[is]" and
        # self.rel[-1] = "of" or "from"

        # [1: len(self.rel)] when self.rel[0] = "[is]" and
        # self.rel[-1] is anything that doesn't start with "["
        if unused_case in [2, 3]:
            # is-rel-to and is-rel-from
            range_end = -1  # same as len(self.rel_words)-1
        elif unused_case == 1:
            # is-rel
            range_end = len(self.rel_words)
        else:
            assert False

        if count_sub_reps(self.rel_words[1:range_end],
                          self.orig_sentL_words) == 1:
            matches = get_matches(
                self.rel_words[1:range_end], self.orig_sentL_words)
            self.set_extags_of_2_matches(matches, "REL")
            assert self.orig_sentL_words[unused_loc] == unused_str
            self.extags[unused_loc] = 'REL'
        elif len(self.rel_words) > 2 and \
                count_sub_reps(self.rel_words[1:range_end],
                               self.orig_sentL_words) == 0:
            matches = get_matches(
                self.rel_words[1:range_end], self.orig_sentL_words)
            # sub doesn't fit in one piece into full
            # but still possible it exists in fractured form
            if has_gt_2_matches(matches):
                self.set_extags_of_gt_2_matches(matches, "REL")
                assert self.orig_sentL_words[unused_loc] == unused_str
                # if sent starts with "[is]" and ends with
                # anything, "[of]", or "[from]"
                # then switch the extag at sent positions of
                # [unused1], [unused2], [unused3]
                # from NONE to REL
                self.extags[unused_loc] = 'REL'

    def set_extags_to_REL_for_all_unused_cases(self):
        """
        similar to Openie6.data_processing.label_is_of_relations()

        This method calls set_extags_to_REL_for_unused_case() for the 3
        unused cases.

        Returns
        -------
        None

        """
        # sent can have implicit "is", "of", "from"
        # inserted in by hand and indicated by "[is]", "[of]" , "[from]"
        # Note that this is different from explicit "is", "of", "from"
        if (not self.is_assigned_d["REL"]) and len(self.rel_words) > 0:
            # IS
            if self.rel == '[is]':
                self.set_the_is_assigned_flag_to_true("REL")
                assert self.orig_sentL_words[-3] == '[unused1]'
                self.extags[-3] = 'REL'
            # IS-OF
            elif self.rel_words[0] == '[is]' and \
                    self.rel_words[-1] == '[of]':
                self.set_extags_to_REL_for_unused_case(2)
            # IS  FROM
            elif self.rel_words[0] == '[is]' and \
                    self.rel_words[-1] == '[from]':
                self.set_extags_to_REL_for_unused_case(3)
            # IS
            elif self.rel_words[0] == '[is]' and \
                    len(self.rel_words) > 1:
                assert self.rel_words[-1].startswith('[') == ""
                self.set_extags_to_REL_for_unused_case(1)

    def set_extags_to_ARG1_if_repeated_arg1(self):
        """
        similar to Openie6.data_processing.label_multiple_arg1()

        This method's only goal in life is to be called by self.set_extags( 
        ). The method returns nothing. It just changes the class attribute 
        self.extags. That attribute is initialized when the class is created 
        to a list of NONE's. This method changes some of those NONE's to 
        ARG1, in sub case where the sub arg1 appears more than once in full. 
        In that sub case, the method chooses just one of the occurrences of 
        arg1, the one that is closest to rel. The method then changes to 
        ARG1 the extags corresponding to that occurrence of arg1.


        Returns
        -------
        None

        """

        if self.is_assigned_d["REL"] and \
                (not self.is_assigned_d["ARG1"]) and \
                count_sub_reps(self.arg1_words, self.orig_sentL_words) > 1:
            start_locs = [start_loc for start_loc in
                          range(len(self.orig_sentL_words)) if
                          sub_exists(self.arg1_words,
                                     self.orig_sentL_words, start_loc)]
            # assert len(start_locs) > 1

            if 'REL' in self.extags:
                # li.index(x) gives first occurrence of x
                rel_loc = self.extags.index('REL')

                xlist = start_locs

                def cost_fun(x):
                    return abs(rel_loc - x)

                loc0, cost0 = \
                    find_xlist_item_that_minimizes_cost_fun(xlist, cost_fun)
                assert self.arg1_words == \
                    self.orig_sentL_words[loc0: loc0 + len(self.arg1_words)]
                self.set_the_is_assigned_flag_to_true("ARG1")
                # only extag the first occurrence of arg1
                self.extags[loc0: loc0 + len(self.arg1_words)] = \
                    ['ARG1'] * len(self.arg1_words)
            else:  # "REL" is not in extags
                assert False

    def set_extags_to_REL_if_repeated_rel(self):
        """
        similar to Openie6.data_processing.label_multiple_rel()
        
        This method's only goal in life is to be called by self.set_extags(
        ). The method returns nothing. It just changes the class attribute
        self.extags. That attribute is initialized when the class is created
        to a list of NONE's. This method changes some of those NONE's to
        REL, in the sub case where the sub rel appears more than once in
        full. In that sub case, the method chooses just one of the
        occurrences of rel, the one that is:

         1. closest to arg1 if arg2 is already taken care
        of (i.e., arg2 is missing or ARG2 is already substituted).

         2. closest to arg2 if arg1 is already taken care
        of (i.e., arg2 is missing or ARG2 is already substituted).

        The method then changes to REL the extags corresponding to that
        occurrence of rel.

        Returns
        -------
        None

        """

        if self.is_assigned_d["ARG1"] and self.is_assigned_d["ARG2"] and \
                (not self.is_assigned_d["REL"]) and \
                len(self.rel_words) > 0:
            rel_words = []
            if count_sub_reps(self.rel_words, self.orig_sentL_words) > 1:
                rel_words = self.rel_words
            elif self.rel_words[0] == '[is]' and \
                    count_sub_reps(self.rel_words[1:],
                                   self.orig_sentL_words) > 1:
                rel_words = self.rel_words[1:]
            elif self.rel_words[0] == '[is]' and \
                    self.rel_words[-1].startswith('[') and \
                    count_sub_reps(self.rel_words[1:-1],
                                   self.orig_sentL_words) > 1:
                rel_words = self.rel_words[1:-1]

            if rel_words:
                start_locs = \
                    [start_loc for start_loc in
                     range(len(self.orig_sentL_words)) if
                     sub_exists(rel_words, self.orig_sentL_words, start_loc)]
                # assert len(start_locs) > 1
                arg2_condition = self.arg2 == "" or 'ARG2' in self.extags
                if 'ARG1' in self.extags and arg2_condition:
                    arg1_loc = self.extags.index('ARG1')

                    if self.arg2 == "":
                        xlist = start_locs

                        def cost_fun(x):
                            return abs(arg1_loc - x)

                        loc0, cost0 = \
                            find_xlist_item_that_minimizes_cost_fun(
                                xlist, cost_fun)

                        assert rel_words == \
                            self.orig_sentL_words[loc0: loc0 + len(rel_words)]
                        self.set_the_is_assigned_flag_to_true("REL")
                        self.extags[loc0: loc0 + len(rel_words)] = \
                            ['REL'] * len(rel_words)

                    else:  # self.arg2 non-empty
                        arg2_loc = self.extags.index('ARG2')
                        xlist = start_locs

                        # this cost function has as minimum
                        # abs(arg1_loc - arg2_loc). This minimum is achieved
                        # by any x in the interval [arg1_loc, arg2_loc]
                        def cost_fun(x):
                            return abs(arg1_loc - x) + abs(arg2_loc - x)

                        loc0, cost0 = \
                            find_xlist_item_that_minimizes_cost_fun(xlist,
                                                                    cost_fun)

                        # assert rel_words == \
                        #        self.orig_sentL_words[
                        #        loc0: loc0 + len(rel_words)]
                        self.set_the_is_assigned_flag_to_true('REL')
                        self.extags[loc0: loc0 + len(rel_words)] = \
                            ['REL'] * len(rel_words)

    def set_extags_to_ARG2_of_loc_and_time(self, arg_name):
        """
        similar to Openie6.data_processing.label_time(),
        similar to Openie6.data_processing.label_loc()

        This method's only goal in life is to be called by self.set_extags(
        ). The method returns nothing. It just changes the class attribute
        self.extags. That attribute is initialized when the class is created
        to a list of NONE's. This method changes some of those NONE's to
        ARG2, in the sub case where a sub loc_arg or time_arg occurs
        unfractured within full. In this case we call set_extags_of_2_matches()


        Parameters
        ----------
        arg_name: str

        Returns
        -------
        None

        """
        if arg_name == "time":
            matches = get_matches(self.time_arg_words,
                                  self.orig_sentL_words)
        elif arg_name == "loc":
            matches = get_matches(self.loc_arg_words,
                                  self.orig_sentL_words)
        else:
            assert False
        if has_2_matches(matches):
            # self.set_extags_of_2_matches(matches, arg_name.upper())
            self.set_extags_of_2_matches(matches, "ARG2")

    def set_extags(self):
        """
        This method calls all the other methods in this class whose names
        begin with `set_extags_to_*`. The method returns nothing.

        When this class is first constructed, it initializes the class
        attribute `self.extags` to a list of NONE's:

        self.extags = ["NONE"] * len(self.orig_sentL_words)

        This method gradually replaces some of the NONE's in that list by
        other extags like "ARG1", "REL" and "ARG2".

        Returns
        -------
        None

        """

        self.set_extags_to_ARG2()
        self.set_extags_to_ARG1_or_REL("arg1")
        self.set_extags_to_ARG1_or_REL("rel")
        self.set_extags_to_REL_for_all_unused_cases()
        self.set_extags_to_ARG1_if_repeated_arg1()
        self.set_extags_to_REL_if_repeated_rel()
        self.set_extags_to_ARG2_of_loc_and_time("loc")
        self.set_extags_to_ARG2_of_loc_and_time("time")

        self.extags_are_set = True

    @staticmethod
    def convert_to_sax_ex(carb_ex):
        """
        This method takes as input a Carb Extraction `carb_ex` and returns
        the equivalent SaxExtraction.
    
        Parameters
        ----------
        carb_ex: carb_subset.oie_readers.extraction.Extraction
    
        Returns
        -------
        SaxExtraction
    
        """
        arg1 = carb_ex.args[0]

        arg2 = " ".join(carb_ex.args[1:])

        # print("lklder", carb_ex.sent)
        # print("arg1", carb_ex.sent)
        # print("arg1", arg1)
        # print("rel", carb_ex.pred)
        # print("arg2", arg2)

        return SaxExtraction(orig_sentL=carb_ex.sent,
                             arg1=arg1,
                             rel=carb_ex.pred,
                             arg2=arg2,
                             confidence=carb_ex.confidence)

    @staticmethod
    def convert_to_carb_ex(sax_ex):
        """
        used this openie6.model.write_to_files to write this method.

        This method takes as input a SaxExtraction `sax_ex` and returns the
        equivalent Carb Extraction.

        Parameters
        ----------
        sax_ex: SaxExtraction

        Returns
        -------
        carb_subset.oie_readers.extraction.Extraction

        """
        carb_ex = Extraction(pred=sax_ex.rel,
                             head_pred_index=None,
                             sent=sax_ex.orig_sentL,
                             confidence=sax_ex.confidence,
                             index=1)
        # confidence only used for ordering exs, doesn't affect scores
        carb_ex.addArg(sax_ex.arg1)
        carb_ex.addArg(sax_ex.arg2)
        return carb_ex

    @staticmethod
    def get_carb_osent2_to_exs(sax_osent2_to_exs):
        """
        This method takes as input a dictionary `sax_osent2_to_exs` mapping
        a sentence osent2 to a list of sax extractions. It returns the same
        dictionary with the sax extractions replaced by carb extractions.

        Parameters
        ----------
        sax_osent2_to_exs: dict[str, list[SaxExtraction]]

        Returns
        -------
        dict[str, list[Extraction]]

        """
        carb_osent2_to_exs = {}
        for osent2, sax_exs in sax_osent2_to_exs.items():
            carb_osent2_to_exs[osent2] = \
                [SaxExtraction.convert_to_carb_ex(sax_ex)
                 for sax_ex in sax_exs]
        return carb_osent2_to_exs

    @staticmethod
    def get_sax_osent2_to_exs(carb_osent2_to_exs):
        """
        This method takes as input a dictionary `carb_osent2_to_exs` mapping
        a sentence osent2 to a list of carb extractions. It returns the same
        dictionary with the carb extractions replaced by sax extractions.


        Parameters
        ----------
        carb_osent2_to_exs: dict[str, list[Extraction]]

        Returns
        -------
        dict[str, list[SaxExtraction]]

        """
        sax_osent2_to_exs = {}
        for osent2, carb_exs in carb_osent2_to_exs.items():
            sax_osent2_to_exs[osent2] = \
                [SaxExtraction.convert_to_sax_ex(carb_ex)
                 for carb_ex in carb_exs]
        return sax_osent2_to_exs

    @staticmethod
    def get_ex_from_ilabels(ex_ilabels, orig_sentL, confidence):
        """
        similar to Openie6.model.process_extraction()

        This method returns a SaxExtraction which it builds from `ex_ilabels`,
        `orig_sentL`, `confidence`.

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
        ex_ilabels: list[int]
            a list of ilabels (ints in range(0:6))
        orig_sentL: str
            original sentence Long (i.e., ending with UNUSED_TOKEN_STR)
        confidence: float

        Returns
        -------
        SaxExtraction

        """
        # ex_ilabels = ex_ilabels.to_list()  # change from torch tensor to list

        rel_words = []
        arg1_words = []
        arg2_words = []
        # loc_time_words=[]
        # args_words = []
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
                arg1_words.append(word)
            elif ex_ilabels[i] == 2:  # REL
                rel_words.append(word)
            elif ex_ilabels[i] in [3, 4, 5]:  # ARG2
                arg2_words.append(word)
            else:
                assert False, ex_ilabels[i]

        rel_clause = ' '.join(rel_words)
        if rel_case == 1:
            rel_clause = 'is ' + rel_clause
        elif rel_case == 2:
            rel_clause = 'is ' + rel_clause + ' of'
        elif rel_case == 3:
            rel_clause = 'is ' + rel_clause + ' from'

        arg1 = ' '.join(arg1_words)
        arg2 = ' '.join(arg2_words)

        # args = ' '.join(args_words)
        # loc_time = ' '.join(l_loc_time)
        # if not self.params.d["no_lt"]: # no_lt = no loc time
        #     arg2 = arg2 + ' ' + loc_time + ' ' + args

        extraction = SaxExtraction(orig_sentL,
                                   arg1,
                                   rel_clause,
                                   arg2,
                                   confidence=confidence)

        return extraction


if __name__ == "__main__":

    def main():
        from AllenTool import AllenTool
        in_fp = "tests/small_allen.tsv"
        at = AllenTool(in_fp)
        for osentL, sax_exs in at.osentL_to_exs.items():
            carb_exs = [SaxExtraction.convert_to_carb_ex(sax_ex) for sax_ex in
                        sax_exs]
            new_sax_exs = [SaxExtraction.convert_to_sax_ex(carb_ex)
                           for carb_ex in carb_exs]
            for k, sax_ex in enumerate(sax_exs):
                new_sax_ex = new_sax_exs[k]
                l_old = [sax_ex.arg1, sax_ex.rel, sax_ex.arg2,
                         sax_ex.confidence]
                l_new = [new_sax_ex.arg1, new_sax_ex.rel, new_sax_ex.arg2,
                         new_sax_ex.confidence]
                for i, old in enumerate(l_old):
                    new = l_new[i]
                    assert old == new


    main()
