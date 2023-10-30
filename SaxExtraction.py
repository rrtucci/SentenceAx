from Params import *
from sax_utils import *
from sax_extraction_utils import *
import numpy as np


class SaxExtraction:
    """
    similar to Openie6.data_processing.py
    
    Important: carb has its own extraction class called `Extraction` at
    carb_subset.oie_readers.extraction. We call ours `SaxExtraction`. sax =
    sentence ax.
    extag = extraction tag
    pred=predicate same as rel=relation
    
    
    assume only one of: `arg1, rel, arg2, time_arg, loc_arg`
    assume `args` list is empty
    
    `orig_sent` will represent the original sentence. commas and periods 
    will be assumed to be isolated (i.e., with blank space before and after)
    
    orig_sentL = orig_sent + UNUSED_TOKEN_STR is the long version of orig_sent
    
    `ex_sent` will represent an extracted sentence (simple sentence). It
    does not contain unused tokens but may contain "is", "of", "from". which 
    do not appear in orig_sent.
    ex_sent = arg1 + rel + arg2

    Attributes
    ----------
    _arg1_words: list[str]
    _arg2_words: list[str]
    _loc_arg_words: list[str]
    _orig_sentL_words: list[str]
    _rel_words: list[str]
    _time_arg_words: list[str]
    arg1: str
    arg2: str
    arg2_is_extagged: bool
    confi: float
    extags: list[str]
    extags_are_set: bool
    is_assigned_d: dict[str, bool]
    loc_arg: str
    orig_sentL: str
    rel: str
    time_arg: str

    """

    def __init__(self,
                 orig_sentL,
                 arg1="",
                 rel="",
                 arg2="",
                 confi=None):
        """

        Parameters
        ----------
        orig_sentL: str
        arg1: str
        rel: str
        arg2: str
        confi: float
        """

        self.confi = confi

        self.orig_sentL = orig_sentL
        self._orig_sentL_words = None

        self.arg1 = arg1
        self._arg1_words = None

        self.rel = rel
        self._rel_words = None

        self.arg2 = arg2
        self._arg2_words = None

        self.time_arg = ""
        self._time_arg_words = None

        self.loc_arg = ""
        self._loc_arg_words = None

        self.arg2_is_extagged = False

        self.extags = ["NONE"] * len(self.orig_sentL_words)
        self.is_assigned_d = {extag_name: False
                              for extag_name in BASE_EXTAGS}
        self.extags_are_set = False

    @property
    def orig_sentL_words(self):
        """

        Returns
        -------
        list[str]

        """
        if not self._orig_sentL_words:
            return get_words(self.orig_sentL)
        return self._orig_sentL_words

    @orig_sentL_words.setter
    def orig_sentL_words(self, value):
        """

        Parameters
        ----------
        value: list[str]

        Returns
        -------
        list[str]

        """
        self._orig_sentL_words = value

    @property
    def arg1_words(self):
        """

        Returns
        -------
        list[str]

        """
        if not self._arg1_words:
            return get_words(self.arg1)
        return self._arg1_words

    @arg1_words.setter
    def arg1_words(self, value):
        """

        Parameters
        ----------
        value: list[str]

        Returns
        -------
        None


        """
        self._arg1_words = value

    @property
    def rel_words(self):
        """

        Returns
        -------
        list[str]

        """
        if not self._rel_words:
            return get_words(self.rel)
        return self._rel_words

    @rel_words.setter
    def rel_words(self, value):
        """

        Parameters
        ----------
        value: list[str]

        Returns
        -------
        None

        """
        self._rel_words = value

    @property
    def arg2_words(self):
        """

        Returns
        -------
        list[str]

        """
        if not self._arg2_words:
            return get_words(self.arg2)
        return self._arg2_words

    @arg2_words.setter
    def arg2_words(self, value):
        """

        Parameters
        ----------
        value: list[str]

        Returns
        -------
        None

        """
        self._arg2_words = value

    @property
    def time_arg_words(self):
        """

        Returns
        -------
        list[str]

        """
        if not self._time_arg_words:
            return get_words(self.time_arg)
        return self._time_arg_words

    @time_arg_words.setter
    def time_arg_words(self, value):
        """

        Parameters
        ----------
        value: list[str]

        Returns
        -------
        None

        """
        self._time_arg_words = value

    @property
    def loc_arg_words(self):
        """

        Returns
        -------
        list[str]

        """
        if not self._loc_arg_words:
            return get_words(self.loc_arg)
        return self._loc_arg_words

    @loc_arg_words.setter
    def loc_arg_words(self, value):
        """

        Parameters
        ----------
        value: list[str]

        Returns
        -------
        None

        """
        self._loc_arg_words = value

    def __eq__(self, other_ex):
        """

        Parameters
        ----------
        other_ex: SaxExtraction

        Returns
        -------
        bool

        """
        return self.get_simple_sent() == other_ex.get_simple_sent()

    # def is_in(self, l_ex):
    #     """
    #
    #     Parameters
    #     ----------
    #     l_ex: list[SaxExtraction]
    #
    #     Returns
    #     -------
    #     bool
    #
    #     """
    #     for ex in l_ex:
    #         if ex == self:
    #             return True
    #     return False
    #
    # def is_not_in(self, l_ex):
    #     """
    #
    #     Parameters
    #     ----------
    #     l_ex: list[SaxExtraction]
    #
    #     Returns
    #     -------
    #     bool
    #
    #     """
    #     return not self.is_in(l_ex)

    def get_simple_sent(self):
        """

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

    def set_the_is_extagged_flag_to_true(self, extag_name):
        """

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
            self.set_the_is_extagged_flag_to_true(extag_name)

    def set_extags_of_gt_2_matches(self, matches, extag_name):
        """

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
            self.set_the_is_extagged_flag_to_true(extag_name)
            for m in matches:
                self.extags[m.b: m.b + m.size] = \
                    [extag_name] * m.size

    def set_extags_of_arg2(self):
        """
        similar to Openie6.data_processing.labearg2_words()


        Returns
        -------
        None

        """

        li_2lt = self.arg2_words + self.loc_arg_words + \
                 self.time_arg_words
        li_2tl = self.arg2_words + self.time_arg_words + \
                 self.loc_arg_words
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
        self.arg2_is_extagged = True

    def set_extags_of_arg1_or_rel(self, arg_name):
        """
        similar to Openie6.data_processing.label_arg(),


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

    def set_extags_for_unused_num(self, unused_num):
        """

        Parameters
        ----------
        unused_num: int

        Returns
        -------
        None

        """
        assert unused_num in [1, 2, 3]
        unused_str = '[unused' + str(unused_num) + ']'
        # this equals -3, -2, -1 for 1, 2, 3
        unused_loc = -4 + unused_num
        if unused_num in [2, 3]:
            last_rel_loc = -1
        elif unused_num == 1:
            last_rel_loc = len(self.rel_words)
        else:
            assert False
        # [1:-1] when self.rel[0] = "[is]"
        # and self.rel[-1] = "of" or 'from"
        # [1: ] when self.rel[0] = "[is]"
        # and self.rel[-1] is anything
        # that doesn't start with "["
        if count_sub_reps(self.rel_words[1:last_rel_loc],
                          self.orig_sentL_words) == 1:
            matches = get_matches(
                self.rel_words[1:last_rel_loc], self.orig_sentL_words)
            self.set_extags_of_2_matches(matches, "REL")
            assert self.orig_sentL_words[unused_loc] == unused_str
            self.extags[unused_loc] = 'REL'
        elif len(self.rel_words) > 2 and \
                count_sub_reps(self.rel_words[1:last_rel_loc],
                               self.orig_sentL_words) == 0:
            matches = get_matches(
                self.rel_words[1:last_rel_loc], self.orig_sentL_words)
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

    def set_extags_of_IS_OF_FROM(self):
        """
        similar to Openie6.data_processing.label_is_of_relations()

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
                self.set_the_is_extagged_flag_to_true("REL")
                assert self.orig_sentL_words[-3] == '[unused1]'
                self.extags[-3] = 'REL'
            # IS-OF
            elif self.rel_words[0] == '[is]' and \
                    self.rel_words[-1] == '[of]':
                self.set_extags_for_unused_num(2)
            # IS  FROM
            elif self.rel_words[0] == '[is]' and \
                    self.rel_words[-1] == '[from]':
                self.set_extags_for_unused_num(3)
            # IS
            elif self.rel_words[0] == '[is]' and \
                    len(self.rel_words) > 1:
                assert self.rel_words[-1].startswith('[') == ""
                self.set_extags_for_unused_num(1)

    def set_extags_of_repeated_arg1(self):
        """
        similar to Openie6.data_processing.label_multiple_arg1()


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
                assert self.arg1_words == self.orig_sentL_words[
                                          loc0: loc0 + len(
                                              self.arg1_words)]
                self.set_the_is_extagged_flag_to_true("ARG1")
                # only extag the first occurrence of arg1
                self.extags[loc0: loc0 + len(self.arg1_words)] = \
                    ['ARG1'] * len(self.arg1_words)
            else:  # "REL" is not in extags
                assert False

    def set_extags_of_repeated_rel(self):
        """
        similar to Openie6.data_processing.label_multiple_rel()


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
                     range(len(self.orig_sentL_words))
                     if
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
                            find_xlist_item_that_minimizes_cost_fun(xlist,
                                                                    cost_fun)

                        assert rel_words == \
                               self.orig_sentL_words[
                               loc0: loc0 + len(rel_words)]
                        self.set_the_is_extagged_flag_to_true("REL")
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
                        self.set_the_is_extagged_flag_to_true('REL')
                        self.extags[loc0: loc0 + len(rel_words)] = \
                            ['REL'] * len(rel_words)

    def set_extags_of_loc_or_time(self, arg_name):
        """
        similar to Openie6.data_processing.label_time(),
        similar to Openie6.data_processing.label_loc()




        Parameters
        ----------
        arg_name: str

        Returns
        -------
        None

        """
        if arg_name == "time":
            matches = get_matches(self.time_arg_words, self.orig_sentL_words)
        elif arg_name == "loc":
            matches = get_matches(self.loc_arg_words, self.orig_sentL_words)
        else:
            assert False
        if has_2_matches(matches):
            self.set_extags_of_2_matches(matches, arg_name.upper())

    def set_extags(self):
        """

        Returns
        -------
        None

        """

        self.set_extags_of_arg2()
        self.set_extags_of_arg1_or_rel("arg1")
        self.set_extags_of_arg1_or_rel("rel")
        self.set_extags_of_IS_OF_FROM()
        self.set_extags_of_repeated_arg1()
        self.set_extags_of_repeated_rel()
        self.set_extags_of_loc_or_time("loc")
        self.set_extags_of_loc_or_time("time")

        self.extags_are_set = True

    @staticmethod
    def convert_to_sax_ex(carb_ex):
        """
        class Extraction:
        def __init__(self, pred, head_pred_index, sent,
             confidence, question_dist = '', index = -1):
        self.pred = pred
        self.head_pred_index = head_pred_index
        self.sent = sent
        self.args = []
        self.confidence = confidence
        self.matched = []
        self.questions = {}
        # self.indsForQuestions = defaultdict(lambda: set())
        self.is_mwp = False
        self.question_dist = question_dist
        self.index = index
    
        def addArg(self, arg, question = None):
        self.args.append(arg)
        if question:
            self.questions[question] = \
            self.questions.get(question,[]) + [Argument(arg)]
    
        class Argument:
        def __init__(self, arg):
            self.words = [x for x in arg[0].strip().split(' ') if x]
            self.posTags = map(itemgetter(1), nltk.pos_tag(self.words))
            self.indices = arg[1]
            self.feats = {}
    
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
                             confi=carb_ex.confidence)

    @staticmethod
    def convert_to_carb_ex(sax_ex):
        """
        openie6.model.write_to_files

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
                             confidence=sax_ex.confi,
                             index=1)
        # confidence only used for ordering exs, doesn't affect scores
        carb_ex.addArg(sax_ex.arg1)
        carb_ex.addArg(sax_ex.arg2)
        return carb_ex

    @staticmethod
    def get_carb_osent2_to_exs(sax_osent2_to_exs):
        """

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
    def get_ex_from_ilabels(ex_ilabels, orig_sentL, confi):
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
        ex_ilabels: list[int]
        orig_sentL: str
        confi: float

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

        rel = ' '.join(rel_words)
        if rel_case == 1:
            rel = 'is ' + rel
        elif rel_case == 2:
            rel = 'is ' + rel + ' of'
        elif rel_case == 3:
            rel = 'is ' + rel + ' from'

        arg1 = ' '.join(arg1_words)
        arg2 = ' '.join(arg2_words)

        # args = ' '.join(args_words)
        # loc_time = ' '.join(l_loc_time)
        # if not self.params.d["no_lt"]: # no_lt = no loc time
        #     arg2 = arg2 + ' ' + loc_time + ' ' + args

        extraction = SaxExtraction(orig_sentL,
                                   arg1,
                                   rel,
                                   arg2,
                                   confi=confi)

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
                l_old = [sax_ex.arg1, sax_ex.rel, sax_ex.arg2, sax_ex.confi]
                l_new = [new_sax_ex.arg1, new_sax_ex.rel, new_sax_ex.arg2,
                         new_sax_ex.confi]
                for i, old in enumerate(l_old):
                    new = l_new[i]
                    assert old == new


    main()
