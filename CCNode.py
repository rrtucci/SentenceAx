from sax_utils import *


class CCNode:
    """
    similar to Openie6.metric.Coordination

    loc = location of a word in orig_sent
    cc = coordinating conjunction.
    cc are FANBOYS = for , and, nor, but, or, yet, so

    CCNode is similar to Openie6.metric.Coordination
    span is similar to a conjunct

    Attributes
    ----------
    ccloc: int
    depth: int
    osent_words: list[str]
    seplocs: list[int]
    spanned_locs: list[int]
    spans: list[tuple[int, int]]
    unspanned_locs: list[int]



    """

    def __init__(self,
                 ccloc,
                 depth,
                 osent_words,
                 seplocs,
                 spans):
        """
        e.g. "difference between apples and oranges"
        ccloc = 3
        words = ["difference", "between", "apples", "and", "oranges"]
        spans=[(0,3), (4,5)]

        a span (similar to a conjunct) is a tuple (i,j), where
        i is position of first token/word
        and j-1 of last token/word.
        hence, span(5, 8) covers range(5, 8) = (5,6,7)
        spans is a list of spans
        a token/word may be a punctuation mark

        location always with respect to word list of original sentence.

        Parameters
        ----------
        ccloc: int
            location of coordinating conjunctions
        depth: int
        seplocs: list[int]
            separator (like commas) locations
        osent_words: list[str]
        spans: list[tuple[int,int]]
        """
        self.ccloc = ccloc
        self.depth = depth
        self.osent_words = osent_words
        self.seplocs = seplocs
        self.spans = spans
        # depth is just a label
        # for distinguishing between CCNodes. Not used for anything

        self.spanned_locs = self.get_spanned_locs()
        # print("lobhj", self.spanned_locs)
        self.unspanned_locs = self.get_unspanned_locs()

    def check_spans(self):
        """

        Returns
        -------
        None

        """
        last_b = -1
        for a, b in self.spans:
            assert a < b
            assert last_b <= a
            last_b = b

    def check_all(self):
        """

        Returns
        -------
        None

        """
        self.check_spans()
        # print("by56x", self.osent_words)
        # print("lkou", self.spans)
        # print("bnhj", self.ccloc)
        # print("bxxx", self.seplocs)
        min0 = self.spans[0][0]
        max0 = self.spans[-1][1] - 1
        assert min0 <= self.ccloc <= max0
        # for loc in self.seplocs:
        #     assert min0<= loc <= max0

    def get_span_pair(self, mid_pair_id, check_answer=False):
        """
        similar to Openie6.metric.Coordination.get_pair()

        used in CCReport.grow()
        
        Parameters
        ----------
        mid_pair_id: int
        check_answer: bool

        Returns
        -------
        list[tuple[int,int], tuple[int, int]]


        """
        span_pair = None
        for i in range(1, len(self.spans)):
            if mid_pair_id < self.spans[i][0]:
                span_pair = (self.spans[i - 1], self.spans[i])
                # there must be at least one point between the
                # 2 spans, or else the 2 spans would be 1
                assert span_pair[0][1] <= mid_pair_id \
                       < span_pair[1][0]
                break
        if check_answer and span_pair is None:
            raise LookupError(
                "Could not find any span_pair for index={}".
                format(mid_pair_id))
        return span_pair

    def is_parent(self, child):
        """
        similar to Openie6.data.is_parent()

        Parameters
        ----------
        child: CCNode

        Returns
        -------
        bool

        """
        # parent, child are instances of CCNode
        ch_min = child.spans[0][0]
        ch_max = child.spans[-1][1] - 1

        # self is parent iff
        # at least one span in self.spans contains all spans of the child
        for span in self.spans:
            if span[0] <= ch_min and ch_max <= span[1] - 1:
                return True
        return False

    def is_child(self, parent):
        """

        Parameters
        ----------
        parent: CCNode

        Returns
        -------
        bool

        """
        return parent.is_parent(self)

    def get_spanned_locs(self, fat=False):
        """

        Parameters
        ----------
        fat: bool

        Returns
        -------
        list[int]

        """
        spanned_locs = []
        for span in self.spans:
            for i in range(span[0], span[1]):
                spanned_locs.append(i)
        min0 = self.spans[0][0]
        max0 = self.spans[-1][1] - 1
        if fat:
            for i in range(len(self.osent_words)):
                if i < min0 or i > max0:
                    spanned_locs.append(i)
        return sorted(spanned_locs)

    def get_unspanned_locs(self):
        """

        Returns
        -------
        list[int]

        """
        unspanned_locs = []
        for i in range(len(self.osent_words)):
            if i not in self.spanned_locs:
                unspanned_locs.append(i)
        return unspanned_locs

    def an_unbreakable_word_is_not_spanned(self):
        """
        similar to Openie6.data.remove_unbreakable_conjuncts()

        used in CCTree.fix_ccnodes()

        Returns
        -------
        bool

        """

        unbreakable_locs = []
        spanned_words = [self.osent_words[loc] for loc in self.spanned_locs]
        for i, word in enumerate(spanned_words):
            if word.lower() in UNBREAKABLE_WORDS:
                unbreakable_locs.append(i)

        for i in unbreakable_locs:
            if i in self.unspanned_locs:
                return True
        return False

    def __str__(self):
        """

        Returns
        -------
        str

        """
        return str(self.spans) + str(self.ccloc)
