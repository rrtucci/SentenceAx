from sax_utils import *


class CCNode:
    """
    CCNode is similar to Openie6.metric.Coordination
    
    This class is for defining the nodes (ccnodes) of a CCTree.

    span is similar to Openie6.conjunct
    loc= location of a word relative to self.osent_words

    SentenceAx uses NLTK both to tokenize sentences into words (see
    sax_utils.get_words()), and to find the POS of each token/word. A
    token/word may be a punctuation mark.

    A span is a tuple (i,j), where i is position of first token/word and j-1
    is the position of last token/word. Hence, span(5, 8) covers range(5,
    8) = (5, 6, 7).

    self.spans is a list of spans.

    e.g. osent = "He ate apples and oranges ."
    self.ccloc = 3
    self.osent_words = ["He", "ate", "apples", "and", "oranges", "."]
    self.spans=[(0,3), (4,5)]
    Note that the spans in self.spans exclude self.ccloc

    Attributes
    ----------
    ccloc: int
        location of cc (coordinating conjunction) (see FANBOYS). 1-1 map
        between cclocs and CCNodes
    depth: int
        0-based position in CCTree. depth is just a label for distinguishing
        between CCNodes. Not used for anything.
    osent_words: list[str]
        list of words in osent (original sentence)
    seplocs: list[int]
        locs of separators (like commas and period)
    spanned_locs: list[int]
        locs that are within a span
    spans: list[tuple[int, int]]
        a list of spans. spans exclude location ccloc
    unspanned_locs: list[int]
        locs that are outside all spans

    """

    def __init__(self,
                 ccloc,
                 depth,
                 osent_words,
                 seplocs,
                 spans):
        """
        Constructor

        Parameters
        ----------
        ccloc: int
        depth: int
        seplocs: list[int]
        osent_words: list[str]
        spans: list[tuple[int,int]]
        """
        self.ccloc = ccloc
        self.depth = depth
        self.osent_words = osent_words
        self.seplocs = seplocs
        self.spans = spans

        self.spanned_locs = self.get_spanned_locs()
        # print("lobhj", self.spanned_locs)
        self.unspanned_locs = self.get_unspanned_locs()

    def check_self(self):
        """
        This method checks that the spans don't overlap. It also checks that
        self.ccloc is not included in the spanned locs. It also checks that
        self.ccloc is between the lowest and highest spanned loc.

        Returns
        -------
        None

        """
        last_b = -1
        for a, b in self.spans:
            assert a < b
            assert last_b <= a
            last_b = b
        # print("by56x", self.osent_words)
        # print("lkou", self.spans)
        # print("bnhj", self.ccloc)
        # print("bxxx", self.seplocs)
        locs = self.get_spanned_locs()
        for loc in locs:
            if self.ccloc == loc:
                assert False
        min0 = self.spans[0][0]
        max0 = self.spans[-1][1] - 1
        assert min0 <= self.ccloc <= max0

    def get_span_pair(self, midpoint_id, allow_None=False):
        """
        similar to Openie6.metric.Coordination.get_pair()

        This method returns two **consecutive** spans such that
        `midpoint_id` is between the 2 spans but outside both of them. If no
        span_pair is found and allow_None=False, it raises an exception.

        Used in CCScoreManager.absorb_new_sample()
        
        Parameters
        ----------
        midpoint_id: int
        allow_None: bool

        Returns
        -------
        list[tuple[int,int], tuple[int, int]]


        """
        span_pair = None
        for i in range(1, len(self.spans)):
            if midpoint_id < self.spans[i][0]:
                span_pair = (self.spans[i - 1], self.spans[i])
                # there must be at least one point between the
                # 2 spans, or else the 2 spans would be 1
                assert span_pair[0][1] <= midpoint_id \
                       < span_pair[1][0]
                break
        if not allow_None and span_pair is None:
            raise LookupError(
                f"Could not find any span_pair for index={midpoint_id}.")
        return span_pair

    def is_parent(self, child):
        """
        similar to Openie6.data.is_parent()
        
        Returns True iff self is a parent of ccnode `child`.

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
        Returns True iff self is a child of ccnode `parent`.

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
        This method returns the word locations, relative to 
        self.osent_words, of all words inside a span (spanned words).

        Parameters
        ----------
        fat: bool
            iff fat=True, the spanned_locs are augmented by adding to them
            all locs to the left of the first span and all locs to the right
            of the last span.

        Returns
        -------
        list[int]

        """
        spanned_locs = []
        for span in self.spans:
            for i in range(span[0], span[1]):
                if i < len(self.osent_words):
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
        This method returns the word locations, relative to 
        self.osent_words, of all words outside a span (unspanned words).

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

        This method returns True iff an unbreakable word is not inside any 
        of the spans.
         
        Used in CCTree.remove_bad_ccnodes()

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
        Returns a string containing self.spans and self.ccloc.

        Returns
        -------
        str

        """
        return str(self.spans) + str(self.ccloc)
