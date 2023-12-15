from utils_gen import *
from collections import OrderedDict
from utiils_span import *


class CCNode:
    """
    CCNode is similar to Openie6.metric.Coordination
    
    This class defines the nodes (ccnodes) of a CCTree. Think of a CCNode by
    its __str__. For example, an __str__ for a CCNode might be (2, 5)6(7,
    23). (2, 5) is its left span `span_pair[0]`, 6 is its `ccloc` (cc
    location) and (7, 23) is its right span `span_pair[1]`. The CCNode's
    ccloc is always located between but outside the range of its left and
    right spans.

    A span is a tuple (i,j), where i is position of first token/word and j-1
    is the position  (location, loc) of last token/word. Hence, span(5,
    8) covers range(5, 8) = (5, 6, 7).

    self.span_pair is a list of 2 spans.
    e.g. osent = "He ate apples and oranges ."
    self.ccloc = 3
    self.osent_words = ["He", "ate", "apples", "and", "oranges", "."]
    self.span_pair=[(0,3), (4,5)]
    Note that the spans in self.span_pair exclude self.ccloc

    span is similar to Openie6.conjunct
    loc= location of a word relative to self.osent_words

    SentenceAx uses NLTK both to tokenize sentences into words (see
    utils_gen.get_words()), and to find the POS of each token/word. A
    token/word may be a punctuation mark. Openie6 mixes NLTK and Spacy (bad!)


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
    sep_locs: list[int]
        locs of separators (like commas and period)
    spanned_locs: list[int]
        locs that are within a span
    spans: list[tuple[int, int]]
        a list of spans. spans exclude location ccloc

    """

    def __init__(self,
                 ccloc,
                 depth,
                 osent_words,
                 span_pair):
        """
        Constructor

        Parameters
        ----------
        ccloc: int
        depth: int
        osent_words: list[str]
        span_pair: list[tuple[int,int]]
        """
        self.ccloc = ccloc
        self.depth = depth
        self.osent_words = osent_words
        self.span_pair = span_pair

        self.spanned_locs = self.get_spanned_locs()
        # print("lobhj", self.spanned_locs)

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
        for a, b in self.span_pair:
            assert a < b
            assert last_b <= a
            last_b = b
        # print("by56x", self.osent_words)
        # print("lkou", self.span_pair)
        # print("bnhj", self.ccloc)
        # print("bxxx", self.sep_locs)
        locs = self.get_spanned_locs()
        for loc in locs:
            if self.ccloc == loc:
                assert False
        min0 = self.span_pair[0][0]
        max0 = self.span_pair[1][1] - 1
        # print("nnmkl", self.span_pair)
        assert min0 <= self.ccloc <= max0, \
            f"min0={min0}, ccloc={self.ccloc}, max0={max0}"

    def is_parent(self, child):
        """
        similar to Openie6.data.is_parent()
        
        Returns True iff self is a parent of ccnode `child`.

        A CCNode `child` is a child of a CCNode `parent` if the child's left
        span, ccloc and right span are all fully contained within a single
        span (either the left or the right one) of the parent ccnode.

        Parameters
        ----------
        child: CCNode

        Returns
        -------
        bool

        """
        # parent, child are instances of CCNode
        ch_min = child.span_pair[0][0]
        ch_max = child.span_pair[1][1] - 1

        # self is parent iff
        # at least one span in self.span_pair contains all spans of the child
        for span in self.span_pair:
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

    def get_spanned_locs(self):
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
        for span in self.span_pair:
            for i in range(*span):
                if i < len(self.osent_words):
                    spanned_locs.append(i)
        return sorted(spanned_locs)

    def get_spanned_unbreakable_word_to_loc(self):
        """
        similar to Openie6.data.remove_unbreakable_conjuncts()

        This method returns a dictionary mapping the spanned unbreakable
        words to their locations.
         
        Used in CCTree.remove_bad_ccnodes()

        Returns
        -------
        dict[str, int]

        """

        spanned_unbreakable_word_to_loc = OrderedDict()
        spanned_words = [self.osent_words[loc] for loc in self.spanned_locs]
        for i, word in enumerate(self.osent_words):
            if word.lower() in SAX_UNBREAKABLE_WORDS and i in self.spanned_locs:
                spanned_unbreakable_word_to_loc[word] = i

        return spanned_unbreakable_word_to_loc

    def __eq__(self, node):
        """
        This method defines equality of 2 CCNode instances.

        Parameters
        ----------
        node: CCNode

        Returns
        -------
        bool

        """
        return self.ccloc == node.ccloc and \
            self.span_pair[0] == node.span_pair[0] and \
            self.span_pair[1] == node.span_pair[1]

    def __str__(self):
        """
        Returns a string containing self.span_pair and self.ccloc.

        Returns
        -------
        str

        """
        return str(self.span_pair[0]) + str(self.ccloc) + str(
            self.span_pair[1])
