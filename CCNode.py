from sax_utils import *

class CCNode:  # formerly Coordination
    """
    loc = location of a word in orig_sent
    cc = coordinating conjunction.
    cc are FANBOYS = for , and, nor, but, or, yet, so

    ccnode is formerly coordination
    span is formerly a conjunct

    """

    def __init__(self, depth):
        """
        e.g. "difference between apples and oranges"
        ccloc = 3
        words = ["difference", "between", "apples", "and", "oranges"]
        spans=[(0,3), (4,5)]

        a span (formerly a conjunct) is a tuple (i,j), where
        i is position of first token/word
        and j-1 of last token/word.
        hence, span(5, 8) covers range(5, 8) = (5,6,7)
        spans is a list of spans
        a token/word may be a punctuation mark

        location always with respect to word list of original sentence.

        Parameters
        ----------
        ccloc:
            location of coordinating conjunctions
        seplocs:
            separator (like commas) locations
        spans
        depth
        """
        self.depth=depth
        self.ccloc = -1
        self.seplocs = []
        self.spans = []


    def check_spans(self):
        last_b = 0
        for a, b in self.spans:
            assert a < b
            assert last_b <= a
            last_b = a

    def check_all(self):
        self.check_spans()
        spanned_locs = self.get_spanned_locs()
        assert self.ccloc in spanned_locs
        for loc in self.seplocs:
            assert loc in spanned_locs
            
    def get_span_pair(self, index, check=False):
        """
        formerly get_pair()
        
        Parameters
        ----------
        index
        check

        Returns
        -------

        """
        span_pair = None
        for i in range(1, len(self.spans)):
            if index < self.spans[i][0]:
                span_pair = (self.spans[i - 1], self.spans[i])
                assert index >= span_pair[0][1]  and \
                       index < span_pair[1][0]
                break
        if check and span_pair is None:
            raise LookupError(
                "Could not find any span_pair for index={}".format(index))
        return span_pair


    def is_parent(self, child):
        # parent, child are instances of CCNode
        ch_min = child.spans[0][0]
        ch_max = child.spans[-1][1] - 1

        # self is parent iff at least one span in self.spans
        # contains all spans of the child
        for span in self.spans:
            if span[0] <= ch_min and span[1] - 1 >= ch_max:
                return True
        return False

    def get_spanned_locs(self, extra_locs=None):
        spanned_locs = []
        for span in self.spans:
            for i in range(span[0], span[1]):
                spanned_locs.append(i)
        min = self.spans[0][0]
        max = self.spans[-1][1] - 1
        if extra_locs:
            for i in extra_locs:
                if i < min or i > max:
                    spanned_locs.append(i)
        return sorted(spanned_locs)

    def get_unspanned_locs(self):
        spanned_locs = self.get_spanned_locs()
        mini = min(spanned_locs)
        maxi = max(spanned_locs)
        unspanned_locs = []
        for i in range(mini, maxi + 1):
            if i not in spanned_locs:
                unspanned_locs.append(i)
        return unspanned_locs


    def omits_unbreakable_words(self, orig_words):
        """
        formerly remove_unbreakable_conjuncts()


        Parameters
        ----------
        orig_words

        Returns
        -------

        """

        unbreakable_locs = []
        spanned_locs = self.get_spanned_locs()
        words = [orig_words[loc] for loc in spanned_locs]
        for i, word in enumerate(words):
            if word.lower() in UNBREAKABLE_WORDS:
                unbreakable_locs.append(i)
                
        unspanned_locs = self.get_unspanned_locs()
        for i in unbreakable_locs:
            if i in unspanned_locs:
                return True
        return False

    def get_simple_sent(self, orig_words):
        spanned_locs = self.get_spanned_locs()
        words = []
        for i in spanned_locs:
            words.append(orig_words[i])
        return " ".join(words)


