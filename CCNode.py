from sax_utils import *

class CCNode:
    """
    similar to metric.Coordination

    loc = location of a word in orig_sent
    cc = coordinating conjunction.
    cc are FANBOYS = for , and, nor, but, or, yet, so

    CCNode is similar to metric.Coordination
    span is similar to a conjunct

    """

    def __init__(self, ccloc, osent_words):
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
        ccloc:
            location of coordinating conjunctions
        seplocs:
            separator (like commas) locations
        spans
        """
        self.osent_words = osent_words
        self.ccloc = ccloc
        self.spanned_locs = []
        self.unspanned_locs = []
        self.seplocs = []
        self.spans = []


    def check_spans(self):
        last_b = -1
        for a, b in self.spans:
            assert a < b
            assert last_b <= a
            last_b = b

    def check_all(self):
        self.check_spans()
        assert self.ccloc in self.spanned_locs
        for loc in self.seplocs:
            assert loc in self.spanned_locs
            
    def get_span_pair(self, mid_pair_id, throw_exception=False):
        """
        similar to metric.Coordination.get_pair()
        
        Parameters
        ----------
        mid_pair_id
        throw_exception

        Returns
        -------

        """
        span_pair = None
        for i in range(1, len(self.spans)):
            if mid_pair_id < self.spans[i][0]:
                span_pair = (self.spans[i - 1], self.spans[i])
                # there must be at least one point between the
                # 2 spans, or else the 2 spans would be 1
                assert mid_pair_id >= span_pair[0][1] and \
                       mid_pair_id < span_pair[1][0]
                break
        if throw_exception and span_pair is None:
            raise LookupError(
                "Could not find any span_pair for index={}".
                format(mid_pair_id))
        return span_pair


    def is_parent(self, child):
        # parent, child are instances of CCNode
        ch_min = child.spans[0][0]
        ch_max = child.spans[-1][1] - 1

        # self is parent iff at least one span in self.spans
        # contains all spans of the child
        for span in self.spans:
            if span[0] <= ch_min and ch_max <= span[1] - 1:
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
        unspanned_locs = []
        for i in range(len(self.osent_words)):
            if i not in self.spanned_locs:
                unspanned_locs.append(i)
        return unspanned_locs


    def an_unbreakable_word_is_not_spanned(self):
        """
        similar to data.remove_unbreakable_conjuncts()


        Parameters
        ----------
        osent_words

        Returns
        -------

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

    def get_cc_sent(self):
        words = []
        for i in self.spanned_locs:
            words.append(self.osent_words[i])
        return " ".join(words)


