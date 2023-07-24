class CCList: # analogous to Coordination

    # cc: coordinating conjunction.
    # i.e., FANBOYS =
    # for , and, nor, but, or, yet, so


    def __init__(self, ccsent, ccloc, spans, seplocs=None, label=None):
        self.ccsent = ccsent
        # location always with respect to words
        # location of coordinating conjunctions
        self.ccloc = ccloc

        # a span (analogous to conjunct) is a tuple (i,j), where
        # i is position of first token/word
        # and j-1 of last token/word.
        # hence, span(5, 8) = range(5, 8) = (5,6,7)
        # spans is a list of spans
        self.spans = spans
        #separator (like commas) locations
        self.seplocs = seplocs
        self.label = label

        self.cctag_to_int = {'CP_START': 2, 'CP': 1,
                      'CC': 3, 'SEP': 4, 'OTHERS': 5, 'NONE': 0}

    def is_parent(self, child):
        # parent, child are instances of CCList
        min = child.spans[0][0]
        max = child.spans[-1][1] - 1

        # parent.span includes more words than child.span
        for span in self.spans:
            if span[0] <= min and span[1]-1 >= max:
                return True
        return False

    def contains_unbreakable_spans(self):

        unbreakable_words = ["between", "among", "sum", "total",
                             "addition", "amount", "value", "aggregate",
                             "gross",
                             "mean", "median", "average", "center",
                             "equidistant",
                             "middle"]

        # e.g. difference between apples and oranges
        # ccloc = 3
        # words = ["difference", "between", "apples", "and", "oranges"]
        # spans=[(0,3), (4,5)]

        unbreakable_locs = []
        words  = self.ccsent.words
        for i, word in enumerate(words):
            if word.lower() in unbreakable_words:
                unbreakable_locs.append(i)
        min = self.spans[0][0]
        max = self.spans[-1][1]-1
        for i in unbreakable_locs:
            if min <= i <= max:
                return True
        return False

    def to_string(self):
        words = self.ccsent.words
        cc_word = words[self.ccloc]
        span_words = [words[span[0]:span[1]] for span in self.spans]
        return cc_word + ": " + ", ".join(span_words)
