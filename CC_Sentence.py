class CC_Sentence: # analogous to Coordination

    # cc: coordinating conjunction.
    # i.e., FANBOYS =
    # for , and, nor, but, or, yet, so


    def __init__(self, cc_loc, spans, sep_locs=None, label=None):
        # location of conjunction
        self.cc_loc = cc_loc

        # a span (analogous to conjunct) is a tuple (i,j), where
        # i is position of first token/word
        # and j of last token/word
        # spans is a list of spans
        self.spans = spans
        #separator (like commas) locations
        self.sep_locs = sep_locs
        self.label = label

        self.cctag_to_int = {'CP_START': 2, 'CP': 1,
                      'CC': 3, 'SEP': 4, 'OTHERS': 5, 'NONE': 0}

    def is_parent(self, child):
        # parent, child are instances of CC_Sentence
        min = child.spans[0][0]
        max = child.spans[-1][-1]

        for span in self.spans:
            if span[0] <= min and span[1] >= max:
                return True
        return False
