class MInput:
    def __init__(self, task):
        self.task = task
        self.num_samples = None
        self.l_sample = None

        self.l_orig_sent = []
        self.lll_ilabel = []
        self.ll_starting_word_loc = []
        self.ll_sentL_id = []

        self.l_pos_mask = []
        self.l_pos_locs = []
        self.l_verb_mask = []
        self.l_verb_locs = []
