from sax_utils import *

class SampleChild:
    def __init__(self, tags=None):
        self.tags=tags
    def get_tag_str(self):
        return " ".join(self.tags)
    def get_nontrivial_locs(self):
        locs = []
        for loc, tag in enumerate(self.tags):
            if tag != "NONE":
                locs.append(loc)
        return locs

class Sample:

    def __init__(self, orig_sent=None):
        self.orig_sent = orig_sent
        self.l_child=None
        self.max_depth = None
        self.confidences = None
    @staticmethod
    def write_samples_file(samples,
                           path,
                           with_confidences,
                           with_unused_tokens):
        with open(path, "w") as f:
            for k, sam in enumerate(samples):
                f.write(str(k+1) + "." + "\n")

                if with_unused_tokens:
                    orig_sentL = sam.orig_sent + UNUSED_TOKENS_STR
                    f.write(orig_sentL + "\n")
                else:
                    f.write(sam.orig_sent)
                    for i, child in enumerate(sam.l_child):
                        end_str = "\n"
                        if with_confidences:
                            end_str = "(" + sam.confidences[i] + ")"
                        f.write(child.get_token_str() + end_str)


class ExTagsSample(Sample):
    def __init__(self, orig_sent=None):
        Sample.__init__(self, orig_sent)
        self.orig_sentL = self.orig_sent + UNUSED_TOKENS_STR

    def construct_from_extraction_list(self, l_ex):
        self.max_depth = len(l_ex)
        self.confidences = []
        self.l_child = []
        for ex in l_ex:
            self.confidences.append(ex.confidence)
            assert ex.orig_sentL == self.orig_sentL
            ex.set_extags_of_all()
            child = SampleChild(ex.sent_extags)
            self.l_child.append(child)

    @staticmethod
    def write_extags_file(samples, path, with_scores=False):
        Sample.write_samples_file(samples,
                                  path,
                                  with_confidences=with_scores,
                                  with_unused_tokens=True)

class CCTagsSample(Sample):
    def __init__(self, orig_sent=None):
        Sample.__init__(self, orig_sent)

    def construct_from_cctree(self, cctree):
        assert cctree.orig_sent == self.orig_sent
        cc_sents, spanned_sents, l_spanned_locs = cctree.get_cc_sents()

        self.max_depth = len(cc_sents)
        self.l_child = []
        for cc_sent in cc_sents:
            child = SampleChild(get_words(cc_sent))
            self.l_child.append(child)


    @staticmethod
    def write_cctags_file(samples, path, with_scores=False):
        Sample.write_samples_file(samples,
                                  path,
                                  with_confidences=with_scores,
                                  with_unused_tokens=False)


class SplitPredSample():
    def __init__(self, max_cc_depth, max_ex_depth):
        self.max_cc_depth = max_cc_depth
        self.max_ex_depth = max_ex_depth
        self.l_child = []
        for i in range(max_cc_depth):
            self.l_child.append(CCTagsSample())
            self.l_child[-1].l_child = []
            for j in range(max_ex_depth):
                self.l_child[-1].l_child.append(ExTagsSample())
































    






