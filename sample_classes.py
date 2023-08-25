from sax_utils import *
from CCTree import *


class SampleChild:
    def __init__(self, tags=None):
        self.tags = tags
        self.confidence = None
        self.simple_sent = None

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
        self.l_child = None
        self.max_depth = None
        self.orig_sentL = None
        self.ll_label = None

    @staticmethod
    def write_samples_file(samples,
                           path,
                           with_confidences,
                           with_unused_tokens):
        with open(path, "w") as f:
            for k, sam in enumerate(samples):
                f.write(str(k + 1) + "." + "\n")

                if with_unused_tokens:
                    orig_sentL = sam.orig_sent + UNUSED_TOKENS_STR
                    f.write(orig_sentL + "\n")
                else:
                    f.write(sam.orig_sent)
                    for child in sam.l_child:
                        end_str = "\n"
                        if with_confidences:
                            end_str = "(" + sam.child.confidence + ")"
                        f.write(child.get_token_str() + end_str)


class ExTagsSample(Sample):
    def __init__(self, orig_sent=None, ll_label=None):
        Sample.__init__(self, orig_sent, ll_label)

        if orig_sent and ll_label:
            self.fill_from_ll_label(orig_sent, ll_label)

    def fill_from_ll_label(self,
                           orig_sent,
                           ll_label):
        self.orig_sent = orig_sent
        self.orig_sentL = self.orig_sent + UNUSED_TOKENS_STR
        self.ll_label = ll_label
        self.max_depth = len(ll_label)

        self.l_child = []
        for i in range(self.max_depth):
            child = SampleChild()
            for l_label in ll_label:
                child.tags = []
                for label in l_label:
                    child.tags.append(LABEL_TO_EXTAG[label])
            simp_words = []
            orig_words = get_words(orig_sent)
            for k, tag in enumerate(child.tags):
                if tag is not "NONE":
                    simp_words.append(orig_words[i])

            child.simple_sent = " ".join(simp_words)
            self.l_child.append(child)

    # def fill_from_extraction_list(self, l_ex):
    #     self.orig_sentL = l_ex[0].orig_sentL
    #     self.orig_sent = self.orig_sentL.split("[used1]")[0].strip()
    #     self.max_depth = len(l_ex)
    #     self.l_child = []
    #     for ex in l_ex:
    #         assert self.orig_sent == ex.orig_sentL
    #         ex.set_extags_of_all()
    #         child = SampleChild(ex.sent_extags)
    #         child.confidence = ex.confidence
    #         child.simple_sent = ex.get_simple_sent()
    #         self.l_child.append(child)


class CCTagsSample(Sample):
    def __init__(self, orig_sent=None, ll_label=None):
        Sample.__init__(self, orig_sent, ll_label)

        if orig_sent and ll_label:
            self.fill_from_ll_label(orig_sent, ll_label)

    def fill_from_ll_label(self,
                                orig_sent,
                                ll_label):

        self.orig_sent = orig_sent
        self.orig_sentL = self.orig_sent + UNUSED_TOKENS_STR
        self.ll_label = ll_label
        self.max_depth = len(ll_label)

        cctree = CCTree(orig_sent, ll_label)
        cc_sents, spanned_sents, l_spanned_locs = cctree.get_cc_sents()

        assert self.max_depth == len(cc_sents)
        self.l_child = []
        for cc_sent in cc_sents:
            child = SampleChild()
            for l_label in ll_label:
                child.tags = []
                for label in l_label:
                    child.tags.append(LABEL_TO_CCTAG[label])

            child.simple_sent = cc_sent
            self.l_child.append(child)

    @staticmethod
    def write_cctags_file(samples, path, with_confidences=False):
        Sample.write_samples_file(samples,
                                  path,
                                  with_confidences=with_confidences,
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
