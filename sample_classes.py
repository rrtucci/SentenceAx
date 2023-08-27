from sax_utils import *
from CCTree import *


class SampleChild:
    def __init__(self, tags=None):
        self.tags = tags
        self.ilabels = None
        self.confi = None
        self.simple_sent = None
        self.depth = None

    def get_tag_str(self):
        return " ".join(self.tags)

    def get_nontrivial_locs(self):
        locs = []
        for loc, tag in enumerate(self.tags):
            if tag != "NONE":
                locs.append(loc)
        return locs


class Sample:

    def __init__(self, orig_sent=None, ll_ilabel=None):
        self.orig_sent = orig_sent
        self.ll_ilabel = ll_ilabel
        self.l_child = None
        self.max_depth = None
        if ll_ilabel:
            self.max_depth = len(ll_ilabel)
            self.set_children(ll_ilabel)

    def set_children(self, ll_ilabel):
        self.l_child = []
        for depth, l_ilabel in enumerate(ll_ilabel):
            self.l_child.append(SampleChild())
            self.l_child[-1].depth = depth
            self.l_child[-1].ilabels = l_ilabel


class ExTagsSample(Sample):
    def __init__(self, orig_sent=None, ll_ilabel=None):
        Sample.__init__(self, orig_sent, ll_ilabel)
        if ll_ilabel:
            self.set_tags(ll_ilabel)

    def set_tags(self, ll_ilabel):
        for depth, l_ilabel in enumerate(ll_ilabel):
            child = self.l_child[depth]
            child.tags = []
            for ilabel in l_ilabel:
                child.tags.append(ILABEL_TO_EXTAG[ilabel])


class CCTagsSample(Sample):
    def __init__(self, orig_sent=None, ll_ilabel=None):
        Sample.__init__(self, orig_sent, ll_ilabel)
        if ll_ilabel:
            self.set_tags(ll_ilabel)

    def set_tags(self, ll_ilabel):
        for depth, l_ilabel in enumerate(ll_ilabel):
            child = self.l_child[depth]
            child.tags = []
            for ilabel in l_ilabel:
                child.tags.append(ILABEL_TO_CCTAG[ilabel])


class SplitPredSample():
    def __init__(self):
        self.l_child = []
        for i in range(MAX_CC_DEPTH):
            self.l_child.append(CCTagsSample())
            self.l_child[-1].l_child = []
            for j in range(MAX_EX_DEPTH):
                self.l_child[-1].l_child.append(ExTagsSample())


def write_samples_file(samples,
                       path,
                       with_confis,
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
                    if with_confis:
                        end_str = "(" + sam.child.confi + ")"
                    f.write(child.get_token_str() + end_str)


def write_extags_file(samples, path, with_confis=False):
    Sample.write_samples_file(samples,
                              path,
                              with_confis=with_confis,
                              with_unused_tokens=True)


def write_cctags_file(samples, path, with_confis=False):
    Sample.write_samples_file(samples,
                              path,
                              with_confis=with_confis,
                              with_unused_tokens=False)
