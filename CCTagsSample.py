from Sample import *
from SAXExtraction import *

class CCTagsSample(Sample):
    def __init__(self, orig_sent):
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



