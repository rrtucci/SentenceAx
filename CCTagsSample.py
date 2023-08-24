from Sample import *
from SAXExtraction import *

class CCTagsSample(Sample):
    def __init__(self, orig_sent):
        Sample.__init__(self, orig_sent)

    def construct_from_cctree(self, cctree):
        cctree.
        self.max_depth = len(l_ex)
        words = get_words(self.orig_sentL)
        self.confidences = []
        self.l_child = []
        for ex in l_ex:
            self.confidences.append(ex.confidence)
            assert ex.orig_sentL == self.orig_sentL
            ex.set_extags_of_all()
            child = SampleChild(ex.sent_extags)
            self.l_child.append(child)


    @staticmethod
    def write_cctags_file(samples, path, with_scores):
        Sample.write_samples_file(samples,
                                  path,
                                  with_confidences=with_scores,
                                  with_unused_tokens=False)



