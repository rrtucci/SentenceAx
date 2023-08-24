from Sample import *
from sax_utils import *


class ExTagsSample(Sample):
    def __init__(self, orig_sent):
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
    def write_extags_file(samples, path, with_scores):
        Sample.write_samples_file(samples,
                                  path,
                                  with_confidences=with_scores,
                                  with_unused_tokens=True)




