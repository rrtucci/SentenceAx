from Sample import *

class SimpSentsSample(Sample):
    def __init__(self, orig_sent):
        Sample.__init__(self, orig_sent)

    @staticmethod
    def write_simple_sents_file(samples,
                                path,
                                with_confidences):
        Sample.write_samples_file(samples,
                                  path,
                                  with_confidences,
                                  with_unused_tokens=False)


