from sax_globals import *
from SampleChild import *
            

class Sample:

    def __init__(self, orig_sent):
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



