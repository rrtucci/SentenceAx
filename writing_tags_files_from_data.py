from sax_globals import *


def write_tags_file(path,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens,
                    ll_score=None):
    with open(path, "w") as f:
        num_samples = len(l_orig_sent)
        for sam in range(num_samples):
            f.write(str(sam + 1) + "." + "\n")

            if with_unused_tokens:
                orig_sentL = l_orig_sent[sam] + UNUSED_TOKENS_STR
                f.write(orig_sentL + "\n")
            else:
                f.write(l_orig_sent[sam].orig_sent)
                for depth in range(len(ll_tags[0])):
                    end_str = "\n"
                    if ll_score:
                        end_str = "(" + ll_score[sam][depth] + ")"
                    f.write(ll_tags[sam][depth] + end_str)


def write_extags_file(path,
                      l_orig_sent,
                      ll_tags,
                      ll_score=None):
    write_tags_file(path,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens=True,
                    ll_score=ll_score)


def write_cctags_file(path,
                      l_orig_sent,
                      ll_tags,
                      ll_score=None):
    write_tags_file(path,
                    l_orig_sent,
                    ll_tags,
                    with_unused_tokens=False,
                    ll_score=ll_score)
