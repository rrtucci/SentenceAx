import difflib
from my_globals import *
from allen_tool import read_allen_file, get_num_sents_in_allen_file
from math import floor


# extag = extraction tag

class Extraction:
    def __init__(self,
                 in_ztz=None,
                 arg1=None,
                 rel=None,
                 arg2=None,
                 confidence=None,
                 time=None,
                 loc=None):
        # allen args
        self.in_ztz = in_ztz
        self.arg1 = arg1
        self.rel = rel
        self.arg2 = arg2
        self.confidence = confidence
        # time, loc
        self.time = time
        self.loc = loc
        # tokens
        self.in_tokens = []
        self.in3_tokens = self.in_tokens + UNUSED_TOKENS
        self.arg1_tokens = []
        self.rel_tokens = []
        self.arg2_tokens = []
        self.loc_tokens = []
        self.time_tokens = []
        self.tokenizables = ["in_ztz", "arg1", "rel", "arg2", "loc", "time"]
        self.tokenize_all()
        # ztz_extags, is_extagged
        self.base_extags = ["ARG1", "REL", "ARG2", "TIME", "LOC"]
        self.ztz_extags = None
        self.name_to_is_extagged = {name: False for name in self.base_extags}

        self.extag_to_int = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                      'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}

    def to_string(self):
         return " ".join([self.arg1, self.rel. self.arg2])

    def tokenize_one_attr(self, name):
        assert name in self.tokenizables
        attr = getattr(self, name)
        attr_tokens = getattr(self, name + "_tokens")
        if len(attr_tokens)==0:
            attr_tokens = tokenize(attr)
                
    def tokenize_all(self):
        for name in self.tokenizables:
            self.tokenize_one_attr(name)
        self.in3_tokens = self.in_tokens + UNUSED_TOKENS

    def set_is_extagged_to_true(self, extag_name):
        assert extag_name in self.base_extags
        self.name_to_is_extagged[extag_name] = True

    def extag_it_matches_2(self, matches, extag_name):
        assert extag_name in self.base_extags
        assert good_matches_2(matches)
        m0 = matches[0]
        self.ztz_extags[m0.b: m0.b + m0.size] = [extag_name] * m0.size
        self.set_is_extagged_to_true(extag_name)

    def extag_it_arg2(self):
        self.tokenize_all()

        if not(self.arg2) and len(args_tokens) == 0 and \
                len(self.loc_tokens) == 0 and \
                len(self.time_tokens) == 0:
            self.arg2_extagged = True

        elif count_subs(
                self.arg2_tokens + args_tokens + self.loc_tokens +
                self.time_tokens, self.in3_tokens) == 1:
            matches = get_matches(
                self.arg2_tokens + args_tokens + self.loc_tokens +
                self.time_tokens, self.in3_tokens)
            self.extag_it_matches_2(matches, "ARG2")

        # elif count_subs(
        #         self.arg2_tokens + args_tokens + self.time_tokens +
        #         self.loc_tokens, self.in3_tokens) == 1:
        #     matches = get_matches(
        #         self.arg2_tokens + args_tokens + self.time_tokens +
        #         self.loc_tokens, self.in3_tokens)
        #     self.extag_it_matches_2(matches, "ARG2")

        elif count_subs(
                self.arg2_tokens + args_tokens + self.time_tokens,
                self.in3_tokens) == 1:
            matches = get_matches(
                self.arg2_tokens + args_tokens + self.time_tokens,
                self.in3_tokens)
            self.extag_it_matches_2(matches, "ARG2")

        elif count_subs(
                self.arg2_tokens + args_tokens + self.loc_tokens,
                self.in3_tokens) == 1:
            matches = get_matches(
                self.arg2_tokens + args_tokens + self.loc_tokens,
                self.in3_tokens)
            self.extag_it_matches_2(matches, "ARG2")

        elif count_subs(
                self.arg2_tokens + self.time_tokens + self.loc_tokens,
                self.in3_tokens) == 1:
            matches = get_matches(
                self.arg2_tokens + self.time_tokens + self.loc_tokens,
                self.in3_tokens)
            self.extag_it_matches_2(matches, "ARG2")

        # elif count_subs(
        #         self.arg2_tokens + self.loc_tokens + self.time_tokens,
        #         self.in3_tokens) == 1:
        #     matches = get_matches(
        #         self.arg2_tokens + self.loc_tokens + self.time_tokens,
        #         self.in3_tokens)
        #     self.extag_it_matches_2(matches, "ARG2")

        elif count_subs(
                self.arg2_tokens + self.time_tokens, self.in3_tokens) == 1:
            matches = get_matches(
                self.arg2_tokens + self.time_tokens, self.in3_tokens)
            self.extag_it_matches_2(matches, "ARG2")

        elif count_subs(
                self.arg2_tokens + self.loc_tokens, self.in3_tokens) == 1:
            matches = get_matches(
                self.arg2_tokens + self.loc_tokens, self.in3_tokens)
            self.extag_it_matches_2(matches, "ARG2")

        elif count_subs(
                self.time_tokens + self.loc_tokens, self.in3_tokens) == 1:
            matches = get_matches(
                self.time_tokens + self.loc_tokens + self.in3_tokens)
            self.extag_it_matches_2(matches, "ARG2")

        # elif count_subs(
        #         self.loc_tokens + self.time_tokens, self.in3_tokens) == 1:
        #     matches = get_matches(
        #         self.loc_tokens + self.time_tokens , self.in3_tokens)
        #     self.extag_it_matches_2(matches, "ARG2")

        elif count_subs(
                self.loc_tokens, self.in3_tokens) == 1:
            matches = get_matches(
                self.loc_tokens, self.in3_tokens)
            self.extag_it_matches_2(matches, "ARG2")

        elif count_subs(
                self.time_tokens, self.in3_tokens) == 1:
            matches = get_matches(
                self.time_tokens, self.in3_tokens)
            self.extag_it_matches_2(matches, "ARG2")

    def extag_it_arg1_rel(self, arg_name):
        self.tokenize_all()
        assert arg_name  in self.tokenizables
        if arg_name == "arg1":
            arg_tokens = getattr(self, "arg1_tokens")
        elif arg_name == "rel":
            arg_tokens = getattr(self, "rel_tokens")
        else:
            assert False
        if count_subs(arg_tokens, self.in3_tokens) == 1:
            matches = get_matches(arg_tokens, self.in3_tokens)
            assert good_matches_2(matches)
            self.set_is_extagged_to_true(arg_name.upper())
            m0 = matches[0]
            self.ztz_extags[m0.b: m0.b + m0.size] = [arg_name.upper()] * m0.size


        elif count_subs(arg_tokens, self.in3_tokens) == 0:
            matches = get_matches(arg_tokens, self.in3_tokens)
            if good_matches_gt_2(matches):
                self.set_is_extagged_to_true(arg_name.upper())
                for m in matches:
                    self.ztz_extags[m.b: m.b + m.size] = \
                        [arg_name.upper()] * m.size

    def extag_it_is_of_from(self):
        self.tokenize_all()
        
        rel_is_extagged = self.name_to_is_extagged["rel"]
        if not rel_is_extagged and len(self.rel_tokens) > 0:
            if self.rel == '[is]':
                self.set_is_extagged_to_true("REL")
                assert self.in3_tokens[-3] == '[unused1]'
                self.ztz_extags[-3] = 'REL'

            elif self.rel_tokens[0] == '[is]' and \
                    self.rel_tokens[-1] == '[of]':
                if len(self.rel_tokens) > 2 and \
                        count_subs(self.rel_tokens[1:-1], self.in3_tokens)\
                        == 1:
                    matches = get_matches(self.rel_tokens[1:-1], 
                                          self.in3_tokens)
                    self.extag_it_matches_2(matches, "REL")
                    rel_is_extagged = True
                    assert self.in3_tokens[-2] == '[unused2]'
                    self.ztz_extags[-2] = 'REL'

                elif len(self.rel_tokens) > 2 and \
                        count_subs(self.rel_tokens[1:-1], self.in3_tokens) \
                        == 0:
                    matches = get_matches(self.rel_tokens[1:-1], 
                                          self.in3_tokens)
                    if good_matches_gt_2(matches):
                        self.set_is_extagged_to_true("REL")
                        for m in matches:
                            self.ztz_extags[m.b: m.b + m.size] = \
                                ["REL"] * m.size
                        assert self.in3_tokens[-2] == '[unused2]'
                        self.ztz_extags[-2] = 'REL'

            elif self.rel_tokens[0] == '[is]' and \
                    self.rel_tokens[-1] == '[from]':
                if len(self.rel_tokens) > 2 and\
                        count_subs(self.rel_tokens[1:-1], self.in3_tokens) \
                        == 1:
                    matches = get_matches(self.rel_tokens[1:-1], 
                                          self.in3_tokens)
                    self.extag_it_matches_2(matches, "REL")
                    rel_is_extagged = True
                    assert self.in3_tokens[-1] == '[unused3]'
                    self.ztz_extags[-1] = 'REL'

                elif len(self.rel_tokens) > 2 and \
                        count_subs(self.rel_tokens[1:-1], self.in3_tokens) \
                        == 0:
                    matches = get_matches(self.rel_tokens[1:-1], 
                                          self.in3_tokens)
                    if good_matches_gt_2(matches):
                        rel_is_extagged = True
                        for m in matches:
                            self.ztz_extags[m.b: m.b + m.size] = \
                                ["REL"] * m.size
                        assert self.in3_tokens[-1] == '[unused3]'
                        self.ztz_extags[-1] = 'REL'

            elif self.rel_tokens[0] == '[is]' and len(self.rel_tokens) > 1:
                assert not self.rel_tokens[-1].startswith('[')
                if count_subs(self.rel_tokens[1:], self.in3_tokens) == 1:
                    matches = get_matches(
                        self.rel_tokens[1:], self.in3_tokens)
                    self.extag_it_matches_2(matches, "REL")
                    self.set_is_extagged_to_true("REL")
                    assert self.in3_tokens[-3] == '[unused1]'
                    self.ztz_extags[-3] = 'REL'

                elif len(self.rel_tokens) > 2 and \
                        count_subs(self.rel_tokens[1:], self.in3_tokens) == 0:
                    matches = get_matches(
                        self.rel_tokens[1:-1], self.in3_tokens)
                    if good_matches_gt_2(matches):
                        self.set_is_extagged_to_true("REL")
                        for m in matches:
                            self.ztz_extags[m.b: m.b + m.size] = \
                                ["REL"] * m.size
                        assert self.in3_tokens[-3] == '[unused1]'
                        self.ztz_extags[-3] = 'REL'

    def extag_it_multiple_arg1(self):
        self.tokenize_all()
        rel_is_extagged = self.name_to_is_extagged["REL"]
        arg1_is_extagged = self.name_to_is_extagged["ARG1"]

        if rel_is_extagged and \
                not arg1_is_extagged and \
                count_subs(self.arg1_tokens, self.in3_tokens) > 1:
            starting_locs = [j for j in
                range(len(self.in3_tokens)) if
                sub_exists(self.arg1_tokens, self.in3_tokens, j)]
            assert len(starting_locs) > 1

            min_dist = int(1E8)
            if 'REL' in self.ztz_extags:
                rel_loc = self.ztz_extags.index('REL')
                final_loc = -1

                for loc in starting_locs:
                    dist = abs(rel_loc - loc)
                    if dist < min_dist:
                        min_dist = dist
                        final_loc = loc

                assert self.arg1_tokens == self.in3_tokens[
                    final_loc: final_loc + len(self.arg1_tokens)]
                self.set_is_extagged_to_true("ARG1")
                self.ztz_extags[final_loc: final_loc + len(self.arg1_tokens)] =\
                    ['ARG1'] * len(self.arg1_tokens)
            else:
                assert False

    def extag_it_multiple_rel(self):
        self.tokenize_all()
        arg1_is_extagged = self.name_to_is_extagged["ARG1"]
        arg2_is_extagged = self.name_to_is_extagged["ARG2"]
        rel_is_extagged = self.name_to_is_extagged["REL"]

        if arg1_is_extagged and arg2_is_extagged and not rel_is_extagged and\
                len(self.rel_tokens) > 0:
            rt = None
            if count_subs(self.rel_tokens, self.in3_tokens) > 1:
                rt = self.rel_tokens
            elif self.rel_tokens[0] == '[is]' and \
                    count_subs(self.rel_tokens[1:], self.in3_tokens) > 1:
                rt = self.rel_tokens[1:]
            elif self.rel_tokens[0] == '[is]' and \
                    self.rel_tokens[-1].startswith('[') and \
                    count_subs(self.rel_tokens[1:-1], self.in3_tokens) > 1:
                rt = self.rel_tokens[1:-1]

            if rt:
                starting_locs = [j for j in range(len(self.in3_tokens))
                        if sub_exists( rt, self.in3_tokens, j)]
                assert len(starting_locs) > 1

                min_dist = int(1e8)
                if 'ARG1' in self.ztz_extags and\ 
                        (not(self.arg2) or 'ARG2' in self.ztz_extags):
                    arg1_loc = self.ztz_extags.index('ARG1')
                    if not(self.arg2):
                        final_loc = -1
                        for loc in starting_locs:
                            dist = abs(arg1_loc - loc)
                            if dist < min_dist:
                                min_dist = dist
                                final_loc = loc

                        assert rt == \
                            self.in3_tokens[final_loc: final_loc + len(rt)]
                        self.set_is_extagged_to_true("REL")
                        self.ztz_extags[final_loc: final_loc + len(rt)] =\
                            ['REL'] * len(rt)

                    else:
                        arg2_loc = self.ztz_extags.index('ARG2')
                        final_loc = -1
                        for loc in starting_locs:
                            dist = abs(arg1_loc - loc) + abs(arg2_loc - loc)
                            if dist < min_dist:
                                min_dist = dist
                                final_loc = loc

                        assert rt == \
                               self.in3_tokens[final_loc: final_loc + len(rt)]
                        self.set_is_extagged_to_true('REL')
                        self.ztz_extags[final_loc: final_loc + len(rt)] =\
                            ['REL'] * len(rt)

    def extag_it_loc_time(self, arg_name):
        if arg_name == "time":
            arg = self.time
        elif arg_name == "loc":
            arg = self.loc
        else:
            assert False
        tokens = tokenize(arg)
        matches = get_matches(tokens, self.in_ztz)
        self.extag_it_matches_2(matches, arg_name.upper())
            
    def do_all_extagging(self):
        self.extag_it_arg2()
        self.extag_it_arg1_rel("arg1")
        self.extag_it_arg1_rel("rel")
        self.extag_it_is_of_from()
        self.extag_it_multiple_arg1()
        self.extag_it_multiple_rel()
        self.extag_it_loc_time("loc")
        self.extag_it_loc_time("time")

def tokenize(ztz):
    if ztz:
        x = ztz.strip().split()
    else:
        x = None
    return x


def count_subs(sub, full): # seq_in_seq
    return str(full)[1:-1].count(str(sub)[1:-1])

def sub_exists(sub, full, loc): # starts_with
    return all(sub[i] == full[loc + i] for i in range(0, len(sub)))


def good_matches_2(matches):
    return len(matches) == 2 and \
        matches[0].a == 0 and \
        matches[0].size == matches[1].a and \
        matches[1].size == 0

def good_matches_gt_2(matches):
    return len(matches) > 2 and\
        matches[0].a == 0 and \
        all(matches[i].a == matches[i - 1].a + matches[i - 1].size
            for i in range(1, len(matches) - 1)) and \
        matches[-2].a + matches[-2].size == matches[-1].a

def get_matches(list0, list1):
    return difflib.SequenceMatcher(None, list0, list1).get_matching_blocks()


def write_extags_file(allen_fpath, # read
                      extags_fpath, # write
                      ztz_id_range):


    ztz_to_extractions = read_allen_file(allen_fpath)
    num_sents = len(ztz_to_extractions.keys())
    assert 0 <= ztz_id_range[0] <= ztz_id_range[1] <= num_sents - 1


    with open(extags_fpath, 'w') as f:
        prev_ztz = ''
        top_of_file = True
        ztz_id = -1
        for ztz, ex in ztz_to_extractions:
            ztz_id += 1
            if ztz_id < ztz_id_range[0] or ztz_id > ztz_id_range[1]:
                continue
            if ztz != prev_ztz:
                new_in_ztz = True
                prev_ztz = ztz
                if top_of_file:
                    top_of_file = False
                else:
                    f.write('\n')
            else:
                new_in_ztz = False
            if ex.name_is_tagged["ARG2"] and \
                ex.name_is_tagged["REL"] and\
                    ex.name_is_tagged["ARG1"]:
                if 'REL' in ex.ztz_tags and 'ARG1' in ex.ztz_tags:
                    if (not ex.arg2) or 'ARG2' in ex.ztz_tags:
                        assert len(ex.in3_tokens) == len(ex.ztz_tags)
                        if new_in_ztz:
                            f.write(' '.join(ex.in3_tokens))
                            f.write('\n')
                        f.write(' '.join(ex.ztz_tags))
                        f.write('\n')

def write_extags_ttt_files(allen_fpath,
                           extags_dir,
                           ttt_fractions=(.6, .2, .2)):
    # ttt = train, tune, test
    assert abs(sum(ttt_fractions)-1) < 1e-8

    extags_train_fpath = extags_dir + "/extags_train.txt"
    # dev=development=validation=tuning
    extags_tune_fpath = extags_dir + "/extags_tune.txt"
    extags_test_fpath = extags_dir + "/extags_test.txt"

    num_sents = get_num_sents_in_allen_file(allen_fpath)
    num_train_sents = floor(ttt_fractions[0]*num_sents)
    num_tune_sents = floor(ttt_fractions[1] * num_sents)
    num_test_sents = floor(ttt_fractions[2] * num_sents)
    more = num_sents - num_train_sents - num_tune_sents - num_test_sents
    num_train_sents += more

    for fpath in [extags_train_fpath, extags_tune_fpath, extags_test_fpath]:
        write_extags_file(allen_fpath, fpath)


    return num_train_sents, num_tune_sents, num_test_sents


