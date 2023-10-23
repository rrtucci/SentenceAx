from Params import *
import spacy
from sax_utils import *
from transformers import AutoTokenizer
from copy import deepcopy
from pprint import pprint


class MInput:
    """
    data processing chain
    tags_in_fp->MInput->PaddedMInput->SaxDataSet->DataLoader

    In Openie6, Openie6.data.process_data() calls
    Openie6.data._process_data() internally. In SentenceAx, class MInput
    does the job of Openie6.data._process_data() and classes PaddedMInput,
    SaxDataSet and DataLoaderTools do the job of Openie6.data.process_data().

    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    l_orig_sent: list[str]
    ll_osent_icode: list[list[int]]
    ll_osent_pos_bool: list[list[int]]
    ll_osent_pos_loc: list[list[int]]
    ll_osent_verb_bool: list[list[int]]
    ll_osent_verb_loc: list[list[int]]
    ll_osent_wstart_loc: list[list[int]]
    lll_ilabel: list[list[list[int]]]
    params: Params
    spacy_model: spacy.Language
    tags_in_fp: str
    verbose: bool

    """

    def __init__(self,
                 params,
                 tags_in_fp,
                 auto_tokenizer,
                 read=True,
                 verbose=False):
        """
        tags_in_fp is an extags or a cctags file.

        if the extags file has no extags, only original sentences, then
        we can use tags_in_fp as for prediction.

        Parameters
        ----------
        params: Params
        tags_in_fp: str
        auto_tokenizer: AutoTokenizer
        read: bool
        verbose: bool
        """
        self.params = params
        self.tags_in_fp = tags_in_fp
        self.auto_tokenizer = auto_tokenizer
        self.verbose = verbose

        # shape=(num_samples,)
        self.l_orig_sent = []
        # shape=(num_samples, num_depths=num_ex, num_ilabels)
        self.lll_ilabel = []

        # following lists have
        # shape is (num_samples, encoding length =100)
        # it's not (num_samples, num_depths=num_ex)
        # each word of orig_sent may be encoded with more than one ilabel
        # os = original sentence
        self.ll_osent_wstart_loc = []  # shape=(num_samples, encoding len)
        self.ll_osent_icode = []  # shape=(num_samples, encoding len

        if USE_SPACY_MODEL:
            self.spacy_model = spacy.load("en_core_web_sm")
        # spacy usage:
        # doc = spacy_model("This is a text")
        # spacy_model.pipe()
        # spacy_model usually abbreviated as nlp

        self.ll_osent_pos_bool = []  # shape=(num_samples, num_words)
        self.ll_osent_pos_loc = []  # shape=(num_samples, num_words)
        self.ll_osent_verb_bool = []  # shape=(num_samples, num_words)
        self.ll_osent_verb_loc = []  # shape=(num_samples, num_words)

        if read:
            self.read_input_tags_file(tags_in_fp)

    @staticmethod
    def encode_l_sent(l_sent,
                      auto_tokenizer,
                      add=True):
        """
        Parameters
        ----------
        l_sent: list[sent]
        auto_tokenizer: AutoTokenizer
        add: bool

        Returns
        -------
        list[list[int]]

        """
        encode = auto_tokenizer.encode
        ll_icode = []
        for sent in l_sent:
            l_icode = encode(sent)
            ll_icode.append(l_icode)
        return ll_icode

    @staticmethod
    def decode_ll_icode(ll_icode,
                        auto_tokenizer):
        """

        Parameters
        ----------
        ll_icode: list[list[int]]
        auto_tokenizer: AutoTokenizer

        Returns
        -------
        list[str]

        """
        decode = auto_tokenizer.decode
        l_sent = []
        for l_icode in ll_icode:
            sent = decode(l_icode)
            l_sent.append(sent)
        return l_sent

    def remerge_tokens(self, tokens):
        """
        similar to Openie6.data.remerge_sent()

        Parameters
        ----------
        tokens: spacy.Doc
            spacy.Doc is a list of Tokens

        Returns
        -------
        spacy.Doc

        """
        # merges spacy tokens which are not separated by white-space
        # does this recursively until no further changes

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(tokens) - 1:
                # print("sdfrt", i)
                tok = tokens[i]
                if not tok.whitespace_:
                    next_tok = tokens[i + 1]
                    # in-place operation.
                    a = tok.i
                    b = a + 2
                    # b = next_tok.idx + len(next_tok)
                    # old
                    # tokens[a:b].merge()
                    # new
                    # print("dfgty***********", a, b, tokens[0:2], len(tokens))

                    if a < b <= len(tokens):
                        if self.verbose:
                            print('\nremerging ', tok, next_tok)
                            print(tokens)
                        with tokens.retokenize() as retokenizer:
                            retokenizer.merge(tokens[a:b])
                        if self.verbose:
                            print(tokens)
                    changed = True
                i += 1
        return tokens

    @staticmethod
    def pos_info(tokens):
        """
        similar to Openie6.data.pos_tags()

        Parameters
        ----------
        tokens: spacy.Doc

        Returns
        -------
        list[int], list[int], list[str]

        """
        pos_locs = []
        pos_bools = []
        pos_words = []
        for token_index, token in enumerate(tokens):
            if token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
                pos_bools.append(1)
                pos_locs.append(token_index)
                pos_words.append(token.lower_)
            else:
                pos_bools.append(0)
        pos_bools.append(0)
        pos_bools.append(0)
        pos_bools.append(0)
        return pos_locs, pos_bools, pos_words

    @staticmethod
    def verb_info(tokens):
        """
        similar to Openie6.data.verb_tags()

        Parameters
        ----------
        tokens: spacy.Doc

        Returns
        -------
        list[int], list[int], list[str]

        """
        verb_locs = []
        verb_bools = []
        verb_words = []
        for token_index, token in enumerate(tokens):
            if token.pos_ in ['VERB'] and \
                    token.lower_ not in LIGHT_VERBS:
                verb_bools.append(1)
                verb_locs.append(token_index)
                verb_words.append(token.lower_)
            else:
                verb_bools.append(0)
        verb_bools.append(0)
        verb_bools.append(0)
        verb_bools.append(0)
        return verb_locs, verb_bools, verb_words

    def fill_pos_and_verb_info(self):
        """

        Returns
        -------
        None

        """
        self.ll_osent_pos_bool = []
        self.ll_osent_pos_loc = []
        self.ll_osent_verb_bool = []
        self.ll_osent_verb_loc = []
        if not USE_SPACY_MODEL or "predict" in self.params.mode:
            self.ll_osent_pos_bool = [[]]
            self.ll_osent_pos_loc = [[]]
            self.ll_osent_verb_bool = [[]]
            self.ll_osent_verb_loc = [[]]
            return
        # print("bbght", self.l_orig_sent)
        for sent_id, spacy_tokens in enumerate(
                self.spacy_model.pipe(self.l_orig_sent,
                                      batch_size=10000)):
            spacy_tokens = self.remerge_tokens(spacy_tokens)
            # assert len(self.l_orig_sent[sent_id].split()) == len(
            #     spacy_tokens)

            pos_locs, pos_bools, pos_words = \
                MInput.pos_info(spacy_tokens)
            self.ll_osent_pos_bool.append(pos_bools)
            self.ll_osent_pos_loc.append(pos_locs)

            verb_locs, verb_bools, verb_words = \
                MInput.verb_info(spacy_tokens)
            self.ll_osent_verb_bool.append(verb_bools)
            if verb_locs:
                self.ll_osent_verb_loc.append(verb_locs)
            else:
                self.ll_osent_verb_loc.append([0])

    def read_input_tags_file(self, tags_in_fp):
        """
        similar to Openie6.data._process_data()



        this reads a file of the form

        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        the tags may be extags or cctags

        each original sentence and its tag sequences constitute a new example

        The file may have no tag lines, only original sentences, in which
        case it can be used for prediction.

        Parameters
        ----------
        tags_in_fp: str

        Returns
        -------
        None

        """
        l_orig_sent = []
        ll_osent_wstart_loc = []  # similar to word_starts
        ll_osent_icode = []  # similar to input_ids
        lll_ilabel = []  # similar to targets, target=extraction
        sentL = ""  # similar to `sentence`

        print("\nMInput started reading '" + tags_in_fp + "'")
        with open(tags_in_fp, "r", encoding="utf-8") as f:
            lines = f.readlines()

        def is_beginning_of_sample(line0):
            return line0 and (not line0.isupper()
                              or has_puntuation(line0, ignored_chs="_"))

        def is_tag_line_of_sample(line0):
            # ignoring "_" because of CP_START
            return line0 and line0.isupper() \
                and not has_puntuation(line0, ignored_chs="_")

        def is_ending_of_sample(k, prev_line0, line0):
            if not line0:
                return True
            if prev_line0 and is_beginning_of_sample(line0):
                return True
            return False

        prev_line = None
        osent_icodes = []
        ll_ilabel = []
        osent_wstart_locs = []
        num_omitted_sents = 0
        k = 0
        for line in lines:
            k += 1
            line = line.strip()
            if line == "":
                # this skips blank lines
                continue  # skip to next line
            # print("kklop", line)
            if is_beginning_of_sample(line):
                # print("kklop-1st", k, line)
                sentL = line
                encoding_d = self.auto_tokenizer.batch_encode_plus(
                    get_words(sentL),
                    add_special_tokens=False)
                # print("encoding_d", e.ncoding_d)
                osent_icodes = [BOS_ICODE]
                osent_wstart_locs = []
                # encoding_d['input_ids'] is a ll_icode
                for icodes in encoding_d['input_ids']:
                    # print("ppokl" , k)
                    # special spacy tokens like \x9c have zero length
                    if len(icodes) == 0:
                        icodes = [100]
                    # note osent_wstart_locs[0]=1 because first
                    # icodes =[BOS_ICODE]
                    osent_wstart_locs.append(len(osent_icodes))
                    # same as osent_icodes.extend(icodes)
                    osent_icodes += icodes
                osent_icodes.append(EOS_ICODE)
                # print("lmki", sentL)
                # print("lmklo", osent_wstart_locs)

            elif is_tag_line_of_sample(line):
                # print("sdfrg-tag", k)

                ilabels = [get_tag_to_ilabel(self.params.task)[tag]
                           for tag in get_words(line)]
                # print("nnmk-line number= " + str(k))
                # assert len(ilabels) == len(osent_wstart_locs)
                ll_ilabel.append(ilabels)
                # print("dfgthj", ll_ilabel)
            else:
                pass
            if is_ending_of_sample(k, prev_line, line):
                # print("ddft-end", k)
                if len(ll_osent_icode) == 0:
                    ll_osent_icode = [[0]]

                sentL_words = get_words(sentL)
                if sentL and len(sentL_words) > 100:
                    num_omitted_sents += 1
                    print(str(num_omitted_sents) +
                          f". The {k}'th line has more than 100 words."
                          f" length={len(sentL_words)}\n[" + sentL[0:60] + "]")
                    # print("prev_line_rrt", prev_line)
                    # print("line_rrt", line)
                    # print(is_beginning_of_sample(line))
                    # print(has_puntuation(line,
                    #                      ignored_chs="_",
                    #                      verbose=True))
                else:
                    ll_osent_icode.append(deepcopy(osent_icodes))
                    # print("dfeg", ll_osent_icode)
                    orig_sent = undoL(sentL)
                    l_orig_sent.append(orig_sent)

                    # note that if li=[2,3]
                    # then li[:100] = [2,3]
                    # print("sdftty", ll_ilabel)
                    if not ll_ilabel:
                        ll_ilabel = [[0]]
                    lll_ilabel.append(deepcopy(ll_ilabel))
                    ll_osent_wstart_loc.append(deepcopy(osent_wstart_locs))
                osent_icodes = []
                ll_ilabel = []
                osent_wstart_locs = []

            prev_line = line

        num_samples = len(l_orig_sent)
        print()
        print("MInput finished reading '" + tags_in_fp + "'")
        print("number of lines= " + str(k))
        print("number of used samples= ", num_samples)
        print("number of omitted samples= ", num_omitted_sents)

        self.l_orig_sent = l_orig_sent
        # ll_osent_icode add extra term [0] at beginning
        self.ll_osent_icode = ll_osent_icode[1:]
        self.lll_ilabel = lll_ilabel
        self.ll_osent_wstart_loc = ll_osent_wstart_loc

        def check_len(li):
            for x in li:
                assert len(x) == num_samples

        check_len([self.ll_osent_icode,
                   self.lll_ilabel,
                   self.ll_osent_wstart_loc])

        # so far, we haven't assumed any spacy derived data nanalysis
        # if spacy is allowed, the example_d can carry more info.
        if self.spacy_model:
            self.fill_pos_and_verb_info()

        # example_d = {
        #     'l_ilabel': l_ilabel,
        #     'll_label': labels_for_each_ex[:EX_NUM_DEPTHS],
        #     'l_word_start_loc': l_word_start_loc,
        #     'orig_sent': orig_sent,
        #     # if spacy_model:
        #     'pos_bools': pos_bools,
        #     'pos_locs': pos_locs,
        #     'verb_bools': verb_bools,
        #     'verb_locs': verb_locs
        # }

        # use of tt.data.Example is deprecated
        # for example_d in l_sample_d:
        #     example = tt.data.Example.fromdict(example_d, fields)
        #     examples.append(example)
        # return examples, orig_sents


if __name__ == "__main__":
    def main1(tags_in_fp, verbose):
        params = Params(1)  # 1, task="ex", mode="train_test"
        model_str = "bert-base-uncased"
        auto_tokenizer = AutoTokenizer.from_pretrained(
            model_str,
            do_lower_case=True,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        m_in = MInput(params,
                      tags_in_fp,
                      auto_tokenizer,
                      verbose=verbose)
        num_samples = len(m_in.l_orig_sent)
        print(to_dict(m_in).keys())
        for k in range(min(num_samples, 6)):
            print("************** k=", k)
            print("num_samples=", num_samples)
            print("get_words(l_osentL[k])=",
                  get_words(redoL(m_in.l_orig_sent[k])))
            print("ll_osent_icode[k]=\n", m_in.ll_osent_icode[k])
            print("ll_osent_pos_loc[k]=\n", m_in.ll_osent_pos_loc[k])
            print("ll_osent_pos_bool[k]=\n", m_in.ll_osent_pos_bool[k])
            print("ll_osent_verb_loc[k]=\n", m_in.ll_osent_verb_loc[k])
            print("ll_osent_verb_bool[k]=\n", m_in.ll_osent_verb_bool[k])
            print("ll_osent_wstart_loc[k]=\n", m_in.ll_osent_wstart_loc[k])
            if verbose:
                print("lll_ilabel=\n", m_in.lll_ilabel)


    def main2():
        l_sent = [
            "We went to the park on Sunday, and then went to the movies.",
            "I wish i new how to make this program work."]
        model_str = "bert-base-uncased"
        do_lower_case = ('uncased' in model_str)
        auto_tokenizer = AutoTokenizer.from_pretrained(
            model_str,
            do_lower_case=do_lower_case,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        ll_icode = MInput.encode_l_sent(l_sent,
                                        auto_tokenizer)
        l_sent2 = MInput.decode_ll_icode(ll_icode,
                                         auto_tokenizer)
        pprint(l_sent)
        print(ll_icode)
        pprint(l_sent2)




    main1(tags_in_fp="tests/extags_test.txt",
          verbose=False)
    main2()
    main1(tags_in_fp="predictions/small_pred.txt",
          verbose=True)
