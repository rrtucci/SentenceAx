from Params import *

from sax_utils import *
from transformers import AutoTokenizer
from copy import deepcopy
from pprint import pprint

# import spacy
import nltk

nltk.download('popular', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)


class MInput:
    """
    MInput = Model Input
    The main method and work horse of this class is read_input_tags_file().
    That method reads the input data from an extags or a cctags or a
    prediction file (prediction files have one sentence per line, with no
    extractions). It works in all 3 cases!

    data processing chain
    (optional allen_fp->)tags_in_fp->MInput->PaddedMInput->SaxDataset
    ->SaxDataLoaderTool

    In Openie6, Openie6.data.process_data() calls
    Openie6.data._process_data() internally. In SentenceAx, class MInput
    does the job of Openie6.data._process_data(). Classes PaddedMInput,
    SaxDataset and SaxDataLoaderTools do the job of
    Openie6.data.process_data().

    Attributes
    ----------
    REMERGE_TOKENS: bool
        no longer used. Used only with Spacy POS
    auto_tokenizer: AutoTokenizer
    l_orig_sent: list[str]
        list of original (before splitting) sentences
    ll_osent_icode: list[list[int]]
        for each sentence in l_orig_sent, this variable gives a list of
        icodes (i.e., integer codes provided by auto_tokenizer.encode())
    ll_osent_pos_bool: list[list[int]] | None
        for each sentence in l_orig_sent, this variable gives a list of
        booleans (0,1) which indicate POS (part of speech) presence in the
        words in osent (original sentence). Only filled if USE_POS_INFO=True
    ll_osent_pos_loc: list[list[int]] | None
        for each sentence in l_orig_sent, this variable gives a list of 
        integers which indicate POS (part of speech) location relative to 
        the words in osent (original sentence). Only filled if 
        USE_POS_INFO=True     
    ll_osent_verb_bool: list[list[int]] | None
        for each sentence in l_orig_sent, this variable gives a list of
        booleans (0,1) which indicate a verb presence in the words in osent
        (original sentence). Only filled if USE_POS_INFO=True
    ll_osent_verb_loc: list[list[int]] | None
        for each sentence in l_orig_sent, this variable gives a list of
        integers which indicate verb location relative to the words in osent
        (original sentence). Only filled if USE_POS_INFO=True
    ll_osent_wstart_loc: list[list[int]]
        for each sentence in l_orig_sent, this variable gives a list of
        integers for word start locations relative to the list ll_osent_icode
    lll_ilabel: list[list[list[int]]]
        If x is the feature vector and y is the classification, this is
        y. If turned into a tensor, its shape is (num of samples, num of
        extractions (depths), number of words in osentL or osent). The
        entries of this tensor are integers from 0 to 5 (ilabels). This
        variable is only filled for supervised training, ttt="train".
    omit_exless: bool
        set to True iff want osents with no extractions to be skipped
    params: Params
        parameters
    # spacy_model: spacy.Language
    #   No longer used. Openie6 uses both NLTK and Spacy. SentenceAx uses only
    #   NLTK
    tags_in_fp: str
        file path for input file. tags_in_fp is the file path to an extags 
        or a cctags file or a predictions file. If an extags file has no
        extags, only original sentences, then we call it a predictions file.
    verbose: bool

    """
    REMERGE_TOKENS = True

    def __init__(self,
                 params,
                 tags_in_fp,
                 auto_tokenizer,
                 read=True,
                 omit_exless=True,
                 verbose=False):
        """
        Constructor
        
        Parameters
        ----------
        params: Params
        tags_in_fp: str
        auto_tokenizer: AutoTokenizer
        read: bool
            Set this to False iff you don't want it to read tags_in_fp
        verbose: bool
        """
        self.params = params
        self.tags_in_fp = tags_in_fp
        self.auto_tokenizer = auto_tokenizer
        self.omit_exless = omit_exless
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

        # if USE_POS_INFO:
        #     self.spacy_model = spacy.load("en_core_web_sm")
        # spacy usage:
        # doc = spacy_model("This is a text")
        # spacy_model.pipe()
        # spacy_model usually abbreviated as nlp

        self.ll_osent_pos_bool = []  # shape=(num_samples, num_words)
        self.ll_osent_pos_loc = []  # shape=(num_samples, num_words)
        self.ll_osent_verb_bool = []  # shape=(num_samples, num_words)
        self.ll_osent_verb_loc = []  # shape=(num_samples, num_words)

        if read:
            self.read_input_tags_file()

    @staticmethod
    def encode_l_sent(l_sent,
                      auto_tokenizer):
        """
        This static method returns ll_icode. For each sent in l_sent,
        ll_icode gives a list of icodes (i.e., integer codes) obtained via
        auto_tokenizer.encode().

        Note that this method is the inverse of decode_ll_icode()

        Parameters
        ----------
        l_sent: list[sent]
        auto_tokenizer: AutoTokenizer

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
        This static method returns l_sent. For each sent in l_sent, ll_icode
        gives a list of icodes (i.e., integer codes) obtained via
        auto_tokenizer.encode().

        Note that this method is the inverse of encode_l_sent()

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

    # The following commented code is no longer used. It uses Spacy to do
    # POS. It has been replaced by code that uses NLTK instead of Spacy to
    # do POS.

    # def remerge_tokens(self, tokens):
    #     """
    #     similar to Openie6.data.remerge_sent()
    #
    #     apparently, spacy separates `` into ` `, etc.
    #
    #     Parameters
    #     ----------
    #     tokens: spacy.Doc
    #         spacy.Doc is a list of Tokens
    #
    #     Returns
    #     -------
    #     spacy.Doc
    #
    #     """
    #     # merges spacy tokens which are not separated by white-space
    #     # does this recursively until no further changes
    #
    #     changed = True
    #     while changed:
    #         changed = False
    #         i = 0
    #         while i < len(tokens) - 1:
    #             # print("sdfrt", i)
    #             tok = tokens[i]
    #             if not tok.whitespace_:
    #                 next_tok = tokens[i + 1]
    #                 # in-place operation.
    #                 a = tok.i
    #                 b = a + 2
    #                 # b = next_tok.idx + len(next_tok)
    #                 # old
    #                 # tokens[a:b].merge()
    #                 # new
    #                 # print("dfgty***********", a, b, tokens[0:2], len(tokens))
    #
    #                 if a < b <= len(tokens):
    #                     if self.verbose:
    #                         print('\nremerging ', tok, next_tok)
    #                         print(tokens)
    #                     with tokens.retokenize() as retokenizer:
    #                         retokenizer.merge(tokens[a:b])
    #                     if self.verbose:
    #                         print(tokens)
    #                 changed = True
    #             i += 1
    #     return tokens

    # @staticmethod
    # def pos_info(tokens):
    #     """
    #     similar to Openie6.data.pos_tags()
    #
    #     Parameters
    #     ----------
    #     tokens: spacy.Doc
    #
    #     Returns
    #     -------
    #     list[int], list[int], list[str]
    #
    #     """
    #     pos_locs = []
    #     pos_bools = []
    #     pos_words = []
    #     for token_index, token in enumerate(tokens):
    #         if token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
    #             pos_bools.append(1)
    #             pos_locs.append(token_index)
    #             pos_words.append(token.lower_)
    #         else:
    #             pos_bools.append(0)
    #     pos_bools.append(0)
    #     pos_bools.append(0)
    #     pos_bools.append(0)
    #     return pos_locs, pos_bools, pos_words
    #
    # @staticmethod
    # def verb_info(tokens):
    #     """
    #     similar to Openie6.data.verb_tags()
    #
    #     Parameters
    #     ----------
    #     tokens: spacy.Doc
    #
    #     Returns
    #     -------
    #     list[int], list[int], list[str]
    #
    #     """
    #     verb_locs = []
    #     verb_bools = []
    #     verb_words = []
    #     for token_index, token in enumerate(tokens):
    #         if token.pos_ in ['VERB'] and \
    #                 token.lower_ not in LIGHT_VERBS:
    #             verb_bools.append(1)
    #             verb_locs.append(token_index)
    #             verb_words.append(token.lower_)
    #         else:
    #             verb_bools.append(0)
    #     verb_bools.append(0)
    #     verb_bools.append(0)
    #     verb_bools.append(0)
    #     return verb_locs, verb_bools, verb_words

    # def fill_pos_and_verb_info(self):
    #     """
    #
    #     Returns
    #     -------
    #     None
    #
    #     """
    #     self.ll_osent_pos_bool = []
    #     self.ll_osent_pos_loc = []
    #     self.ll_osent_verb_bool = []
    #     self.ll_osent_verb_loc = []
    #     if not USE_POS_INFO or "predict" in self.params.action:
    #         self.ll_osent_pos_bool = None
    #         self.ll_osent_pos_loc = None
    #         self.ll_osent_verb_bool = None
    #         self.ll_osent_verb_loc = None
    #         return
    #     # print("bbght", self.l_orig_sent)
    #     for sent_id, spacy_tokens in enumerate(
    #             self.spacy_model.pipe(self.l_orig_sent,
    #                                   batch_size=10000)):
    #         if REMERGE_TOKENS:
    #             spacy_tokens = self.remerge_tokens(spacy_tokens)
    #
    #             # assert len(self.l_orig_sent[sent_id].split()) == len(
    #             #      spacy_tokens)
    #
    #         pos_locs, pos_bools, pos_words = \
    #             MInput.pos_info(spacy_tokens)
    #         self.ll_osent_pos_bool.append(pos_bools)
    #         self.ll_osent_pos_loc.append(pos_locs)
    #
    #         verb_locs, verb_bools, verb_words = \
    #             MInput.verb_info(spacy_tokens)
    #         self.ll_osent_verb_bool.append(verb_bools)
    #         if verb_locs:
    #             self.ll_osent_verb_loc.append(verb_locs)
    #         else:
    #             self.ll_osent_verb_loc.append([0])

    @staticmethod
    def pos_info(pos_tags):
        """
        similar to Openie6.data.pos_tags()
        
        This method is called by fill_pos_and_verb_info(). It returns the
        variables pos_locs (list of locs of POS words), pos_bools (list of
        bools 0,1 indicating words that are POS), pos_words (list of words
        that are POS). All word locations relative to osent. POS=part of
        speech.

        Parameters
        ----------
        pos_tags: list[tuple(str, str)]
            example (("John", "NOUN"), ("eats", "VERB"))

        Returns
        -------
        list[int], list[int], list[str]

        """
        pos_locs = []
        pos_bools = []
        pos_words = []
        for pos_tag_index, pos_tag in enumerate(pos_tags):
            if pos_tag[1] in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
                pos_bools.append(1)
                pos_locs.append(pos_tag_index)
                pos_words.append(pos_tag[0].lower())
            else:
                pos_bools.append(0)
        pos_bools.append(0)
        pos_bools.append(0)
        pos_bools.append(0)
        return pos_locs, pos_bools, pos_words

    @staticmethod
    def verb_info(pos_tags):
        """
        similar to Openie6.data.verb_tags()
        
        This method is called by fill_pos_and_verb_info(). It returns the
        variables verb_locs (list of verb locs), verb_bools (list of bools
        0,1 related to verbs), verb_words (list of verbs). All word
        locations relative to osent.

        Parameters
        ----------
        pos_tags: list[tuple(str, str)]
            example (("John", "NOUN"), ("eats", "VERB"))

        Returns
        -------
        list[int], list[int], list[str]

        """
        verb_locs = []
        verb_bools = []
        verb_words = []
        for pos_tag_index, pos_tag in enumerate(pos_tags):
            if pos_tag[1] in ['VERB'] and \
                    pos_tag[0].lower() not in LIGHT_VERBS:
                verb_bools.append(1)
                verb_locs.append(pos_tag_index)
                verb_words.append(pos_tag[0].lower())
            else:
                verb_bools.append(0)
        verb_bools.append(0)
        verb_bools.append(0)
        verb_bools.append(0)
        return verb_locs, verb_bools, verb_words

    def fill_pos_and_verb_info(self):
        """
        This method fills the variables: self.ll_osent_pos_bool,
        self.ll_osent_pos_loc, self.ll_osent_verb_bool, self.ll_osent_verb_loc

        It does this by calling pos_info() and verb_info() for each sentence
        in l_orig_sent.

        Returns
        -------
        None

        """
        self.ll_osent_pos_bool = []
        self.ll_osent_pos_loc = []
        self.ll_osent_verb_bool = []
        self.ll_osent_verb_loc = []
        # Openie6 does not use  pos info if 'predict' in hparams.mode
        # but I don't see why
        # if not USE_POS_INFO or "predict" in self.params.action:
        if not USE_POS_INFO:
            self.ll_osent_pos_bool = None
            self.ll_osent_pos_loc = None
            self.ll_osent_verb_bool = None
            self.ll_osent_verb_loc = None
            return
        # print("bbght", self.l_orig_sent)
        for sent_id, sent in enumerate(self.l_orig_sent):
            # important: note that we use pos_tag for sent, not sentL.
            # nlkt and spacy both split "[unused1]" to "[", unused1, "]"
            pos_tags = nltk.pos_tag(get_words(sent, algo="nltk"),
                                    tagset='universal')

            pos_locs, pos_bools, pos_words = \
                MInput.pos_info(pos_tags)
            self.ll_osent_pos_bool.append(pos_bools)
            self.ll_osent_pos_loc.append(pos_locs)

            verb_locs, verb_bools, verb_words = \
                MInput.verb_info(pos_tags)
            self.ll_osent_verb_bool.append(verb_bools)
            if verb_locs:
                self.ll_osent_verb_loc.append(verb_locs)
            else:
                self.ll_osent_verb_loc.append([0])

    def read_input_tags_file(self):
        """
        similar to Openie6.data._process_data()

        This method reads a file of the form

        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        The tags may be extags or cctags (or no tag lines)

        Each original sentence and its tag sequences constitute a new sample.

        The file may have no tag lines, only original sentences, in which
        case it can be used for prediction.

        Returns
        -------
        None

        """
        l_orig_sent = []
        ll_osent_wstart_loc = []  # similar to Openie6.word_starts
        ll_osent_icode = []  # similar to Openie6.input_ids
        lll_ilabel = []  # similar to Openie6.targets, target=extraction
        sentL = ""  # similar to Openie6.sentence

        print("\nMInput started reading '" + self.tags_in_fp + "'")
        print("...")
        with open(self.tags_in_fp, "r", encoding="utf-8") as f:
            lines = f.readlines()

        def is_empty_line_of_sample(line0):
            return not line0 or LINE_SEPARATOR in line0

        def is_osent_line_of_sample(line0):
            if is_empty_line_of_sample(line0):
                return False
            return (not line0.isupper()
                    or has_puntuation(line0, ignored_chs="_"))

        def is_tag_line_of_sample(line0):
            if is_empty_line_of_sample(line0):
                return False
            # ignoring "_" because of CP_START
            return line0.isupper() and \
                not has_puntuation(line0, ignored_chs="_")

        def is_finalization_of_sample(prev_line0, line0):
            if is_empty_line_of_sample(line0) and \
                    not is_empty_line_of_sample(prev_line0):
                return True
            if is_osent_line_of_sample(line0) and \
                    not is_empty_line_of_sample(prev_line0):
                return True
            return False

        prev_line = None
        osent_icodes = []
        ll_ilabel = []
        osent_wstart_locs = []
        num_omitted_sents = 0
        k = 0
        k_osent = 0
        # add empty last sentence so last sentence of file is considered
        for line in lines + [LINE_SEPARATOR]:
            k += 1
            line = line.strip()
            # if line == "":
            #     # this skips blank lines
            #     continue  # skip to next line
            # # print("kklop", line)
            if is_finalization_of_sample(prev_line, line):
                # print("ddft-end", k)
                if not ll_osent_icode:
                    ll_osent_icode = [[0]]  # 0 = PAD
                if len(osentL_words) > MAX_NUM_OSENTL_WORDS or \
                        len(osentL_words) <= 4:
                    num_omitted_sents += 1
                    print(
                        f"{str(num_omitted_sents)}. Line {k_osent} has > "
                        f"{MAX_NUM_OSENTL_WORDS} words."
                        f" length={len(osentL_words)}\n[" +
                        osentL[0:60] + "]")
                    # print("prev_line_rrt", prev_line)
                    # print("line_rrt", line)
                    # print(is_osent_line_of_sample(line))
                    # print(has_puntuation(line,
                    #                      ignored_chs="_",
                    #                      verbose=True))

                elif not ll_ilabel and self.omit_exless:
                    num_omitted_sents += 1
                    print(
                        f"{str(num_omitted_sents)}. Line {k_osent} "
                        "has no valid extractions.")
                else:
                    orig_sent = undoL(osentL)
                    l_orig_sent.append(orig_sent)
                    ll_osent_icode.append(deepcopy(osent_icodes))
                    # print("dfeg", ll_osent_icode)

                    # note that if li=[2,3]
                    # then li[:100] = [2,3]
                    # print("sdftty", ll_ilabel)
                    if not ll_ilabel:
                        # no extractions
                        ll_ilabel = [[0]]
                    lll_ilabel.append(deepcopy(ll_ilabel))
                    ll_osent_wstart_loc.append(deepcopy(osent_wstart_locs))
                    ll_ilabel = []
                    osent_wstart_locs = []
                # } if > MAX_NUM_OSENTL_WORDS words or else
            # } if is_finalization

            if is_osent_line_of_sample(line):
                k_osent = k
                # print("kklop-1st", k, line)
                osentL = line
                if "[unused" not in osentL:
                    # this is useful for predict files, which contain no
                    # extag lines or unused tokens
                    osentL = redoL(osentL)
                osentL_words = get_words(osentL)
                encoding_d = self.auto_tokenizer.batch_encode_plus(
                    osentL_words,
                    add_special_tokens=False  # necessary
                    # additional_special_tokens=UNUSED_TOKENS # refused
                )
                # specified when initialized self.auto_tokenizer
                # add_special_tokens=False,
                # additional_special_tokens=UNUSED_TOKENS
                # but ignored unless repeat it
                # print("encoding_d", e.ncoding_d)
                osent_icodes = [BOS_ICODE]
                osent_wstart_locs = []
                # encoding_d['input_ids'] is a ll_icode
                for icodes in encoding_d['input_ids']:
                    # print("ppokl" , k)
                    # special spacy tokens like \x9c have zero length
                    if len(icodes) == 0:
                        icodes = [100]  # SEP_ICODE = 100
                    # note osent_wstart_locs[0]=1 because first
                    # icodes =[BOS_ICODE]
                    osent_wstart_locs.append(len(osent_icodes))
                    # same as osent_icodes.extend(icodes)
                    osent_icodes += icodes
                osent_icodes.append(EOS_ICODE)
                # print("lmklo", k, str_list(osent_wstart_locs))
                # end of if osent line

            if is_tag_line_of_sample(line):
                # print("sdfrg-tag", k)
                # some tag lines have too many or too few NONE at the end

                # print("lmklo", k, str_list(osent_wstart_locs))
                line_words = get_words(line)
                # print("lmklo1", k, line_words)
                # some line_words shorter than len(osentL_words)
                line_words += ["NONE"] * 15
                max_len = min(len(osentL_words), MAX_NUM_OSENTL_WORDS)
                line_words = line_words[:max_len]
                # print("lmklo2", k, line_words)

                ilabels = [get_tag_to_ilabel(self.params.task)[tag]
                           for tag in line_words]
                # print("mmklo", line_words)
                # print_list("ilabels", ilabels)
                # print("nnmk-line number= " + str(k))

                # print("xxcv", str_list(get_words(osentL)))
                # print("vvbn-line-words", str_list(get_words(line)))
                # assert len(ilabels) == len(osent_wstart_locs), \
                #     f"{str(len(ilabels))} != {str(len(osent_wstart_locs))}"
                if is_valid_label_list(ilabels, self.params.task, "ilabels"):
                    ll_ilabel.append(ilabels)
                # print("dfgthj", ll_ilabel)
                # end of if tag line
            # } if osent line or tag line
            prev_line = line
        # } line loop
        num_samples = len(l_orig_sent)
        print("MInput finished reading '" + self.tags_in_fp + "'")
        print("number of lines= " + str(len(lines)))  # exclude empty line
        print("number of used samples= ", num_samples)
        print("number of omitted samples= ", num_omitted_sents)
        print()

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
        if USE_POS_INFO:
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
    def main1(tags_in_fp,
              pid=1,
              omit_exless=True,
              verbose=False):
        # pid=1, task="ex", action="train_test"
        # pid=5, task="cc", action="train_test"
        params = Params(pid)
        model_str = "bert-base-cased"
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
                      omit_exless=omit_exless,
                      verbose=verbose)
        num_samples = len(m_in.l_orig_sent)
        # print(to_dict(m_in).keys())
        print("num_samples=", num_samples)
        for isam in [0, 1, 2, 3, -2, - 1]:
            print("************** isam=", isam)
            print_list("get_words(l_osentL[isam])",
                       get_words(redoL(m_in.l_orig_sent[isam])))
            print_list("ll_osent_icode[isam]",
                       m_in.ll_osent_icode[isam])
            print_list("ll_osent_pos_loc[isam]",
                       m_in.ll_osent_pos_loc[isam])
            print_list("ll_osent_pos_bool[isam]",
                       m_in.ll_osent_pos_bool[isam])
            print_list("ll_osent_verb_loc[isam]",
                       m_in.ll_osent_verb_loc[isam])
            print_list("ll_osent_verb_bool[isam]",
                       m_in.ll_osent_verb_bool[isam])
            print_list("ll_osent_wstart_loc[isam]",
                       m_in.ll_osent_wstart_loc[isam])
            if verbose:
                print("lll_ilabel[isam]", m_in.lll_ilabel[isam])


    def main2():
        l_sent = [
            "We went to the park on Sunday, and then went to the movies.",
            "I wish i new how to make this program work."]
        model_str = "bert-base-cased"
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


    main1(tags_in_fp="tests/small_extags.txt",
          verbose=False)
    main1(tags_in_fp="tests/small_extagsN.txt",
          verbose=False)
    main2()
    # preds file has no valid exs
    main1(tags_in_fp="predicting/small_pred.txt",
          omit_exless=False,
          verbose=False)

    main1(tags_in_fp="tests/small_cctags.txt",
          pid=5,
          verbose=True)
