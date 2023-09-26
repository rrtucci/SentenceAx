from sax_globals import *
import spacy
from sax_utils import *
from transformers import AutoTokenizer


class MInput:
    """

    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    l_orig_sent: list[str]
    l_osent_ilabels: list[list[int]]
    l_osent_pos_locs: list[list[int]]
    l_osent_pos_mask: list[list[int]]
    l_osent_verb_locs: list[list[int]]
    l_osent_verb_mask: list[list[int]]
    l_osent_word_start_locs: list[list[int]]
    lll_ilabel: list[list[list[int]]]
    num_samples: int
    spacy_model: spacy.Language
    task: str
    use_spacy_model: bool

    """

    def __init__(self, in_fp, task, auto_tokenizer, use_spacy_model):
        """

        Parameters
        ----------
        in_fp: str
        task: str
        auto_tokenizer: AutoTokenizer
        use_spacy_model: bool
        """
        self.task = task
        self.auto_tokenizer = auto_tokenizer
        self.use_spacy_model = use_spacy_model

        self.num_samples = None
        # shape=(num_samples,)
        self.l_orig_sent = []
        # shape=(num_samples, max_depth=num_ex, num_ilabels)
        self.lll_ilabel = []

        # following lists have
        # shape is (num_samples, encoding length =100)
        # it's not (num_samples, max_depth=num_ex)
        # each word of orig_sent may be encoded with more than one ilabel
        # os = original sentence
        self.l_osent_word_start_locs = []  # shape=(num_samples, encoding len)
        self.l_osent_ilabels = []  # shape=(num_samples, encoding len

        self.use_spacy_model = use_spacy_model
        if self.use_spacy_model:
            self.spacy_model = spacy.load("en_core_web_sm")
        # spacy usage:
        # doc = spacy_model("This is a text")
        # spacy_model.pipe()
        # spacy_model usually abbreviated as nlp

        self.l_osent_pos_mask = []  # shape=(num_samples, num_words)
        self.l_osent_pos_locs = []  # shape=(num_samples, num_words)
        self.l_osent_verb_mask = []  # shape=(num_samples, num_words)
        self.l_osent_verb_locs = []  # shape=(num_samples, num_words)

        self.read_input_extags_file(in_fp)

    @staticmethod
    def remerge_sent(tokens):
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
                tok = tokens[i]
                if not tok.whitespace_:
                    next_tok = tokens[i + 1]
                    # in-place operation.
                    tokens.merge(tok.idx,
                                 next_tok.idx + len(next_tok))
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
        pos_mask = []
        pos_words = []
        for token_index, token in enumerate(tokens):
            if token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
                pos_mask.append(1)
                pos_locs.append(token_index)
                pos_words.append(token.lower_)
            else:
                pos_mask.append(0)
        pos_mask.append(0)
        pos_mask.append(0)
        pos_mask.append(0)
        return pos_locs, pos_mask, pos_words

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
        verb_mask = []
        verb_words = []
        for token_index, token in enumerate(tokens):
            if token.pos_ in ['VERB'] and \
                    token.lower_ not in LIGHT_VERBS:
                verb_mask.append(1)
                verb_locs.append(token_index)
                verb_words.append(token.lower_)
            else:
                verb_mask.append(0)
        verb_mask.append(0)
        verb_mask.append(0)
        verb_mask.append(0)
        return verb_locs, verb_mask, verb_words

    def fill_pos_and_verb_info(self):
        """

        Returns
        -------
        None

        """
        l_osent_pos_mask = []
        l_osent_pos_locs = []
        l_osent_verb_mask = []
        l_osent_verb_locs = []
        if not self.use_spacy_model:
            return
        for sent_id, spacy_tokens in enumerate(
                self.spacy_model.pipe(self.l_orig_sent, batch_size=10000)):
            spacy_tokens = MInput.remerge_sent(spacy_tokens)
            assert len(self.l_orig_sent[sent_id].split()) == len(
                spacy_tokens)

            pos_locs, pos_mask, pos_words = \
                MInput.pos_info(spacy_tokens)
            l_osent_pos_mask.append(pos_mask)
            l_osent_pos_locs.append(pos_locs)

            verb_locs, verb_mask, verb_words = \
                MInput.verb_info(spacy_tokens)
            l_osent_verb_mask.append(verb_mask)
            if verb_locs:
                l_osent_verb_locs.append(verb_locs)
            else:
                l_osent_verb_locs.append([0])

        self.l_osent_pos_mask = l_osent_pos_mask
        self.l_osent_pos_locs = l_osent_pos_locs
        self.l_osent_verb_mask = l_osent_verb_mask
        self.l_osent_verb_locs = l_osent_verb_locs

    def read_input_extags_file(self, in_fp):
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


        Parameters
        ----------
        in_fp: str

        Returns
        -------
        None

        """
        l_orig_sent = []
        l_osent_word_start_locs = []  # similar to word_starts
        l_osent_ilabels = []  # similar to input_ids
        ll_ex_ilabels = []  # similar to targets target=extraction
        sentL = None  # similar to `sentence`

        with open(in_fp, "r") as f:
            lines = f.readlines()

        def is_first_line_of_sample(line0):
            return '[used' in line0

        def is_tag_line_of_sample(line0):
            return len(line0) != 0 and '[used' not in line0

        def is_end_of_sample(prev_line0, line0):
            return not line0 or \
                (prev_line0 and is_first_line_of_sample(line0))

        prev_line = None
        for line in lines:
            line = line.strip()
            if line == "":
                # this skips blank lines
                continue  # skip to next line

            if is_first_line_of_sample(line):
                sentL = line
                encoding_d = self.auto_tokenizer.batch_encode_plus(
                    sentL.split())
                os_ilabels = [BOS_ILABEL]
                os_word_start_locs = []
                # encoding_d['input_ids'] is a ll_ilabel
                for l_ilabel0 in encoding_d['input_ids']:
                    # special spacy tokens like \x9c have zero length
                    if len(l_ilabel0) == 0:
                        l_ilabel0 = [100]
                    # note os_word_start_locs[0]=1 because first
                    # ilabels =[BOS_ILABEL]
                    os_word_start_locs.append(len(os_ilabels))
                    # same as ilabels.extend(l_ilabel0)
                    os_ilabels += l_ilabel0
                os_ilabels.append(EOS_ILABEL)
                assert len(sentL.split()) == len(os_word_start_locs)
                l_ex_ilabels = []
            elif is_tag_line_of_sample(line):
                ex_ilabels = [EXTAG_TO_ILABEL[tag] for tag in line.split()]
                assert ex_ilabels == len(os_word_start_locs)
                l_ex_ilabels.append(ex_ilabels)
            else:
                assert False
            if is_end_of_sample(prev_line, line):
                if len(l_osent_ilabels) == 0:
                    l_osent_ilabels = [[0]]

                if len(sentL.split()) <= 100:
                    l_osent_ilabels.append(os_ilabels)
                    orig_sent = unL(sentL)
                    l_orig_sent.append(orig_sent)

                    # note that if li=[2,3]
                    # then li[:100] = [2,3]
                    ll_ex_ilabels.append(l_ex_ilabels)
                    l_osent_word_start_locs.append(os_word_start_locs)

                os_ilabels = []
                os_word_start_locs = []
                l_ex_ilabels = []
            prev_line = line
        self.l_orig_sent = l_orig_sent
        self.lll_ilabel = ll_ex_ilabels

        self.l_osent_ilabels = l_osent_ilabels
        self.l_osent_word_start_locs = l_osent_word_start_locs

        # so far, we haven't assumed any spacy derived data nanalysis
        # if spacy is allowed, the example_d can carry more info.
        if self.spacy_model:
            self.fill_pos_and_verb_info()

        # example_d = {
        #     'l_ilabel': l_ilabel,
        #     'll_label': labels_for_each_ex[:MAX_EX_DEPTH],
        #     'l_word_start_loc': l_word_start_loc,
        #     'orig_sent': orig_sent,
        #     # if spacy_model:
        #     'pos_mask': pos_mask,
        #     'pos_locs': pos_locs,
        #     'verb_mask': verb_mask,
        #     'verb_locs': verb_locs
        # }

        # use of tt.data.Example is deprecated
        # for example_d in l_sample_d:
        #     example = tt.data.Example.fromdict(example_d, fields)
        #     examples.append(example)
        # return examples, orig_sents

if __name__ == "__main__":
    def main():
        in_fp = "testing_files/extags_test.txt"
        task = "test"
        model_str = "bert-base-uncased"
        auto_tokenizer = AutoTokenizer.from_pretrained(
            model_str,
            do_lower_case=True,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        use_spacy_model = True
        m_imput = MInput(in_fp, task, auto_tokenizer, use_spacy_model)

    main()