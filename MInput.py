from sax_globals import *
import spacy

class MInput:
    def __init__(self, task, auto_tokenizer, use_spacy_model):
        self.task = task
        self.auto_tokenizer = auto_tokenizer
        self.use_spacy_model = use_spacy_model
        
        self.num_samples = None
        self.l_sample = None

        self.l_orig_sent = []
        self.lll_ilabel = []
        self.ll_starting_word_loc = []
        self.ll_sentL_id = []

        self.use_spacy_model = use_spacy_model
        if self.use_spacy_model:
            self.spacy_model = spacy.load("en_core_web_sm")
        # spacy usage:
        # doc = spacy_model("This is a text")
        # spacy_model.pipe()
        # spacy_model usually abbreviated as nlp
        self.l_pos_mask = []
        self.l_pos_locs = []
        self.l_verb_mask = []
        self.l_verb_locs = []
        

    def absorb_l_orig_sent(self, l_orig_sent):
        for k, orig_sent in enumerate(l_orig_sent):
            self.l_sample[k].orig_sent = orig_sent

    def absorb_lll_ilabel(self, lll_ilabel):
        for sample_id, ll_ilabel in enumerate(lll_ilabel):
            self.l_sample[sample_id].absorb_children(ll_ilabel)

    def absorb_all_possible(self):
        self.num_samples = len(self.lll_ilabel)
        self.absorb_l_orig_sent(self.l_orig_sent)
        self.absorb_lll_ilabel(self.lll_ilabel)

    @staticmethod
    def remerge_sent(tokens):
        """
        similar to data.remerge_sent()

        Parameters
        ----------
        tokens

        Returns
        -------

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
        similar to data.pos_tags()

        Parameters
        ----------
        tokens

        Returns
        -------

        """
        pos_mask = []
        pos_locs = []
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
        return pos_mask, pos_locs, pos_words

    @staticmethod
    def verb_info(tokens):
        """
        similar to data.verb_tags()

        Parameters
        ----------
        tokens

        Returns
        -------

        """
        verb_mask, verb_locs, verb_words = [], [], []
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
        return verb_mask, verb_locs, verb_words

    def fill_pos_and_verb_info(self):
        if not self.use_spacy_model:
            return
        for sent_id, spacy_tokens in enumerate(
                self.spacy_model.pipe(self.l_orig_sent, batch_size=10000)):
            spacy_tokens = MInput.remerge_sent(spacy_tokens)
            assert len(self.l_orig_sent[sent_id].split()) == len(
                spacy_tokens)

            pos_mask, pos_locs, pos_words = \
                MInput.pos_info(spacy_tokens)
            self.l_pos_mask.append(pos_mask)
            self.l_pos_locs.append(pos_locs)

            verb_mask, verb_locs, verb_words = \
                MInput.verb_info(spacy_tokens)
            self.l_verb_mask.append(verb_mask)
            if verb_locs:
                self.l_verb_locs.append(verb_locs)
            else:
                self.l_verb_locs.append([0])

    def absorb_allen_input_file(self, in_fp):
        """
        similar to data._process_data()


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
        """

        ilabels_for_each_ex = []  # a list of a list of labels, list[list[in]]
        orig_sents = []

        if type(in_fp) == type([]):
            in_lines = None
        else:
            with(in_fp, "r") as f:
                in_lines = f.readlines()

        prev_line = ""
        for line in in_lines:
            line = line.strip()
            if '[used' in line:  # it's the  beginning of an example
                sentL = line
                encoding = self.auto_tokenizer.batch_encode_plus(
                    sentL.split())
                sentL_ids = [BOS_ILABEL]
                l_starting_word_loc = []
                for ids in encoding['input_ids']:
                    # special spacy tokens like \x9c have zero length
                    if len(ids) == 0:
                        ids = [100]
                    l_starting_word_loc.append(len(sentL_ids))
                    sentL_ids += ids  # same as sentL_ids.extend(ids)
                sentL_ids.append(EOS_ILABEL)

                orig_sent = sentL.split('[unused1]')[0].strip()
                orig_sents.append(orig_sent)

            elif line and '[used' not in line:  # it's a line of tags
                ilabels = [TAG_TO_ILABEL[tag] for tag in line.split()]
                # take away last 3 ids for unused tokens
                ilabels = ilabels[:len(l_starting_word_loc)]
                ilabels_for_each_ex.append(ilabels)
                prev_line = line
            # last line of file or empty line after example
            # line is either "" or None
            elif len(prev_line) != 0 and not line:
                if len(ilabels_for_each_ex) == 0:
                    ilabels_for_each_ex = [[0]]
                # note that if li=[2,3]
                # then li[:100] = [2,3]

                if len(sentL.split()) <= 100:
                    self.ll_sentL_id.append(sentL_ids)
                    self.lll_ilabel.append(
                        ilabels_for_each_ex[:MAX_EX_DEPTH])
                    self.ll_starting_word_loc.append(l_starting_word_loc)
                    self.l_orig_sent = orig_sent
                ilabels_for_each_ex = []
                prev_line = line

            else:
                assert False

        # so far, we haven't assumed any spacy derived data nanalysis
        # if spacy is allowed, the example_d can carry more info.
        if self.spacy_model:
            self.fill_pos_and_verb_info()

        # example_d = {
        #     'sentL_ids': sentL_ids,
        #     'll_label': labels_for_each_ex[:MAX_EX_DEPTH],
        #     'l_starting_word_loc': l_starting_word_loc,
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
        
        self.num_samples = len(self.lll_ilabel)
