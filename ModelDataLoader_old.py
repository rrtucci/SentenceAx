from sax_globals import *
from allen_tool import *
from transformers import AutoTokenizer
import spacy
import torch
from torch.utils.data import DataLoader
import pickle
import os
# use of
# tt.data.Field,
# tt.data.Example
# tt.Dataset
# are deprecated
import torchtext as tt


class ModelDataLoader:

    def __init__(self, params_d):

        self.params_d = params_d

    @staticmethod
    def remerge_sent(tokens):
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
    def pos_mask(tokens):
        pos_mask = []
        pos_indices = []
        pos_words = []
        for token_index, token in enumerate(tokens):
            if token.pos_ in ['ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB']:
                pos_mask.append(1)
                pos_indices.append(token_index)
                pos_words.append(token.lower_)
            else:
                pos_mask.append(0)
        pos_mask.append(0)
        pos_mask.append(0)
        pos_mask.append(0)
        return pos_mask, pos_indices, pos_words

    @staticmethod
    def verb_mask(tokens):
        verb_mask, verb_indices, verb_words = [], [], []
        for token_index, token in enumerate(tokens):
            if token.pos_ in ['VERB'] and \
                    token.lower_ not in LIGHT_VERBS:
                verb_mask.append(1)
                verb_indices.append(token_index)
                verb_words.append(token.lower_)
            else:
                verb_mask.append(0)
        verb_mask.append(0)
        verb_mask.append(0)
        verb_mask.append(0)
        return verb_mask, verb_indices, verb_words


    @staticmethod
    def pad_data(data):
        padded_data_d = {}

        fields = data[0][-1]
        TEXT = fields['text'][1]
        text_list = [example[2].text for example in data]
        padded_data_d['text'] = torch.tensor(TEXT.pad(text_list))

        LABELS = fields['labels'][1]
        labels_list = [example[2].labels for example in data]
        # max_depth = max([len(l) for l in labels_list])
        max_depth = 5
        for i in range(len(labels_list)):
            pad_depth = max_depth - len(labels_list[i])
            num_words = len(labels_list[i][0])
            # print(num_words, pad_depth)
            labels_list[i] = labels_list[i] + [[0] * num_words] * pad_depth
        # print(labels_list)
        padded_data_d['labels'] = torch.tensor(LABELS.pad(labels_list))

        WORD_STARTS = fields['word_starts'][1]
        word_starts_list = [example[2].word_starts for example in data]
        padded_data_d['word_starts'] = \
            torch.tensor(WORD_STARTS.pad(word_starts_list))

        META_DATA = fields['meta_data'][1]
        meta_data_list = [META_DATA.vocab.stoi[example[2].meta_data]
                          for example in data]
        padded_data_d['meta_data'] = \
            torch.tensor(META_DATA.pad(meta_data_list))

        # padded_data_d = {
        #     'text': padded_text,
        #     'labels': padded_labels,
        #     'word_starts': padded_word_starts,
        #     'meta_data': padded_meta_data}

        if 'pos' in fields:
            POS = fields['pos'][1]
            pos_list = [example[2].pos for example in data]
            padded_pos = torch.tensor(POS.pad(pos_list))
            padded_data_d['pos'] = padded_pos

            POS_INDEX = fields['pos_index'][1]
            pos_index_list = [example[2].pos_index for example in data]
            padded_pos_index = torch.tensor(POS_INDEX.pad(pos_index_list))
            padded_data_d['pos_index'] = padded_pos_index

        if 'verb' in fields:
            VERB = fields['verb'][1]
            verb_list = [example[2].verb for example in data]
            padded_verb = torch.tensor(VERB.pad(verb_list))
            padded_data_d['verb'] = padded_verb

            VERB_INDEX = fields['verb_index'][1]
            verb_index_list = [example[2].verb_index for example in data]
            padded_verb_index = torch.tensor(VERB_INDEX.pad(verb_index_list))
            padded_data_d['verb_index'] = padded_verb_index

        return padded_data_d

    def get_examples(self, inp_fp, fields, auto_tokenizer,
                     tag_to_ilabel, spacy_model=None):
        # formerly _process_data()
        """
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

        examples = []  # list[example]
        example_ds = []  # list[example_d]
        ilabels_for_each_ex = []  # a list of a list of ilabels, list[list[in]]
        original_sents = []

        if type(inp_fp) == type([]):
            inp_lines = None
        else:
            inp_lines = open(inp_fp, 'r').readlines()

        prev_line = ""
        for line in inp_lines:
            line = line.strip()
            if '[used' in line:  # it's the  beginning of an example
                sent_plus = line
                encoding = auto_tokenizer.batch_encode_plus(sent_plus.split())
                input_ids = [BOS_TOKEN_ID]
                word_starts = []
                for ids in encoding['input_ids']:
                    # special spacy tokens like \x9c have zero length
                    if len(ids) == 0:
                        ids = [100]
                    word_starts.append(len(input_ids))
                    input_ids += ids  # same as input_ids.extend(ids)
                input_ids.append(EOS_TOKEN_ID)

                original_sent = sent_plus.split('[unused1]')[0].strip()
                original_sents.append(original_sent)

            elif line and '[used' not in line:  # it's a line of tags
                ilabels = [tag_to_ilabel[tag] for tag in line.split()]
                # take away last 3 ids for unused tokens
                ilabels = ilabels[:len(word_starts)]
                ilabels_for_each_ex.append(ilabels)
                prev_line = line
            # last line of file or empty line after example
            # line is either "" or None
            elif len(prev_line) != 0 and not line:
                if len(ilabels_for_each_ex) == 0:
                    ilabels_for_each_ex = [[0]]
                # note that if li=[2,3]
                # then li[:100] = [2,3]
                example_d = {
                    'text': input_ids,
                    'labels': ilabels_for_each_ex[:MAX_EXTRACTION_LENGTH],
                    'word_starts': word_starts,
                    'meta_data': original_sent
                }
                if len(sent_plus.split()) <= 100:
                    example_ds.append(example_d)
                ilabels_for_each_ex = []
                prev_line = line

            else:
                assert False

        # so far, we haven't assumed any spacy derived data nanalysis
        # if spacy is allowed, the example_d can carry more info.
        if spacy_model != None:
            sents = [example_d['meta_data'] for example_d in example_ds]
            for sent_index, spacy_tokens in enumerate(
                    spacy_model.pipe(sents, batch_size=10000)):
                spacy_tokens = ModelDataLoader.remerge_sent(spacy_tokens)
                assert len(sents[sent_index].split()) == len(
                    spacy_tokens)
                example_d = example_ds[sent_index]

                pos, pos_indices, pos_words = \
                    ModelDataLoader.pos_mask(spacy_tokens)
                example_d['pos_index'] = pos_indices
                example_d['pos'] = pos
                verb_mask, verb_indices, verb_words = \
                    ModelDataLoader.verb_mask(spacy_tokens)
                if len(verb_indices) != 0:
                    example_d['verb_index'] = verb_indices
                else:
                    example_d['verb_index'] = [0]
                example_d['verb'] = verb_mask

        # use of tt.Example is deprecated
        for example_d in example_ds:
            example = tt.data.Example.fromdict(example_d, fields)
            examples.append(example)
        return examples, original_sents

    def get_ttt_datasets(self, predict_sentences=None):
        # formerly process_data()
        # this method call AutoTokenizer
        train_fp = self.params_d["train_fp"]
        dev_fp = self.params_d["dev_fp"]
        test_fp = self.params_d["test_fp"]

        do_lower_case = 'uncased' in self.params_d["mode"]l_str
        auto_tokenizer = AutoTokenizer.from_pretrained(
            self.params_d["mode"]l_str,
            do_lower_case=do_lower_case,
            use_fast=True,
            data_dir='data/pretrained_cache',
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)

        spacy_model = spacy.load("en_core_web_sm")
        # spacy usage:
        # doc = spacy_model("This is a text")
        # spacy_model.pipe()
        # spacy_model usually abbreviated as nlp
        pad_id = auto_tokenizer.convert_tokens_to_ids(
            auto_tokenizer.pad_token)

        TEXT = tt.data.Field(use_vocab=False, batch_first=True,
                             pad_token=pad_id)
        WORD_STARTS = tt.data.Field(use_vocab=False, batch_first=True,
                                    pad_token=0)
        POS = tt.data.Field(use_vocab=False, batch_first=True, pad_token=0)
        POS_INDEX = tt.data.Field(use_vocab=False, batch_first=True,
                                  pad_token=0)
        VERB = tt.data.Field(use_vocab=False, batch_first=True, pad_token=0)
        VERB_INDEX = tt.data.Field(use_vocab=False, batch_first=True,
                                   pad_token=0)
        META_DATA = tt.data.Field(sequential=False)
        VERB_WORDS = tt.data.Field(sequential=False)
        POS_WORDS = tt.data.Field(sequential=False)
        LABELS = tt.data.NestedField(
            tt.data.Field(use_vocab=False, batch_first=True, pad_token=-100),
            use_vocab=False)

        fields = {'text': ('text', TEXT),
                  'labels': ('labels', LABELS),
                  'word_starts': ('word_starts', WORD_STARTS),
                  'meta_data': ('meta_data', META_DATA)}
        if 'predict' not in self.params_d["mode"]:
            fields['pos'] = ('pos', POS)
            fields['pos_index'] = ('pos_index', POS_INDEX)
            fields['verb'] = ('verb', VERB)
            fields['verb_index'] = ('verb_index', VERB_INDEX)

        if self.params_d["task"] == "ex":
            tag_to_ilabel = EXTAG_TO_ILABEL
        elif self.params_d["task"] == "cc":
            tag_to_ilabel = CCTAG_TO_ILABEL
        else:
            assert False

        model_str = self.params_d["mode"]l_str.replace("/", "_")
        cached_train_fp = f'{train_fp}.{model_str}.pkl'
        cached_dev_fp = f'{dev_fp}.{model_str}.pkl'
        cached_test_fp = f'{test_fp}.{model_str}.pkl'

        original_sents = []
        if 'predict' in self.params_d["mode"]:
            # no caching used in predict mode
            if predict_sentences == None:  # predict
                if self.params_d["inp"] != None:
                    predict_f = open(self.params_d["inp"], 'r')
                else:
                    predict_f = open(self.params_d["predict_fp"], 'r')
                predict_lines = predict_f.readlines()
                fullstops = []
                predict_sentences = []
                for line in predict_lines:
                    # Normalize the quotes - similar to that in training data
                    line = line.replace('’', '\'')
                    line = line.replace('”', '\'\'')
                    line = line.replace('“', '\'\'')

                    # tokenized_line = line.split()

                    # Why use both nltk and spacy to word tokenize
                    # get_ttt_datasets() uses nltk.word_tokenize()
                    # get_examples() uses spacy_model.pipe(sents...)
                    # get_examples() uses transformers.AutoTokenizer

                    tokenized_line = ' '.join(nltk.word_tokenize(line))
                    predict_sentences.append(
                        tokenized_line + UNUSED_TOKENS_STR)
                    predict_sentences.append('\n')

            # this use of get_examples() is wrong
            # get_examples()
            # returns: examples, original_sents
            predict_examples, original_sents = \
                self.get_examples(predict_fp,
                                  fields,
                                  auto_tokenizer,
                                  tag_to_ilabel,
                                  spacy_model=None)
            META_DATA.build_vocab(
                tt.data.Dataset(predict_examples, fields=fields.values()))

            predict_dataset = [
                (len(example.text), idx, example, fields)
                for idx, example in enumerate(predict_examples)]
            train_dataset, dev_dataset, test_dataset = \
                predict_dataset, predict_dataset, predict_dataset
        else:
            if not os.path.exists(
                    cached_train_fp) or self.params_d["build_cache"]:
                train_examples, _ = self.get_examples(train_fp,
                                                      fields,
                                                      auto_tokenizer,
                                                      tag_to_ilabel,
                                                      spacy_model)
                pickle.dump(train_examples, open(cached_train_fp, 'wb'))
            else:
                train_examples = pickle.load(open(cached_train_fp, 'rb'))

            if not os.path.exists(cached_dev_fp) or self.params_d["build_cache"]:
                dev_examples, _ = self.get_examples(dev_fp,
                                                    fields,
                                                    auto_tokenizer,
                                                    tag_to_ilabel,
                                                    spacy_model)
                pickle.dump(dev_examples, open(cached_dev_fp, 'wb'))
            else:
                dev_examples = pickle.load(open(cached_dev_fp, 'rb'))

            if not os.path.exists(cached_test_fp) or self.params_d["build_cache"]:
                test_examples, _ = self.get_examples(test_fp,
                                                     fields,
                                                     auto_tokenizer,
                                                     tag_to_ilabel,
                                                     spacy_model)
                pickle.dump(test_examples, open(cached_test_fp, 'wb'))
            else:
                test_examples = pickle.load(open(cached_test_fp, 'rb'))

            META_DATA.build_vocab(
                tt.data.Dataset(train_examples,
                                fields=fields.values()),
                tt.data.Dataset(dev_examples, fields=fields.values()),
                tt.data.Dataset(test_examples, fields=fields.values()))

            train_dataset = [(len(example.text), idx, example, fields) for
                             idx, example in enumerate(train_examples)]
            dev_dataset = [(len(example.text), idx, example, fields) for
                           idx, example in enumerate(dev_examples)]
            test_dataset = [(len(example.text), idx, example, fields) for
                            idx, example in enumerate(test_examples)]
            train_dataset.sort()  # to simulate bucket sort (along with pad_data)

        return train_dataset, dev_dataset, test_dataset, \
            META_DATA.vocab, original_sents

    def get_ttt_dataloaders(self, type, predict_sentences=None):
        train_dataset, val_dataset, test_dataset, \
            meta_data_vocab, original_sents = self.get_ttt_datasets(
            predict_sentences)
        # this method calls DataLoader

        if type == "train":
            return DataLoader(train_dataset,
                              batch_size=self.params_d["batch_size"],
                              collate_fn=self.pad_data,
                              shuffle=True,
                              num_workers=1)
        elif type == "val":
            return DataLoader(val_dataset,
                              batch_size=self.params_d["batch_size"],
                              collate_fn=self.pad_data,
                              num_workers=1)
        elif type == "test":
            return DataLoader(test_dataset,
                              batch_size=self.params_d["batch_size"],
                              collate_fn=self.pad_data,
                              num_workers=1)
        else:
            assert False