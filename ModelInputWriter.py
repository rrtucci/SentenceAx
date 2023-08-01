from my_globals import *
from allen_tool import *
from transformers import AutoTokenizer
import spacy
import nltk


# Refs:
# https://spacy.io/usage/spacy-101/

class ModelInputWriter:

    def __init__(self, params_d):

        self.params_d = params_d

    def get_examples(self, inp_fp, fields, tokenizer,
                     label_dict, spacy_model=None):
        # formerly _process_data()
        model_str = self.params_d.model_str
        examples, exampleDs, targets, lang_targets, orig_sentences = \
            [], [], [], [], []

        sentence = None
        max_extraction_length = 5

        if type(inp_fp) == type([]):
            inp_lines = inp_fp
        else:
            inp_lines = open(inp_fp, 'r').readlines()

        new_example = True
        for line_num, line in tqdm(enumerate(inp_lines)):
            line = line.strip()
            if line == '':
                new_example = True

            if '[unused' in line or new_example:
                if sentence is not None:
                    if len(targets) == 0:
                        targets = [[0]]
                        lang_targets = [[0]]
                    orig_sentence = sentence.split('[unused1]')[0].strip()
                    orig_sentences.append(orig_sentence)

                    exampleD = {'text': input_ids,
                                'labels': targets[:max_extraction_length],
                                'word_starts': word_starts,
                                'meta_data': orig_sentence}
                    if len(sentence.split()) <= 100:
                        exampleDs.append(exampleD)

                    targets = []
                    sentence = None
                # starting new example
                if line is not '':
                    new_example = False
                    sentence = line

                    tokenized_words = tokenizer.batch_encode_plus(
                        sentence.split())
                    input_ids, word_starts, lang = [
                        self.params_d.bos_token_id], [], []
                    for tokens in tokenized_words['input_ids']:
                        if len(tokens) == 0:  # special spacy_tokens like \x9c
                            tokens = [100]
                        word_starts.append(len(input_ids))
                        input_ids.extend(tokens)
                    input_ids.append(self.params_d.eos_token_id)
                    assert len(sentence.split()) == len(
                        word_starts), ipdb.set_trace()
            else:
                if sentence is not None:
                    target = [label_dict[i] for i in line.split()]
                    target = target[:len(word_starts)]
                    assert len(target) == len(word_starts), ipdb.set_trace()
                    targets.append(target)

        if spacy_model != None:
            sentences = [ed['meta_data'] for ed in exampleDs]
            for sentence_index, spacy_sentence in tqdm(enumerate(
                    spacy_model.pipe(sentences, batch_size=10000))):
                spacy_sentence = remerge_sent(spacy_sentence)
                assert len(sentences[sentence_index].split()) == len(
                    spacy_sentence), ipdb.set_trace()
                exampleD = exampleDs[sentence_index]

                pos, pos_indices, pos_words = pos_tags(spacy_sentence)
                exampleD['pos_index'] = pos_indices
                exampleD['pos'] = pos
                verb, verb_indices, verb_words = verb_tags(spacy_sentence)
                if len(verb_indices) != 0:
                    exampleD['verb_index'] = verb_indices
                else:
                    exampleD['verb_index'] = [0]
                exampleD['verb'] = verb

        for exampleD in exampleDs:
            example = data.Example.fromdict(exampleD, fields)
            examples.append(example)
        return examples, orig_sentences

    def get_ttt_datasets(self, predict_sentences=None):
        # formerly precess_data()
        train_fp, dev_fp, test_fp = \
            self.params_d.train_fp, self.params_d.dev_fp, self.params_d.test_fp
        self.params_d.bos_token_id, self.params_d.eos_token_id = 101, 102

        do_lower_case = 'uncased' in self.params_d.model_str
        tokenizer = AutoTokenizer.from_pretrained(self.params_d.model_str,
                                                  do_lower_case=do_lower_case,
                                                  use_fast=True,
                                                  data_dir='data/pretrained_cache',
                                                  add_special_tokens=False,
                                                  additional_special_tokens=
                                                  ['[unused1]', '[unused2]',
                                                   '[unused3]'])

        nlp = spacy.load("en_core_web_sm")
        pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        TEXT = data.Field(use_vocab=False, batch_first=True,
                          pad_token=pad_index)
        WORD_STARTS = data.Field(use_vocab=False, batch_first=True,
                                 pad_token=0)
        POS = data.Field(use_vocab=False, batch_first=True, pad_token=0)
        POS_INDEX = data.Field(use_vocab=False, batch_first=True, pad_token=0)
        VERB = data.Field(use_vocab=False, batch_first=True, pad_token=0)
        VERB_INDEX = data.Field(use_vocab=False, batch_first=True, pad_token=0)
        META_DATA = data.Field(sequential=False)
        VERB_WORDS = data.Field(sequential=False)
        POS_WORDS = data.Field(sequential=False)
        LABELS = data.NestedField(
            data.Field(use_vocab=False, batch_first=True, pad_token=-100),
            use_vocab=False)

        fields = {'text': ('text', TEXT), 'labels': ('labels', LABELS),
                  'word_starts': (
                      'word_starts', WORD_STARTS),
                  'meta_data': ('meta_data', META_DATA)}
        if 'predict' not in self.params_d.mode:
            fields['pos'] = ('pos', POS)
            fields['pos_index'] = ('pos_index', POS_INDEX)
            fields['verb'] = ('verb', VERB)
            fields['verb_index'] = ('verb_index', VERB_INDEX)

        if self.params_d.task == 'oie':
            label_dict = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                          'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}
        else:  # self.params_d.task == 'conj':
            label_dict = {'CP_START': 2, 'CP': 1,
                          'CC': 3, 'SEP': 4, 'OTHERS': 5, 'NONE': 0}

        cached_train_fp, cached_dev_fp, cached_test_fp = \
            f'{train_fp}.{self.params_d.model_str.replace("/", "_")}.pkl', \
                f'{dev_fp}.{self.params_d.model_str.replace("/", "_")}.pkl', \
                f'{test_fp}.{self.params_d.model_str.replace("/", "_")}.pkl'

        all_sentences = []
        if 'predict' in self.params_d.mode:
            # no caching used in predict mode
            if predict_sentences == None:  # predict
                if self.params_d.inp != None:
                    predict_f = open(self.params_d.inp, 'r')
                else:
                    predict_f = open(self.params_d.predict_fp, 'r')
                predict_lines = predict_f.readlines()
                fullstops = []
                predict_sentences = []
                for line in predict_lines:
                    # Normalize the quotes - similar to that in training data
                    line = line.replace('’', '\'')
                    line = line.replace('”', '\'\'')
                    line = line.replace('“', '\'\'')

                    # tokenized_line = line.split()
                    tokenized_line = ' '.join(nltk.word_tokenize(line))
                    predict_sentences.append(
                        tokenized_line + ' [unused1] [unused2] [unused3]')
                    predict_sentences.append('\n')

            predict_examples, all_sentences = \
                self.get_examples(predict_sentences, fields,
                                  tokenizer, label_dict, None)
            META_DATA.build_vocab(
                data.Dataset(predict_examples, fields=fields.values()))

            predict_dataset = [(len(ex.text), idx, ex, fields) for idx,
            ex in enumerate(predict_examples)]
            train_dataset, dev_dataset, test_dataset = \
                predict_dataset, predict_dataset, predict_dataset
        else:
            if not os.path.exists(
                    cached_train_fp) or self.params_d.build_cache:
                train_examples, _ = self.get_examples(train_fp,
                                                      fields, tokenizer,
                                                      label_dict, nlp)
                pickle.dump(train_examples, open(cached_train_fp, 'wb'))
            else:
                train_examples = pickle.load(open(cached_train_fp, 'rb'))

            if not os.path.exists(cached_dev_fp) or self.params_d.build_cache:
                dev_examples, _ = self.get_examples(dev_fp, fields,
                                                    tokenizer,
                                                    label_dict, nlp)
                pickle.dump(dev_examples, open(cached_dev_fp, 'wb'))
            else:
                dev_examples = pickle.load(open(cached_dev_fp, 'rb'))

            if not os.path.exists(cached_test_fp) or self.params_d.build_cache:
                test_examples, _ = self.get_examples(test_fp,
                                                     fields,
                                                     tokenizer, label_dict,
                                                     nlp)
                pickle.dump(test_examples, open(cached_test_fp, 'wb'))
            else:
                test_examples = pickle.load(open(cached_test_fp, 'rb'))

            META_DATA.build_vocab(data.Dataset(train_examples,
                                               fields=fields.values()),
                                  data.Dataset(
                                      dev_examples, fields=fields.values()),
                                  data.Dataset(test_examples,
                                               fields=fields.values()))

            train_dataset = [(len(ex.text), idx, ex, fields) for
                             idx, ex in enumerate(train_examples)]
            dev_dataset = [(len(ex.text), idx, ex, fields) for
                           idx, ex in enumerate(dev_examples)]
            test_dataset = [(len(ex.text), idx, ex, fields) for
                            idx, ex in enumerate(test_examples)]
            train_dataset.sort()  # to simulate bucket sort (along with pad_data)

        return train_dataset, dev_dataset, test_dataset, \
            META_DATA.vocab, all_sentences

