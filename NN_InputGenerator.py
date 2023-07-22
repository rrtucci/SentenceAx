from my_globals import *


# Refs:
# https://spacy.io/usage/spacy-101/

class NN_InputGenerator:

    def __init__(self):

        self.simple_to_complex_sents= None # analogous to conj_mapping
        self.set_simple_to_complex_sents_dict()

    def set_simple_to_complex_sents_dict(self):
        simple_to_complex_sents = {}
        content = open(EXT_SAMPLES_PATH).read()
        complex_ztz = ''
        for sample in content.split('\n\n'):
            for i, line in enumerate(sample.strip('\n').split('\n')):
                if i == 0:
                    complex_ztz = line
                else:
                    simple_to_complex_sents[line] = complex_ztz
        return simple_to_complex_sents


    def set_train_tune_test_datasets(self):