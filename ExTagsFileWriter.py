from allen_tool import *
from math import floor


class ExTagsFileWriter:

    def __init__(self,
                 allen_fp):
        """
        formerly data.data_processing()


        Parameters
        ----------
        allen_fp
        ttt_fractions
        """
        self.allen_fp = allen_fp

        self.sent_to_extractions = read_allen_file(allen_fp)

    def get_sentences(self):
        return self.sent_to_extractions.keys()

    def get_num_sents(self):
        return len(self.sent_to_extractions.keys())

    def get_extags(self, model, sentences,
                   orig_sentences, sentence_indices_list):
        """
        formerly run.get_labels()

        Parameters
        ----------
        model
        sentences
        orig_sentences
        sentence_indices_list

        Returns
        -------

        """

        lines = []
        outputs = model.outputs
        idx1, idx2, idx3 = 0, 0, 0
        count = 0
        prev_orig_sentence = ''

        for i in range(0, len(sentence_indices_list)):
            if len(sentence_indices_list[i]) == 0:
                sentence = orig_sentences[i].split('[unused1]')[
                    0].strip().split()
                sentence_indices_list[i].append(list(range(len(sentence))))

            lines.append(
                '\n' + orig_sentences[i].split('[unused1]')[0].strip())
            for j in range(0, len(sentence_indices_list[i])):
                assert len(sentence_indices_list[i][j]) == len(
                    outputs[idx1]['meta_data'][
                        idx2].strip().split())
                sentence = outputs[idx1]['meta_data'][
                               idx2].strip() + UNUSED_TOKENS_STR
                assert sentence == sentences[idx3]
                orig_sentence = orig_sentences[i]
                predictions = outputs[idx1]['predictions'][idx2]

                all_extractions, all_str_labels, len_exts = [], [], []
                for prediction in predictions:
                    if prediction.sum().item() == 0:
                        break

                    labels = [0] * len(orig_sentence.strip().split())
                    prediction = prediction[:len(sentence.split())].tolist()
                    for idx, value in enumerate(
                            sorted(sentence_indices_list[i][j])):
                        labels[value] = prediction[idx]

                    labels = labels[:-3]
                    if 1 not in prediction and 2 not in prediction:
                        continue

                    str_labels = \
                        ' '.join([EXTAG_TO_ILABEL[x] for x in labels])
                    lines.append(str_labels)

                idx3 += 1
                idx2 += 1
                if idx2 == len(outputs[idx1]['meta_data']):
                    idx2 = 0
                    idx1 += 1

        lines.append('\n')
        return lines

    def write_extags_file(self,
                          out_fp,
                          sent_id_range):

        num_sents = self.get_num_sents()
        assert 0 <= sent_id_range[0] <= sent_id_range[1] <= num_sents - 1

        with open(out_fp, 'w') as f:
            prev_sent = ''
            top_of_file = True
            sent_id = -1
            for sent, ex in self.sent_to_extractions:
                sent_id += 1
                if sent_id < sent_id_range[0] or sent_id > sent_id_range[1]:
                    continue
                if sent != prev_sent:
                    new_in_sent = True
                    prev_sent = sent
                    if top_of_file:
                        top_of_file = False
                    else:
                        f.write('\n')
                else:
                    new_in_sent = False
                if ex.name_is_tagged["ARG2"] and \
                        ex.name_is_tagged["REL"] and \
                        ex.name_is_tagged["ARG1"]:
                    if 'REL' in ex.sent_tags and 'ARG1' in ex.sent_tags:
                        if (not ex.arg2) or 'ARG2' in ex.sent_tags:
                            assert len(ex.in3_tokens) == len(ex.sent_tags)
                            if new_in_sent:
                                f.write(' '.join(ex.in3_tokens))
                                f.write('\n')
                            f.write(' '.join(ex.sent_tags))
                            f.write('\n')

    def write_extags_ttt_files(self,
                               out_dir,
                               ttt_fractions=(.6, .2, .2)):
        """
        ttt = train, tune, test
        tuning=dev=development=validation

        Parameters
        ----------
        out_dir
        ttt_fractions

        Returns
        -------

        """
        assert abs(sum(ttt_fractions) - 1) < 1e-8

        def get_num_ttt_sents():
            num_sents = self.get_num_sents()
            num_train_sents = floor(ttt_fractions[0] * num_sents)
            num_tune_sents = floor(ttt_fractions[1] * num_sents)
            num_test_sents = floor(ttt_fractions[2] * num_sents)
            num_extra_sents = num_sents - num_train_sents - \
                              num_tune_sents - num_test_sents
            num_train_sents += num_extra_sents
            return num_train_sents, num_tune_sents, num_test_sents

        num_train_sents, num_tune_sents, num_test_sents = \
            get_num_ttt_sents()
        train_range = range(0, num_train_sents)
        tune_range = range(
            num_train_sents,
            num_train_sents + num_tune_sents)
        test_range = range(
            num_train_sents + num_tune_sents,
            num_train_sents + num_tune_sents + num_test_sents)

        train_fp = out_dir + "/extags_train.txt"
        tune_fp = out_dir + "/extags_tune.txt"
        test_fp = out_dir + "/extags_test.txt"

        self.write_extags_file(train_fp, train_range)
        self.write_extags_file(tune_fp, tune_range)
        self.write_extags_file(test_fp, test_range)
