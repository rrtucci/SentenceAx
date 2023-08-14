from AllenTool import *
from math import floor

class ExTagsFileWriter:
    """
    * extags (openie-data\openie4_labels)
    Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
    ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
    NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

    l_output_d = {
        "meta_data":
        "ground_truth":
        "loss":
        "predictions":
        "scores":
    }


    """

    def __init__(self,
                 source,
                 allen_fp=None,
                 l_output_d=None):
        """


        Parameters
        ----------
        allen_fp
        l_output_d
        source
        """
        self.source = source
        assert source in ["allen", "predictions"]
        one_hot = (bool(allen_fp), bool(l_output_d))
        assert one_hot[0] + one_hot[1] == 1
        
        self.allen_fp = None
        self.l_output_d = None
        self.sent_to_extractions = None
        if source == "allen":
            self.allen_fp = allen_fp
            self.sent_to_extractions = read_allen_file(allen_fp)
        elif source == "predictions":
            self.l_output_d = l_output_d

    def get_sentences(self):
        assert self.source == "allen"
        return self.sent_to_extractions.keys()

    def get_num_sents(self):
        assert self.source == "allen"
        return len(self.sent_to_extractions.keys())

    def write_extags_file_from_predictions(self,
                                           out_fp,
                                           sentences,
                                           orig_sentences,
                                           l_sent_spanned_locs):
        """
        formerly run.get_labels()
        EXTAG_TO_ILABEL = {'NONE': 0, 'ARG1': 1, 'REL': 2, 'ARG2': 3,
                   'LOC': 4, 'TIME': 4, 'TYPE': 5, 'ARGS': 3}

        Parameters
        ----------
        model
        sentences
        orig_sentences
        l_sent_spanned_locs

        Returns
        -------

        """
        assert self.source == "predictions"

        lines = []
        idx1, idx2, idx3 = 0, 0, 0
        count = 0
        prev_orig_sentence = ''

        for i in range(0, len(l_sent_spanned_locs)):
            if len(l_sent_spanned_locs[i]) == 0:
                sentence = orig_sentences[i].split('[unused1]')[
                    0].strip().split()
                l_sent_spanned_locs[i].append(list(range(len(sentence))))

            lines.append(
                '\n' + orig_sentences[i].split('[unused1]')[0].strip())
            for j in range(0, len(l_sent_spanned_locs[i])):
                assert len(l_sent_spanned_locs[i][j]) == len(
                    self.l_output_d[idx1]['meta_data'][idx2].
                    strip().split())
                sentence = self.l_output_d[idx1]['meta_data'][
                               idx2].strip() + UNUSED_TOKENS_STR
                assert sentence == sentences[idx3]
                orig_sentence = orig_sentences[i]
                predictions = self.l_output_d[idx1]['predictions'][idx2]

                all_extractions = []
                all_str_extags = []
                len_exts = []
                for prediction in predictions:
                    if prediction.sum().item() == 0:
                        break

                    extags = [0] * len(orig_sentence.strip().split())
                    prediction = prediction[:len(sentence.split())].tolist()
                    for idx, value in enumerate(
                            sorted(l_sent_spanned_locs[i][j])):
                        extags[value] = prediction[idx]

                    extags = extags[:-3]
                    if 1 not in prediction and 2 not in prediction:
                        continue

                    str_ilabels = \
                        ' '.join([EXTAG_TO_ILABEL[x] for x in extags])
                    lines.append(str_ilabels)

                idx3 += 1
                idx2 += 1
                if idx2 == len(self.l_output_d[idx1]['meta_data']):
                    idx2 = 0
                    idx1 += 1

        lines.append('\n')
        return lines

    def write_extags_file_from_allen_file(self,
                                          out_fp,
                                          sent_id_range):
        assert self.source == "allen"
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
        assert self.source == "allen"
        num_sents = self.get_num_sents()
        num_train_sents, num_tune_sents, num_test_sents = \
            get_num_ttt_sents(num_sents, ttt_fractions)

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

        self.write_extags_file_from_allen_file(train_fp, train_range)
        self.write_extags_file_from_allen_file(tune_fp, tune_range)
        self.write_extags_file_from_allen_file(test_fp, test_range)
