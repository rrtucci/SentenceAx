from AllenTool import *
from math import floor

class ExTagsFileWriter:
    """
    * extags (openie-data\openie4_labels)
    Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
    ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
    NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE


    """

    def __init__(self,
                 allen_fp):
        """


        Parameters
        ----------
        allen_fp
        ld_output
        source
        """
        self.allen_fp = allen_fp
        at = AllenTool(allen_fp)
        self.sent_to_extractions = at.read_allen_file()
        self.num_sents = len(self.sent_to_extractions)

    @staticmethod
    def write_extags_file_from_predictions(ld_output,
                                           l_sentL, # original sentences
                                           ll_sent_loc):
        """
        formerly run.get_labels()
        LABEL_TO_EXTAG={0: 'NONE', 1: 'ARG1', 2: 'REL', 3: 'ARG2',
                 4: 'ARG2', 5: 'NONE'}
        output_d= {
            "meta_data":
            "ground_truth":
            "loss":
            "predictions":
            "scores":
        }


        Parameters
        ----------
        l_sentL
        ll_sent_loc

        Returns
        -------

        """

        lines = []
        sample_id, ex_id, word_id = 0, 0, 0

        for i in range(0, len(ll_sent_loc)):
            if len(ll_sent_loc[i]) == 0:
                words = l_sentL[i].split('[unused1]')[0].strip().split()
                ll_sent_loc[i].append(list(range(len(words))))

            lines.append(
                '\n' + l_sentL[i].split('[unused1]')[0].strip())
            for j in range(0, len(ll_sent_loc[i])):
                assert len(ll_sent_loc[i][j]) == len(
                    ld_output[sample_id]['meta_data'][ex_id].
                    strip().split())
                sentL = ld_output[sample_id]['meta_data'][
                               ex_id].strip() + UNUSED_TOKENS_STR
                assert sentL == l_sentL[i]
                ll_pred_label = ld_output[sample_id]['predictions'][
                    ex_id]

                for pred_labels in ll_pred_label:
                    # You can use x.item() to get a Python number
                    # from a torch tensor that has one element
                    if pred_labels.sum().item() == 0:
                        break

                    labels = [0] * len(sentL.strip().split())
                    pred_labels = pred_labels[:len(sentL.split())].tolist()
                    for k, loc in enumerate(
                            sorted(ll_sent_loc[i][j])):
                        labels[loc] = pred_labels[k]

                    labels = labels[:-3]
                    # 1: arg1, 2: rel
                    if 1 not in pred_labels and 2 not in pred_labels:
                        continue

                    str_extags = \
                        ' '.join([LABEL_TO_EXTAG[i] for i in labels])
                    lines.append(str_extags)

                word_id += 1
                ex_id += 1
                if ex_id == len(ld_output[sample_id]['meta_data']):
                    ex_id = 0
                    sample_id += 1

        lines.append('\n')
        return lines

    def write_extags_file_from_allen_file(self,
                                          out_fp,
                                          sent_id_range):
        assert 0 <= sent_id_range[0] <= sent_id_range[1] <= self.num_sents - 1

        with open(out_fp, 'w') as f:
            prev_sent = ''
            top_of_file = True
            sent_id = -1
            for sent, ex in self.sent_to_extractions:
                sent_id += 1
                if sent_id < sent_id_range[0] or sent_id >= sent_id_range[1]:
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
        num_train_sents, num_tune_sents, num_test_sents = \
            get_num_ttt_sents(self.num_sents, ttt_fractions)

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
