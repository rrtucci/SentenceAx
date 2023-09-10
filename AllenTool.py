import re
from unidecode import unidecode
from collections import OrderedDict
from SaxExtraction import *


class AllenTool:
    """
    
    Hercule Poirot is a fictional Belgian detective , created by Agatha Christie .	<arg1> Hercule Poirot </arg1> <rel> is </rel> <arg2> a fictional Belgian detective , created by Agatha Christie </arg2>	0.95
    Hercule Poirot is a fictional Belgian detective , created by Agatha Christie .	<arg1> Hercule Poirot </arg1> <rel> is </rel> <arg2> a fictional Belgian detective </arg2>	-108.0506591796875
    Hercule Poirot is a fictional Belgian detective , created by Agatha Christie .	<arg1> a fictional Belgian detective </arg1> <rel> created </rel> <arg2> by Agatha Christie </arg2>	0.92
    Hercule Poirot is a fictional Belgian detective , created by Agatha Christie .	<arg1> a fictional Belgian detective </arg1> <rel> be created </rel> <arg2> by Agatha Christie </arg2>	-108.0506591796875
    Hercule Poirot is a fictional Belgian detective , created by Agatha Christie .	<arg1> Hercule Poirot </arg1> <rel> is </rel> <arg2> a fictional Belgian detective created by Agatha Christie </arg2>	-108.0506591796875

    No blank lines between examples.
    All lines start with original sent, without unused string.
    
    """

    def __init__(self, allen_fp):
        self.allen_fp = allen_fp
        # ex =extraction sent=sentence
        self.sentL_to_exs = self.read_allen_file()
        #print("mklop", self.sentL_to_exs)
        self.num_sents = len(self.sentL_to_exs)
        # print("lkop", self.num_sents)

    @staticmethod
    def get_ex_from_allen_line(line):
        tab_sep_vals = line.strip().split('\t')
        in_sent = tab_sep_vals[0]
        score = float(tab_sep_vals[2])
        # if len(tab_sep_vals) == 4:
        #     num_exs = int(tab_sep_vals[3])
        # else:
        #     num_exs = None

        assert len(re.findall("<arg1>.*</arg1>", tab_sep_vals[1])) == 1
        assert len(re.findall("<rel>.*</rel>", tab_sep_vals[1])) == 1
        assert len(re.findall("<arg2>.*</arg2>", tab_sep_vals[1])) == 1

        parts = []
        for str0 in ['arg1', 'rel', 'arg2']:
            begin_tag = '<' + str0 + '>'
            end_tag = '</' + str0 + '>'
            part = re.findall(begin_tag + '.*' + end_tag, tab_sep_vals[1])[0]
            # print("vcbgh", part)
            part = ' '.join(get_words(part.strip(begin_tag).strip(end_tag)))
            parts.append(part)
        ex = SaxExtraction(in_sent, parts[0], parts[1], parts[2], score)

        return ex

    @staticmethod
    def get_allen_line_from_ex(ex):
        str0 = ex.orig_sentL + "\t"
        str0 += " <arg1> " + ex.arg1 + r" <\arg1> "
        str0 += " <rel> " + ex.pred + r" <\rel> "
        str0 += " <arg2> " + ex.arg2 + r" <\arg2> \t"
        str0 += str(ex.score)
        return str0

    def read_allen_file(self):
        with open(self.allen_fp, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [unidecode(line) for line in lines]
        sentL_to_exs = OrderedDict()
        exs = []
        prev_in_sentL = ''
        for line in lines:
            # print("bnhk", line)
            ex = AllenTool.get_ex_from_allen_line(line)
            if (prev_in_sentL and ex.orig_sentL != prev_in_sentL) :
                sentL_to_exs[prev_in_sentL] = exs
                exs = []
            exs.append(ex)
            prev_in_sentL = ex.orig_sentL
        # last sample
        sentL_to_exs[prev_in_sentL] = exs
        # print("zlpd", sentL_to_exs)
        return sentL_to_exs

    def write_allen_alternative_file(self,
                                     out_fp,
                                     first_sample_id=1,
                                     last_sample_id=-1,
                                     ftype="ex",
                                     numbered=False):
        """
        * extags (openie-data\openie4_labels)
        2.
        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE
        
        sample id 1-based
        
        Parameters
        ----------


        Returns
        -------

        """

        if last_sample_id != -1:
            assert 1 <= first_sample_id <= last_sample_id <= self.num_sents

        with open(out_fp, 'w', encoding='utf-8') as f:
            sample_id = 0
            num_sams = 0
            for sent, l_ex in self.sentL_to_exs.items():
                sample_id += 1
                if sample_id < first_sample_id:
                    continue
                if last_sample_id>0 and sample_id> last_sample_id:
                    continue
                if numbered:
                    f.write(str(sample_id) + ".\n")
                num_sams += 1
                f.write(sent + "\n")
                for ex in l_ex:
                    if ftype == "ex": # extags file
                        ex.set_extags()
                        f.write(" ".join(ex.extags)+ "\n")
                    elif ftype == "ss": # simple sentences file
                        f.write(ex.get_simple_sent() + "\n")
                    else:
                        assert False
            print("finished file " + out_fp + ":    "+
                  str(num_sams) + " samples.")

                # if ex.name_is_tagged["ARG1"] and \
                #         ex.name_is_tagged["REL"] and \
                #         ex.name_is_tagged["ARG2"]:
                #     if 'ARG1' in ex.sent_tags and 'REL' in ex.sent_tags:
                #         if (not ex.arg2) or 'ARG2' in ex.sent_tags:
                #             f.write(ex.orig_sentL + "\n")
                #             if new_in_sent:
                #                 f.write(' '.join(ex.in3_tokens))
                #                 f.write('\n')
                #             f.write(' '.join(ex.sent_tags))
                #             f.write('\n')

    def write_extags_ttt_files(self,
                               out_dir,
                               ttt_fractions=(.6, .2, .2)):
        """
        ttt = train, val, test
        tuning=dev=development=validation

        Parameters
        ----------
        out_dir
        ttt_fractions

        Returns
        -------

        """
        num_train_sents, num_val_sents, num_test_sents = \
            get_num_ttt_sents(self.num_sents, ttt_fractions)

        train_first_last = (
            1,
            num_train_sents)
        val_first_last = (
            num_train_sents+1,
            num_train_sents + num_val_sents)
        test_first_last = (
            num_train_sents + num_val_sents+1,
            num_train_sents + num_val_sents + num_test_sents)

        # print("ghjy", train_first_last, val_first_last, test_first_last)

        train_fp = out_dir + "/extags_train.txt"
        val_fp = out_dir + "/extags_val.txt"
        test_fp = out_dir + "/extags_test.txt"

        self.write_allen_alternative_file(train_fp, *train_first_last)
        self.write_allen_alternative_file(val_fp, *val_first_last)
        self.write_allen_alternative_file(test_fp, *test_first_last)


if __name__ == "__main__":
    def main1(ftype):
        allen_fp = "testing_files/small_allen.tsv"
        if ftype=="ex":
            out_fp = "testing_files/small_extags.txt"
        elif ftype=="ss":
            out_fp = "testing_files/small_simple_sents.txt"
        else:
            assert False
        at = AllenTool(allen_fp)
        at.write_allen_alternative_file(out_fp,
                                        first_sample_id=1,
                                        last_sample_id=-1,
                                        ftype=ftype,
                                        numbered=True)
    def main2():
        allen_fp = "testing_files/small_allen.tsv"
        out_dir = "testing_files"
        at = AllenTool(allen_fp)
        at.write_extags_ttt_files(out_dir)



    # main1("ss")
    main1("ex")
    #main2()