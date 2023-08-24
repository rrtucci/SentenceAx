import re
from unidecode import unidecode
from collections import OrderedDict
from SAXExtraction import *

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
        self.sent_to_extractions = self.read_allen_file()
        self.num_sents = len(self.sent_to_extractions)
    
    @staticmethod
    def write_allen_line(ex):
        str0 = ex.orig_sentL_pair[0]
        str0 += " <arg1> " + ex.arg1 + r" <\arg1> "
        str0 += " <rel> " + ex.pred + r" <\rel> "
        str0 += " <arg2> " + ex.arg2 + r" <\arg2> "
        str0 += str(ex.confidence)
        return str0
    
    @staticmethod
    def read_allen_line(line):
        tab_sep_vals = line.strip().split('\t')
        in_sent = tab_sep_vals[0]
        confidence = float(tab_sep_vals[2])
        # if len(tab_sep_vals) == 4:
        #     num_extractions = int(tab_sep_vals[3])
        # else:
        #     num_extractions = None
    
        assert len(re.findall("<arg1>.*</arg1>", tab_sep_vals[1])) == 1
        assert len(re.findall("<rel>.*</rel>", tab_sep_vals[1])) == 1
        assert len(re.findall("<arg2>.*</arg2>", tab_sep_vals[1])) == 1
    
        parts = []
        for str0 in ['arg1', 'rel', 'arg2']:
            begin_tag, end_tag = '<' + str0 + '>', '</' + str0+ '>'
            part = re.findall(begin_tag + '.*' + end_tag, tab_sep_vals[1])[0]
            # print("vcbgh", part)
            part = ' '.join(get_words(part.strip(begin_tag).strip(end_tag)))
            parts.append(part)
        ex = SAXExtraction(in_sent, parts[0], parts[1], parts[2], confidence)
    
        return ex
       
    def read_allen_file(self):
        with open(self.allen_fp, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [unidecode(line) for line in lines]
        sent_to_extractions = OrderedDict()
        extractions = []
        prev_in_sent = ''
        for line in lines:
            # print("bnhk", line)
            ex = AllenTool.read_allen_line(line)
            extractions.append(ex)
            # print("ooyp1", ex.orig_sentL_pair[0])
            # print("ooyp2", prev_in_sent)
            # print("ooyp3", ex.orig_sentL_pair[0] == prev_in_sent)
            if ex.orig_sentL_pair[0] != prev_in_sent:
                if prev_in_sent:
                    sent_to_extractions[prev_in_sent] = extractions[:-1]
                    # print("llkml", prev_in_sent, extractions)
                    prev_in_sent = ex.orig_sentL_pair[0]
                    extractions = [ex]
                else:
                    prev_in_sent = ex.orig_sentL_pair[0]
    
        # print("zlpd", sent_to_extractions)
        return sent_to_extractions
    
    def write_simp_sents_file(self,
                          out_fp,
                          first_sample_id,
                          last_sample_id):
 
        assert 1<= first_sample_id <= last_sample_id <= self.num_sents

        with open(out_fp, 'w') as f:
            sample_id = 0
            for sent, l_ex in self.sent_to_extractions.items():
                sample_id += 1
                if sample_id < first_sample_id or sample_id > last_sample_id:
                    continue
                f.write(str(sample_id) + ".\n")
                f.write(sent + "\n")
                for ex in l_ex:
                    f.write(ex.get_simple_sent() + "/n")
    def write_extags_file(self,
                          out_fp,
                          first_sample_id,
                          last_sample_id):
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

        assert 1<= first_sample_id <= last_sample_id <= self.num_sents

        with open(out_fp, 'w') as f:
            sample_id = 0
            for sent, l_ex in self.sent_to_extractions.items():
                sample_id += 1
                if sample_id < first_sample_id or sample_id > last_sample_id:
                    continue
                f.write(str(sample_id) + ".\n")
                f.write(sent + "\n")
                for ex in l_ex:
                    ex.set_extags()
                    f.write(" ".join(ex.extags()))
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

        train_first_last = (
            1, 
            num_train_sents)
        tune_first_last = (
            num_train_sents+1,
            num_train_sents + num_tune_sents)
        test_first_last = (
            num_train_sents + num_tune_sents + 1,
            num_train_sents + num_tune_sents + num_test_sents)

        train_fp = out_dir + "/extags_train.txt"
        tune_fp = out_dir + "/extags_tune.txt"
        test_fp = out_dir + "/extags_test.txt"

        self.write_extags_file(train_fp, *train_first_last)
        self.write_extags_file(tune_fp, *tune_first_last)
        self.write_extags_file(test_fp, *test_first_last)


if __name__ == "__main__":
    def main1():
        in_path = "input_data/imojie-data/train/oie4_extractions.tsv"
        with open(in_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [unidecode(line) for line in lines]
        for k, line in enumerate(lines[0:5]):
            at = AllenTool(in_path)
            ex = at.read_allen_line(line)
            print(str(k+1) + ".")
            print(ex.orig_sentL_pair[0])
            print("arg1=", ex.arg1_pair[0])
            print("rel=", ex.rel_pair[0])
            print("arg2=", ex.arg2_pair[0])

    def main2():
        in_path = "input_data/imojie-data/train/oie4_extractions.tsv"
        at = AllenTool(in_path)
        sent_to_extractions = at.read_allen_file()
        # print("llkp", list(sent_to_extractions.keys())[0:2])
        for sent in list(sent_to_extractions.keys())[0:5]:
            extractions = sent_to_extractions[sent]
            print(sent)
            for k, ex in enumerate(extractions):
                print(str(k+1) + ".")
                print("arg1=", ex.arg1_pair[0])
                print("rel=", ex.rel_pair[0])
                print("arg2=", ex.arg2_pair[0])
            print()
    


    # main1()
    main2()
