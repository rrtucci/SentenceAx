import re
from unidecode import unidecode
from collections import OrderedDict
from Extraction_sax import *

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
    
    @staticmethod
    def write_allen_line(ex):
        str0 = ex.sentL_pair[0]
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
            part = ' '.join(part.strip(begin_tag).strip(end_tag).
                            strip().split())
            parts.append(part)
        ex = Extraction_sax(in_sent, parts[0], parts[1], parts[2], confidence)
    
        return ex
    
    def get_num_sents_in_allen_file(self):
        with open(self.allen_fp, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        prev_in_sent = ''
        num_sents = 1
        for line in lines:
            ex = AllenTool.read_allen_line(line)
            if ex.sentL_pair[0] != prev_in_sent:
                num_sents += 1
            prev_in_sent = ex.sentL_pair[0]
        return num_sents
    
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
            # print("ooyp1", ex.sentL_pair[0])
            # print("ooyp2", prev_in_sent)
            # print("ooyp3", ex.sentL_pair[0] == prev_in_sent)
            if ex.sentL_pair[0] != prev_in_sent:
                if prev_in_sent:
                    sent_to_extractions[prev_in_sent] = extractions[:-1]
                    # print("llkml", prev_in_sent, extractions)
                    prev_in_sent = ex.sentL_pair[0]
                    extractions = [ex]
                else:
                    prev_in_sent = ex.sentL_pair[0]
    
        # print("zlpd", sent_to_extractions)
        return sent_to_extractions


if __name__ == "__main__":
    def main1():
        in_path = "data/imojie-data/train/oie4_extractions.tsv"
        with open(in_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [unidecode(line) for line in lines]
        for k, line in enumerate(lines[0:5]):
            at = AllenTool(in_path)
            ex = at.read_allen_line(line)
            print(str(k+1) + ".")
            print(ex.sentL_pair[0])
            print("arg1=", ex.arg1_pair[0])
            print("rel=", ex.rel_pair[0])
            print("arg2=", ex.arg2_pair[0])

    def main2():
        in_path = "data/imojie-data/train/oie4_extractions.tsv"
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
