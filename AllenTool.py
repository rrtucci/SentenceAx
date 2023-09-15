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

    This class is for extracting info from an "AllenNLP" file, or "Allen"
    file, for short. Above are 5 lines of an Allen file. They have the same
    original sentences (the first sentence of each line). Hence, they all
    refer to the same "sample". The five lines differ in what comes after
    the original sentence ("osent"). What comes after the osent in each line
    is an "extraction". In Allen files, an extraction is composed of 3 parts:

    argument1, demarcated by `<arg1></arg1>`
    relationship, demarcated by `<rel></rel>`
    argument2, demarcated by `<arg2></arg2>`

    No blank lines between samples.

    osentL = osent + " [ unused1] [unused2] [unused3]"
    An `osent` plus the UNUSUSED_TOKEN_STR will be denoted by `osentL`,
    where the "L" stands for "Long"

    Attributes
    ----------
    allen_fp: str
        Allen file path
    num_sents: int
        number of sentences
    sentL_to_exs: dict[str,list[SaxExtraction]]
        sentenceLong to extraction list mapping
    
    """

    def __init__(self, allen_fp):
        """
        Constructor. Note that it automatically reads the Allen file with
        path `allen_fp` and stores its information in the attribute
        `self.sentL_to_exs`.

        Parameters
        ----------
        allen_fp: str
        """
        self.allen_fp = allen_fp
        # ex =extraction sent=sentence
        self.sentL_to_exs = self.read_allen_file()
        # print("mklop", self.sentL_to_exs)
        self.num_sents = len(self.sentL_to_exs)
        # print("lkop", self.num_sents)

    @staticmethod
    def get_ex_from_allen_line(line):
        """
        This method takes as input a single line `line` from an Allen file
        and returns an object of the class SaxExtraction. (Sax stands for
        Sentence Ax, the name of this app).

        Parameters
        ----------
        line: str

        Returns
        -------
        SaxExtraction

        """
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
        """
        Thiis method takes as input an object `ex` of the class
        `SaxExtraction` and it returns a string formatted as a line of an
        Allen file. Hence, this method is the inverse of the method
        `get_ex_from_allen_line()`.

        Parameters
        ----------
        ex: SaxExtraction

        Returns
        -------
        str

        """
        str0 = ex.orig_sentL + "\t"
        str0 += " <arg1> " + ex.arg1 + r" <\arg1> "
        str0 += " <rel> " + ex.rel + r" <\rel> "
        str0 += " <arg2> " + ex.arg2 + r" <\arg2> \t"
        str0 += str(ex.score)
        return str0

    def read_allen_file(self):
        """
        This method reads the file with file path `self.allen_fp` and it
        outputs that file's info as the dictionary `senL_to_exs`. This
        method is called by the class constructor.

        Returns
        -------
        dict[str, list[SaxExtraction]]

        """
        with open(self.allen_fp, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [unidecode(line) for line in lines]
        sentL_to_exs = OrderedDict()
        exs = []
        prev_in_sentL = ''
        for line in lines:
            # print("bnhk", line)
            ex = AllenTool.get_ex_from_allen_line(line)
            if prev_in_sentL and ex.orig_sentL != prev_in_sentL:
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
        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        Above 3 lines come from an "extags file". Extags are extraction tags
        from the set set('NONE', 'ARG1', 'REL', 'ARG2', 'LOC', 'TIME',
        'TYPE', 'ARGS'). These 3 lines represent a single "sample". The
        first line is osentL, the next 2 lines give the extags of
        extractions from that osent. The osent line and the 2 extraction
        lines have the same number of words. (You can get the words of a
        text string using sax_utils.get_words()).

        This method writes an extags file (if ftype=="ex") or a simple
        sentences file (if ftype=="ss"). In a simple sentences file,
        the extractions are represented by the osent words of the extags
        which are different from NONE. We refer to "ex" and "ss" files as
        "alternatives" to an Allen file.

        If `numbered` is set to False, the output file will have no blank
        lines separating samples. If it is set to True, it will have a line
        with a number and period at the beginning of each sample.


        Parameters
        ----------
        out_fp: str
            output file path
        first_sample_id: int
            number of first sample to be written in output file (1-based)
        last_sample_id: int
            number of last sample to be written in output file (1-based)
        ftype: str
            ftype in ["ex", "ss"]. If ftype=="ex", an extags file is
            written. If ftype=="ss", a simple sentences file is written
            instead.
        numbered:
            True iff a 1-based number label will be written as the first
            line of each sample.

        Returns
        -------
        None

        """

        if last_sample_id != -1:
            assert 1 <= first_sample_id <= last_sample_id <= self.num_sents

        with open(out_fp, 'w', encoding='utf-8') as f:
            sample_id = 0
            num_sams = 0
            for sent, l_ex in self.sentL_to_exs.items():
                sample_id += 1
                if sample_id < first_sample_id or sample_id > last_sample_id:
                    continue
                if numbered:
                    f.write(str(sample_id) + ".\n")
                num_sams += 1
                f.write(sent + "\n")
                for ex in l_ex:
                    if ftype == "ex":  # extags file
                        ex.set_extags()
                        f.write(" ".join(ex.extags) + "\n")
                    elif ftype == "ss":  # simple sentences file
                        f.write(ex.get_simple_sent() + "\n")
                    else:
                        assert False
            print("finished file " + out_fp + ":    " +
                  str(num_sams) + " samples.")

    def write_extags_ttt_files(self,
                               out_dir,
                               ttt_fractions=(.6, .2, .2)):
        """
        ttt = train, dev, test
        tuning=dev=development=validation

        This method uses as input the samples loaded by the constructor from
        an Allen file. The method outputs 3 files into the directory
        `out_dir`. Those 3 files will be used for training, development and
        testing, respectively.



        Parameters
        ----------
        out_dir: str
            Output directory
        ttt_fractions: list[float, float, float]
            The 3 fractions [f_train, f_dev, f_test] should add to one. They
            indicate the fraction of the samples from the input Allen file
            that will be devoted to training, development and testing.

        Returns
        -------

        """
        num_train_sents, num_val_sents, num_test_sents = \
            get_num_ttt_sents(self.num_sents, ttt_fractions)

        train_first_last = (
            1,
            num_train_sents)
        val_first_last = (
            num_train_sents + 1,
            num_train_sents + num_val_sents)
        test_first_last = (
            num_train_sents + num_val_sents + 1,
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
        if ftype == "ex":
            out_fp = "testing_files/small_extags.txt"
        elif ftype == "ss":
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


    main1("ss")
    main1("ex")
    main2()
