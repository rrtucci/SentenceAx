import re
from unidecode import unidecode
from collections import OrderedDict
from SaxExtraction import *
from words_tags_ilabels_translation import translate_words_to_extags
from words_tags_ilabels_translation import translate_extags_to_ilabels


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
    refer to the same "sample". The five lines
    differ in what comes after the original sentence ("osent"). What comes
    after the osent in each line is an "extraction (ex)". In Allen files,
    an extraction is composed of 3 parts:

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
    osentL_to_exs: dict[str,list[SaxExtraction]]
        sentenceLong to extraction list mapping.

    """

    def __init__(self, allen_fp):
        """
        Constructor. Note that it automatically reads the Allen file with
        path `allen_fp` and stores the file's information in the attribute
        `self.osentL_to_exs`.

        Parameters
        ----------
        allen_fp: str
        """
        self.allen_fp = allen_fp
        self.osentL_to_exs = self.read_allen_file()
        # print("mklop", self.osentL_to_exs)
        self.num_sents = len(self.osentL_to_exs)

        # print("lkop", self.num_sents)

    @staticmethod
    def print_osent2_to_exs(osent2_to_exs, start_id=None, end_id=None):
        """
        This method prints `osent2_to_exs` in the sample range
        `start_d:end_id`.

        Parameters
        ----------
        osent2_to_exs: dict[str, list[SaxExtraction]]
        start_id: int|None
        end_id: int|None

        Returns
        -------
        None

        """
        if not start_id:
            start_id = 0
        if not end_id:
            end_id = len(osent2_to_exs)
        for sent in list(osent2_to_exs.keys())[start_id: end_id]:
            print()
            print(sent)
            for ex in osent2_to_exs[sent]:
                print("(part) " + ex.get_simple_sent())

    @staticmethod
    def get_osent2_to_exs_from_lll_ilabel(l_osent2,
                                          lll_ilabel,
                                          ll_confidence,
                                          sent_to_sent):
        """
        similar to Openie6.metric.Carb.__call__()

        This static  method takes as input `lll_ilabel` and other variables
        and returns `osent2_to_exs` (one of the attributes of class AllenTool).

        Note this is the inverse of get_lll_ilabel_from_osent2_to_exs().

        osent = original sentence
        osentL = osent + UNUSED_TOKENS_STR  #L=long
        osent2 = either osent or osentL

        This method does not care internally whether we are using `osentL,
        lll_ilabels` or `osent, lll_ilabels`. That is why we use the symbol
        `osent2`, which can stand for `osent` or `osentL`


        Parameters
        ----------
        l_osent2: list[str]
        lll_ilabel: list[list[list[int]]]
        ll_confidence: list[list[float]]
        sent_to_sent: dict[str, str]
            a dictionary that makes small fixes on osent2

        Returns
        -------
        dict[str, list[SaxExtraction]]

        """

        osent2_to_exs = {}
        for sam_id, osent2 in enumerate(l_osent2):
            add_key_to_this_d(key=osent2,
                              transform_d=sent_to_sent,
                              this_d=osent2_to_exs)

            num_exs = len(ll_confidence[sam_id])
            for depth in range(num_exs):
                ilabels = lll_ilabel[sam_id][depth]
                if sum(ilabels):
                    ex0 = SaxExtraction.get_ex_from_ilabels(
                        ilabels,
                        osent2,
                        ll_confidence[sam_id][depth])
                    if ex0.arg1 and ex0.rel:
                        add_key_value_pair_to_this_d(
                            key=osent2,
                            value=ex0,
                            transform_d=sent_to_sent,
                            this_d=osent2_to_exs)
        return osent2_to_exs

    @staticmethod
    def get_lll_ilabel_from_osent2_to_exs(osent2_to_exs):
        """
        This static method takes as input `osent2_to_exs` (one of the
        attributes of class AllenTool). It returns as output `l_osent2`,
        `lll_ilabel`, `ll_confidence`

        Note this is the inverse of get_osent2_to_exs_from_lll_ilabel()

        osent = original sentence
        osentL = osent + UNUSED_TOKENS_STR
        
        This method does not care internally whether we are using `osentL,
        lll_ilabels` or `osent, lll_ilabels`. That is why we use the symbol
        `osent2`, which can stand for either `osent` or `osentL`

        Parameters
        ----------
        osent2_to_exs: dict[str, list[SaxExtraction]]

        Returns
        -------
        list[str], list[list[list[int]]], list[list[float]]

        """
        l_osent2, l_exs = zip(*osent2_to_exs.items())

        def get_ilabels(ex):
            extags = translate_words_to_extags(ex)
            ilabels = translate_extags_to_ilabels(extags)
            # ilabels = ilabels[0: len(l_osent2[sam_id])]
            return ilabels

        lll_ilabel = [[get_ilabels(ex) for ex in exs] for
                      exs in l_exs]
        ll_confidence = [[ex.confidence for ex in exs] for exs in l_exs]
        return l_osent2, lll_ilabel, ll_confidence

    @staticmethod
    def get_ex_from_allen_line(line):
        """
        This method takes as input a single line `line` from an Allen file
        and returns an instance of the class SaxExtraction.

        Parameters
        ----------
        line: str

        Returns
        -------
        SaxExtraction

        """
        tab_sep_vals = line.strip().split('\t')
        in_sentL = tab_sep_vals[0] + UNUSED_TOKENS_STR
        confidence = float(tab_sep_vals[2])
        # if len(tab_sep_vals) == 4:
        #     num_exs = int(tab_sep_vals[3])
        # else:
        #     num_exs = None

        assert len(re.findall("<arg1>.*</arg1>", tab_sep_vals[1])) == 1
        assert len(re.findall("<rel>.*</rel>", tab_sep_vals[1])) == 1
        assert len(re.findall("<arg2>.*</arg2>", tab_sep_vals[1])) == 1

        sent_parts = []
        for str0 in ['arg1', 'rel', 'arg2']:
            begin_tag = '<' + str0 + '>'
            end_tag = '</' + str0 + '>'
            sent_part = \
                re.findall(begin_tag + '.*' + end_tag, tab_sep_vals[1])[0]
            # print("vcbgh", sent_part)
            sent_part = ' '.join(
                get_words(sent_part.strip(begin_tag).strip(end_tag)))
            sent_parts.append(sent_part)

        ex = SaxExtraction(in_sentL,
                           sent_parts[0],
                           sent_parts[1],
                           sent_parts[2],
                           confidence)

        return ex

    @staticmethod
    def get_allen_line_from_ex(ex):
        """
        This method takes as input an `ex` instance of the class
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
        str0 += str(ex.confidence)
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
        osentL_to_exs = OrderedDict()
        exs = []
        prev_osentL = ''
        for line in lines:
            # print("bnhk", line)
            ex = AllenTool.get_ex_from_allen_line(line)
            if prev_osentL and ex.orig_sentL != prev_osentL:
                osentL_to_exs[prev_osentL] = exs
                exs = []  # start exs afresh
            exs.append(ex)
            prev_osentL = ex.orig_sentL
        # last sample
        osentL_to_exs[prev_osentL] = exs
        # print("zlpd", osentL_to_exs)
        return osentL_to_exs

    def write_allen_alternative_file(self,
                                     out_fp,
                                     first_last_sample_id=(1, -1),
                                     ftype="ex",
                                     numbered=False):

        """
        Hercule Poirot is a fictional Belgian detective , created by Agatha Christie . [unused1] [unused2] [unused3]
        ARG1 ARG1 REL ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 ARG2 NONE NONE NONE NONE
        NONE NONE NONE ARG1 ARG1 ARG1 ARG1 NONE REL ARG2 ARG2 ARG2 NONE NONE NONE NONE

        Above 3 lines come from an "extags file". Extags are extraction tags
        from the set {'NONE', 'ARG1', 'REL', 'ARG2', 'LOC', 'TIME',
        'TYPE', 'ARGS'}. These 3 lines represent a single "sample". The
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
        with the string (LINE_SEPARATOR + 1-based sample number) at the
        beginning of
        each sample.


        Parameters
        ----------
        out_fp: str
            output file path
        first_last_sample_id: tuple(int, int)
            number of first and last sample to be written in output file (
            1-based)
        ftype: str
            ftype in ["ex", "ss"]. If ftype=="ex", an extags file is
            written. If ftype=="ss", a simple sentences file is written
            instead.
        numbered:
            True iff the string (LINE_SEPARATOR + 1-based sample number)
            will be written as the first line of each sample.

        Returns
        -------
        None

        """
        first_sample_id, last_sample_id = first_last_sample_id
        if last_sample_id != -1:
            assert 1 <= first_sample_id <= last_sample_id <= self.num_sents
        else:
            last_sample_id = self.num_sents

        with open(out_fp, 'w', encoding='utf-8') as f:
            sample_id = 0
            num_sams = 0
            for osentL, l_ex in self.osentL_to_exs.items():
                sample_id += 1
                if sample_id < first_sample_id or \
                        sample_id > last_sample_id:
                    continue
                if numbered:
                    f.write(LINE_SEPARATOR + str(sample_id) + "\n")
                num_sams += 1
                f.write(osentL + "\n")
                for ex in l_ex:
                    if ftype == "ex":  # extags file
                        # print("llko34", sample_id, ex)
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
        ttt = train, tune, test
        tuning=dev=development=validation

        This method uses as input the samples loaded by the constructor from
        an Allen file. The method outputs 3 files into the directory
        `out_dir`. Those 3 files will be used for training, tuning (
        validation) and testing, respectively.

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
        None

        """
        num_train_sents, num_tune_sents, num_test_sents = \
            get_num_ttt_sents(self.num_sents, ttt_fractions)

        train_first_last = (
            1,
            num_train_sents)
        tune_first_last = (
            num_train_sents + 1,
            num_train_sents + num_tune_sents)
        test_first_last = (
            num_train_sents + num_tune_sents + 1,
            num_train_sents + num_tune_sents + num_test_sents)

        print("first_last pairs for train, tune, test:", train_first_last,
              tune_first_last,
              test_first_last)

        tags_train_fp = out_dir + "/extags_train.txt"
        tags_tune_fp = out_dir + "/extags_tune.txt"
        tags_test_fp = out_dir + "/extags_test.txt"

        self.write_allen_alternative_file(
            tags_train_fp, first_last_sample_id=train_first_last)
        self.write_allen_alternative_file(
            tags_tune_fp, first_last_sample_id=tune_first_last)
        self.write_allen_alternative_file(
            tags_test_fp, first_last_sample_id=test_first_last)


if __name__ == "__main__":
    def main1(ftype, numbered):

        if not numbered:
            str0 = ""
        else:
            str0 = "N"
            print("'N' suffix to file name means file samples are numbered")

        allen_fp = "tests/small_allen.tsv"
        if ftype == "ex":
            out_fp = "tests/small_extags" + str0 + ".txt"
        elif ftype == "ss":
            out_fp = "tests/small_simple_sents" + str0 + ".txt"
        else:
            assert False
        at = AllenTool(allen_fp)
        at.write_allen_alternative_file(out_fp,
                                        first_last_sample_id=(1, -1),
                                        ftype=ftype,
                                        numbered=numbered)


    def main2():
        allen_fp = "tests/small_allen.tsv"
        out_dir = "tests"
        at = AllenTool(allen_fp)
        at.write_extags_ttt_files(out_dir)


    main1("ss", numbered=True)
    main1("ex", numbered=True)
    main1("ex", numbered=False)
    main2()
