import re
from unidecode import unidecode
from collections import OrderedDict
from Extraction import *


def write_allen_line(ext):
    str0 = ext.in_ztz
    str0 += " <arg1> " + ext.arg1 + r" <\arg1> "
    str0 += " <rel> " + ext.rel + r" <\rel> "
    str0 += " <arg2> " + ext.arg2 + r" <\arg2> "
    str0 += str(ext.confidence)
    return str0

def parse_allen_line(line):
    tab_sep_vals = line.strip().split('\t')
    in_ztz = tab_sep_vals[0]
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
        part = ' '.join(part.strip(begin_tag).strip(end_tag).strip().split())
        parts.append(part)
    ext = Extraction(in_ztz, parts[0], parts[1], parts[2], confidence)

    return ext

def parse_allen_file(allen_fpath):
    with open(allen_fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [unidecode(line) for line in lines]
    ztz_to_extractions = OrderedDict()
    extractions = []
    prev_in_ztz = ''
    for line in lines:
        # print("bnhk", line)
        ext = parse_allen_line(line)
        extractions.append(ext)
        # print("ooyp1", ext.in_ztz)
        # print("ooyp2", prev_in_ztz)
        # print("ooyp3", ext.in_ztz == prev_in_ztz)
        if ext.in_ztz != prev_in_ztz:
            if prev_in_ztz:
                ztz_to_extractions[prev_in_ztz] = extractions[:-1]
                # print("llkml", prev_in_ztz, extractions)
                prev_in_ztz = ext.in_ztz
                extractions = [ext]
            else:
                prev_in_ztz = ext.in_ztz

    # print("zlpd", ztz_to_extractions)
    return ztz_to_extractions


if __name__ == "__main__":
    def main1():
        in_path = "data/imojie-data/train/oie4_extractions.tsv"
        with open(in_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [unidecode(line) for line in lines]
        for k, line in enumerate(lines[0:5]):
            ext = parse_allen_line(line)
            print(str(k+1) + ".")
            print(ext.in_ztz)
            print("arg1=", ext.arg1)
            print("rel=", ext.rel)
            print("arg2=", ext.arg2)

    def main2():
        in_path = "data/imojie-data/train/oie4_extractions.tsv"
        ztz_to_extractions = parse_allen_file(in_path)
        # print("llkp", list(ztz_to_extractions.keys())[0:2])
        for ztz in list(ztz_to_extractions.keys())[0:5]:
            extractions = ztz_to_extractions[ztz]
            print(ztz)
            for k, ext in enumerate(extractions):
                print(str(k+1) + ".")
                print("arg1=", ext.arg1)
                print("rel=", ext.rel)
                print("arg2=", ext.arg2)
            print()


    # main1()
    main2()
