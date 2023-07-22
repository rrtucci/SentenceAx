import numpy as np
from CC_Sentence import *


def remove_unbreakable_spans(cc_loc_to_spans, words):
    unbreakable_locs = []

    unbreakable_words = ["between", "among", "sum", "total",
                         "addition", "amount", "value", "aggregate", "gross",
                         "mean", "median", "average", "center", "equidistant",
                         "middle"]

    for i, word in enumerate(words):
        if word.lower() in unbreakable_words:
            unbreakable_locs.append(i)

    to_remove = []
    span_start = 0

    for cc_loc in cc_loc_to_spans:
        span_end = cc_loc_to_spans[cc_loc].spans[0][0] - 1
        for i in unbreakable_locs:
            if span_start <= i <= span_end:
                to_remove.append(cc_loc)
        span_start = cc_loc_to_spans[cc_loc].spans[-1][-1] + 1

    for k in set(to_remove):
        cc_loc_to_spans.pop(k)


def get_sentences(sentences, cc_locs_same_level,
                  cc_loc_to_csent, sentence_locs):
    for cc_loc in cc_locs_same_level:

        if len(sentences) == 0:

            for span in cc_loc_to_csent[cc_loc].spans:
                sentence = []
                for i in range(span[0], span[1] + 1):
                    sentence.append(i)
                sentences.append(sentence)

            min = cc_loc_to_csent[cc_loc].spans[0][0]
            max = cc_loc_to_csent[cc_loc].spans[-1][-1]

            for sentence in sentences:
                for i in sentence_locs:
                    if i < min or i > max:
                        sentence.append(i)

        else:
            to_add = []
            to_remove = []

            for sentence in sentences:
                if cc_loc_to_csent[cc_loc].spans[0][0] in sentence:
                    sentence.sort()

                    min = cc_loc_to_csent[cc_loc].spans[0][0]
                    max = cc_loc_to_csent[cc_loc].spans[-1][-1]

                    for span in cc_loc_to_csent[cc_loc].spans:
                        new_sentence = []
                        for i in sentence:
                            if i in range(span[0], span[1] + 1) or \
                                    i < min or i > max:
                                new_sentence.append(i)

                        to_add.append(new_sentence)

                    to_remove.append(sentence)

            for sent in to_remove:
                sentences.remove(sent)
            sentences.extend(to_add)


def get_tree(cc_loc_to_csent):
    child_cc_locs = []

    par_cc_loc_to_child_cc_loc = {}
    child_cc_loc_to_par_cc_loc = {}

    for par_cc_loc in cc_loc_to_csent:
        assert cc_loc_to_csent[par_cc_loc].cc_loc == par_cc_loc
        child_cc_locs.append([])
        for child_cc_loc in cc_loc_to_csent:
            if cc_loc_to_csent[child_cc_loc] is not None:
                if cc_loc_to_csent[par_cc_loc].is_parent(
                        cc_loc_to_csent[child_cc_loc]):
                    child_cc_locs[-1].append(child_cc_loc)

        par_cc_loc_to_child_cc_loc[par_cc_loc] = child_cc_locs[-1]

    child_cc_locs.sort(key=list.__len__)

    for i in range(0, len(child_cc_locs)):
        for child_cc_loc in child_cc_locs[i]:
            for j in range(i + 1, len(child_cc_locs)):
                if child_cc_loc in child_cc_locs[j]:
                    child_cc_locs[j].remove(child_cc_loc)

    for par_cc_loc in cc_loc_to_csent:
        for child_cc_loc in par_cc_loc_to_child_cc_loc[par_cc_loc]:
            child_cc_loc_to_par_cc_loc[child_cc_loc] = par_cc_loc

    root_cc_loc = []
    for par_cc_loc in cc_loc_to_csent:
        if par_cc_loc not in child_cc_loc_to_par_cc_loc:
            root_cc_loc.append(par_cc_loc)

    return root_cc_loc, child_cc_loc_to_par_cc_loc, par_cc_loc_to_child_cc_loc


def csents_to_sentences(csents, words):
    for k in range(len(csents)):
        if csents[k] is None:
            csents.pop(k)

    for k in range(len(csents)):
        if words[csents[k].cc_loc] in ['nor', '&']:  # , 'or']:
            csents.pop(k)

    num_csents = len(csents)
    # for k in list(csents):
    #     if len(csents[k].spans) < 3 and words[csents[k].cc_loc].lower() == 'and':
    #         csents.pop(k)
    # if len(csents[k].spans) < 3:
    #     csents.pop(k)
    # else:
    #     named_entity = False
    #     for span in csents[k].spans:
    #         # if not words[span[0]][0].isupper():
    #         if (span[1]-span[0]) > 0 or len(csents)>1:
    #             named_entity = True
    #     if named_entity:
    #         # conj_words = []
    #         # for span in csents[k].spans:
    #         #     conj_words.append(' '.join(words[span[0]:span[1]+1]))
    #         # open('temp.txt', 'a').write('\n'+' '.join(words)+'\n'+'\n'.join(conj_words)+'\n')
    #         csents.pop(k)

    remove_unbreakable_spans(csents, words)

    conj_words = []
    for k in range(len(csents)):
        for span in csents[k].spans:
            conj_words.append(' '.join(words[span[0]:span[1] + 1]))

    sentence_indices = []
    for i in range(0, len(words)):
        sentence_indices.append(i)

    root_cc_locs, child_cc_loc_to_par_cc_loc, par_cc_loc_to_child_cc_loc =\
        get_tree(csents)

    sentences = []
    count = len(root_cc_locs)
    new_count = 0

    conj_same_level = []

    while (len(root_cc_locs) > 0):

        root_cc_locs.pop(0)
        count -= 1
        conj_same_level.append(root_cc_locs)

        for child_cc_loc in par_cc_loc_to_child_cc_loc[root_cc_locs]:
            root_cc_locs.append(child_cc_loc)
            new_count += 1

        if count == 0:
            get_sentences(sentences, conj_same_level,
                          csents, sentence_indices)
            count = new_count
            new_count = 0
            conj_same_level = []

    word_sentences = [' '.join([words[i] for i in sorted(sentence)])
                      for sentence in sentences]

    return word_sentences, conj_words, sentences
    # return '\n'.join(word_sentences) + '\n'


def csents_to_string(csents, words):
    conj_str = ''
    for k in range(len(csents)):
        if csents[k] == None:
            conj_str += words[k]+': None  \n'
            continue
        cc_word = words[csents[k].cc_loc]
        conj_str += cc_word+': '
        for span in csents[k].spans:
            span_words = ' '.join(words[span[0]:span[1]+1])
            conj_str += span_words+'; '
        conj_str = conj_str[:-2]+'  \n'
    return conj_str

def post_process_csents(csents, is_quote):
    new_csents = []
    # `np.argwhere(is_quote)` returns indices,. as a column,
    # of entries >0
    offsets = np.delete(is_quote.cumsum(), index=np.argwhere(is_quote))
    for csent in csents:
        csent.cc_loc = csent.cc_loc + offsets[csent.cc_loc]
        if csent is not None:
            spans = [(b + offsets[b], e + offsets[e])
                         for (b, e) in csent.spans]
            seps = [s + offsets[s] for s in csent.seps]
            csent = CC_Sentence(csent.cc_loc, spans, seps, csent.label)
        new_csents.append(csent)
    return new_csents

def get_csents(depth_to_labels):
    csents = []

    for depth in range(len(depth_to_labels)):
        csent = None
        spans = []
        start_loc = -1
        is_conjunction = False
        labels = depth_to_labels[depth]

        for i, label in enumerate(labels):
            if label != 1:  # conjunction can end
                if is_conjunction and csent != None:
                    is_conjunction = False
                    spans.append((start_loc, i-1))
            if label == 0 or label == 2:  # csentination phrase can end
                if csent != None and\
                        len(spans) >= 2 and\
                        cc_loc > spans[0][1] and\
                        cc_loc < spans[-1][0]:
                    csent = CC_Sentence(cc_loc, spans, label=depth)
                    # if correct:
                    #     csentination = clean_spans(csentination, words)
                    csents.append(csent)
                    csent = None

            if label == 0:
                continue
            if label == 1:  # can start a conjunction
                if not is_conjunction:
                    is_conjunction = True
                    start_loc = i
            if label == 2:  # starts a csentination phrase
                cc_loc, spans, seps = -1, [], []
                is_conjunction = True
                start_loc = i
            if label == 3 and csent != None:
                cc_loc = i
            if label == 4 and csent != None:
                seps.append(i)
            if label == 5:  # nothing to be done
                continue
            if label == 3 and csent == None:
                # csentinating words which do not have associated spans
                csents[i] = None

    return csents
