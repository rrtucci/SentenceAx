import numpy as np
from CCList import *


# def remove_unbreakable_spans(cclists, words):
#     unbreakable_locs = []
# 
#     unbreakable_words = ["between", "among", "sum", "total",
#                          "addition", "amount", "value", "aggregate", "gross",
#                          "mean", "median", "average", "center", "equidistant",
#                          "middle"]
# 
#     # e.g. difference between apples and oranges
#     # ccloc = 3
#     # words = ["difference", "between", "apples", "and", "oranges"]
#     # spans=[(0,2), (4,4)]
# 
#     for i, word in enumerate(words):
#         if word.lower() in unbreakable_words:
#             unbreakable_locs.append(i)
# 
#     to_be_removed_cclists = []
# 
#     for cclist in cclists:
#         spans = cclist.spans
#         cclist_start = spans[0][0]
#         cclist_end = spans[-1][-1]
#         for i in unbreakable_locs:
#             if cclist_start <= i <= cclist_end:
#                 to_be_removed_cclists.append(cclist)
# 
#     for cclist in cclists:
#         if cclist in to_be_removed_cclists:
#             cclists.pop(cclists.index(cclist))


def get_sentences(sentences, cclocs_same_level,
                  ccloc_to_cclist, sentence_locs):
    for ccloc in cclocs_same_level:

        if len(sentences) == 0:

            for span in ccloc_to_cclist[ccloc].spans:
                sentence = []
                for i in range(span[0], span[1]):
                    sentence.append(i)
                sentences.append(sentence)

            min = ccloc_to_cclist[ccloc].spans[0][0]
            max = ccloc_to_cclist[ccloc].spans[-1][1]-1

            for sentence in sentences:
                for i in sentence_locs:
                    if i < min or i > max:
                        sentence.append(i)

        else:
            to_be_added_sents = []
            to_be_removed_sents = []

            for sentence in sentences:
                if ccloc_to_cclist[ccloc].spans[0][0] in sentence:
                    sentence.sort()

                    min = ccloc_to_cclist[ccloc].spans[0][0]
                    max = ccloc_to_cclist[ccloc].spans[-1][1]-1

                    for span in ccloc_to_cclist[ccloc].spans:
                        new_sentence = []
                        for i in sentence:
                            if i in range(span[0], span[1]) or \
                                    i < min or i > max:
                                new_sentence.append(i)

                        to_be_added_sents.append(new_sentence)

                    to_be_removed_sents.append(sentence)

            for sent in to_be_removed_sents:
                sentences.remove(sent)
            sentences.extend(to_be_added_sents)


def get_tree(ccloc_to_cclist):
    child_cclocs = []

    # par = parent
    par_ccloc_to_child_ccloc = {}
    child_ccloc_to_par_ccloc = {}

    for par_ccloc in ccloc_to_cclist:
        assert ccloc_to_cclist[par_ccloc].ccloc == par_ccloc
        child_cclocs.append([])
        for child_ccloc in ccloc_to_cclist:
            if ccloc_to_cclist[child_ccloc] is not None:
                if ccloc_to_cclist[par_ccloc].is_parent(
                        ccloc_to_cclist[child_ccloc]):
                    child_cclocs[-1].append(child_ccloc)

        par_ccloc_to_child_ccloc[par_ccloc] = child_cclocs[-1]

    child_cclocs.sort(key=list.__len__)

    for i in range(0, len(child_cclocs)):
        for child_ccloc in child_cclocs[i]:
            for j in range(i + 1, len(child_cclocs)):
                if child_ccloc in child_cclocs[j]:
                    child_cclocs[j].remove(child_ccloc)

    for par_ccloc in ccloc_to_cclist:
        for child_ccloc in par_ccloc_to_child_ccloc[par_ccloc]:
            child_ccloc_to_par_ccloc[child_ccloc] = par_ccloc

    root_ccloc = []
    for par_ccloc in ccloc_to_cclist:
        if par_ccloc not in child_ccloc_to_par_ccloc:
            root_ccloc.append(par_ccloc)

    return root_ccloc, child_ccloc_to_par_ccloc, par_ccloc_to_child_ccloc


def cclists_to_sentences(cclists, words):
    for k in range(len(cclists)):
        if cclists[k] is None:
            cclists.pop(k)

    for k in range(len(cclists)):
        if words[cclists[k].ccloc] in ['nor', '&']:  # , 'or']:
            cclists.pop(k)

    num_cclists = len(cclists)
    # for k in list(cclists):
    #     if len(cclists[k].spans) < 3 and words[cclists[k].ccloc].lower() == 'and':
    #         cclists.pop(k)
    # if len(cclists[k].spans) < 3:
    #     cclists.pop(k)
    # else:
    #     named_entity = False
    #     for span in cclists[k].spans:
    #         # if not words[span[0]][0].isupper():
    #         if (span[1]-span[0]) > 0 or len(cclists)>1:
    #             named_entity = True
    #     if named_entity:
    #         # conj_words = []
    #         # for span in cclists[k].spans:
    #         #     conj_words.append(' '.join(words[span[0]:span[1]+1]))
    #         # open('temp.txt', 'a').write('\n'+' '.join(words)+'\n'+'\n'.join(conj_words)+'\n')
    #         cclists.pop(k)

    remove_unbreakable_spans(cclists, words)

    conj_words = []
    for k in range(len(cclists)):
        for span in cclists[k].spans:
            conj_words.append(' '.join(words[span[0]:span[1] + 1]))

    sentence_indices = []
    for i in range(0, len(words)):
        sentence_indices.append(i)

    root_cclocs, child_ccloc_to_par_ccloc, par_ccloc_to_child_ccloc =\
        get_tree(cclists)

    sentences = []
    count = len(root_cclocs)
    new_count = 0

    conj_same_level = []

    while (len(root_cclocs) > 0):

        root_cclocs.pop(0)
        count -= 1
        conj_same_level.append(root_cclocs)

        for child_ccloc in par_ccloc_to_child_ccloc[root_cclocs]:
            root_cclocs.append(child_ccloc)
            new_count += 1

        if count == 0:
            get_sentences(sentences, conj_same_level,
                          cclists, sentence_indices)
            count = new_count
            new_count = 0
            conj_same_level = []

    word_sentences = [' '.join([words[i] for i in sorted(sentence)])
                      for sentence in sentences]

    return word_sentences, conj_words, sentences
    # return '\n'.join(word_sentences) + '\n'


# def cclists_to_string(cclists, words):
#     conj_str = ''
#     for k in range(len(cclists)):
#         if cclists[k] == None:
#             conj_str += words[k]+': None  \n'
#             continue
#         cc_word = words[cclists[k].ccloc]
#         conj_str += cc_word+': '
#         for span in cclists[k].spans:
#             span_words = ' '.join(words[span[0]:span[1]+1])
#             conj_str += span_words+'; '
#         conj_str = conj_str[:-2]+'  \n'
#     return conj_str

def post_process_cclists(cclists, is_quote):
    new_cclists = []
    # `np.argwhere(is_quote)` returns indices,. as a column,
    # of entries >0
    offsets = np.delete(is_quote.cumsum(), index=np.argwhere(is_quote))
    for cclist in cclists:
        cclist.ccloc = cclist.ccloc + offsets[cclist.ccloc]
        if cclist is not None:
            spans = [(b + offsets[b], e + offsets[e])
                         for (b, e) in cclist.spans]
            seps = [s + offsets[s] for s in cclist.seps]
            cclist = CCList(cclist.ccloc, spans, seps, cclist.tag)
        new_cclists.append(cclist)
    return new_cclists

def get_cclists(depth_to_tags):
    cclists = []

    for depth in range(len(depth_to_tags)):
        cclist = None
        spans = []
        start_loc = -1
        is_conjunction = False
        tags = depth_to_tags[depth]

        for i, tag in enumerate(tags):
            if tag != 1:  # conjunction can end
                if is_conjunction and cclist != None:
                    is_conjunction = False
                    spans.append((start_loc, i-1))
            if tag == 0 or tag == 2:  # cclistination phrase can end
                if cclist != None and\
                        len(spans) >= 2 and\
                        ccloc > spans[0][1] and\
                        ccloc < spans[-1][0]:
                    cclist = CCList(ccloc, spans, tag=depth)
                    # if correct:
                    #     cclistination = clean_spans(cclistination, words)
                    cclists.append(cclist)
                    cclist = None

            if tag == 0:
                continue
            if tag == 1:  # can start a conjunction
                if not is_conjunction:
                    is_conjunction = True
                    start_loc = i
            if tag == 2:  # starts a cclistination phrase
                ccloc, spans, seps = -1, [], []
                is_conjunction = True
                start_loc = i
            if tag == 3 and cclist != None:
                ccloc = i
            if tag == 4 and cclist != None:
                seps.append(i)
            if tag == 5:  # nothing to be done
                continue
            if tag == 3 and cclist == None:
                # cclistinating words which do not have associated spans
                cclists[i] = None

    return cclists
