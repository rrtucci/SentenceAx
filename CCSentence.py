from CCList import *
from numpy import np


class CCSentence:
    def __init__(self, words):
        self.words = words
        self.extra_locs = []

        self.cclists = None
        self.spanned_locs_list = []

    def get_tree(self):
        # par = parent
        par_ccloc_to_child_cclocs = {}
        child_ccloc_to_par_cclocs = {}

        for par_cclist in self.cclists:
            par_ccloc = par_cclist.ccloc
            child_cclocs_list = []
            for child_cclist in self.cclists:
                child_ccloc = child_cclist.ccloc
                if par_cclist.is_parent(child_cclist):
                    child_cclocs_list.append(child_ccloc)
            child_cclocs_list.sort()
            par_ccloc_to_child_cclocs[par_ccloc] = child_cclocs_list

        # tree so allow only one parent for each node

        child_cclocs_list = par_ccloc_to_child_cclocs.values().sort(
            key=list.__len__)
        # consider i < j
        for i in range(0, len(child_cclocs_list)):
            for j in range(i + 1, len(child_cclocs_list)):
                intersection = \
                    list(set(child_cclocs_list[i]) & set(child_cclocs_list[j]))
                child_cclocs_list[j] -= intersection

        # same as before but with par and child swapped
        for child_cclist in self.cclists:
            child_ccloc = child_cclist.ccloc
            par_cclocs = []
            for par_cclist in self.cclists:
                par_ccloc = par_cclist.ccloc
                if child_cclist.is_childent(par_cclist):
                    par_cclocs.append(par_ccloc)
            assert len(par_cclocs) == 1  # tree so only one parent per node
            child_ccloc_to_par_cclocs[child_ccloc] = par_cclocs

        root_cclocs = []
        for cclist in self.cclists:
            ccloc = cclist.ccloc
            if not child_ccloc_to_par_cclocs[ccloc]:
                root_cclocs.append(ccloc)

        return root_cclocs, \
            child_ccloc_to_par_cclocs, \
            par_ccloc_to_child_cclocs

    # get_sentences()
    def set_spanned_locs_list(self, cclists_same_level, extra_locs):


        for cclist in cclists_same_level:

            if len(self.spanned_locs_list) == 0:
                cclist.set_spanned_locs(extra_locs)
                self.spanned_locs_list.append(cclist.spanned_locs)

            else:
                cclist.set_spanned_locs(None)
                self.spanned_locs_list.append(cclist.spanned_locs)

                # to_be_added_loc_lists = []
                # to_be_removed_loc_lists = []
                # for spanned_locs in self.spanned_locs_list:
                #     if cclist.spans[0][0] in spanned_locs:
                #         spanned_locs.sort()
                #         new_spanned_locs = cclist.get_spanned_locs()
                #         to_be_added_loc_lists.append(new_spanned_locs)
                #         to_be_removed_loc_lists.append(spanned_locs)
                #
                #
                # for loc_list in to_be_removed_loc_lists:
                #     self.spanned_locs_list.remove(loc_list)
                # self.spanned_locs_list.extend(to_be_added_loc_lists)

    def fix_cclists(self):
        for cclist in self.cclists:
            if self.words[cclist.ccloc] in ['nor', '&'] or \
                    cclist.contains_unbreakable_spans():
                k = self.cclists.index(cclist)
                self.cclists.pop(k)

    def get_spanned_phrases(self):  # coords_to_sentences()

        self.fix_cclists()

        spanned_words = []
        for cclist in self.cclists:
            for span in cclist.spans:
                spanned_words.append(' '.join(self.words[span[0]:span[1]]))

        root_cclocs, child_ccloc_to_par_cclocs, par_ccloc_to_child_cclocs = \
            self.get_tree()

        self.spanned_locs_list = []
        root_count = len(root_cclocs)
        new_child_count = 0

        cclists_same_level = []

        while len(root_cclocs) > 0:

            root_cclocs.pop(0)
            root_count -= 1
            cclists_same_level.append(root_cclocs)

            for child_ccloc in par_ccloc_to_child_cclocs[root_cclocs]:
                root_cclocs.append(child_ccloc)
                new_child_count += 1

            if root_count == 0:
                self.set_spanned_locs_list(cclists_same_level, self.extra_locs)
                root_count = new_child_count
                new_child_count = 0
                cclists_same_level = []
        spanned_phrases = [' '.join([self.words[i] for i in
                                     sorted(spanned_locs)]) for spanned_locs in
                           self.spanned_locs_list]

        return spanned_phrases

    # def get_shifted_cclists(self, arr):  # post_process()
    #     new_cclists = []
    #     # arr is 1 dim numpy array
    #     # `np.argwhere(arr)` returns indices, as a column,
    #     # of entries of arr are >0
    #
    #     # np.argwhere([0, 5, 8, 0]) = [1, 2]
    #     # [0, 5, 8, 0].cumsum() = [0, 5, 13, 13]
    #     # np.delete([0, 5, 13, 13], [1, 2]) = [0, 13]
    #     shift = np.delete(arr.cumsum(), index=np.argwhere(arr))
    #     for cclist in self.cclists:
    #         ccloc = cclist.ccloc + shift[cclist.ccloc]
    #         spans = [(b + shift[b], e + shift[e])
    #                  for (b, e) in cclist.spans]
    #         seps_locs = [s + shift[s] for s in cclist.seps_locs]
    #         cclist = CCList(ccloc, spans, seps_locs, cclist.tag)
    #         new_cclists.append(cclist)
    #     return new_cclists

    def set_cclists(self, depth_to_tags): # get_coords()
        self.cclists = []

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
                        spans.append((start_loc, i - 1))
                if tag == 0 or tag == 2:  # cclist phrase can end
                    if cclist and \
                            len(spans) >= 2 and \
                            ccloc > spans[0][1] and \
                            ccloc < spans[-1][0]:
                        cclist = CCList(self, ccloc, seplocs,
                                        spans, tag=depth)
                        self.cclists.append(cclist)
                        cclist = None

                if tag == 0:
                    continue
                if tag == 1:  # can start a conjunction
                    if not is_conjunction:
                        is_conjunction = True
                        start_loc = i
                if tag == 2:  # starts a cclist phrase
                    ccloc, spans, seplocs = -1, [], []
                    is_conjunction = True
                    start_loc = i
                if tag == 3 and cclist != None:
                    ccloc = i
                if tag == 4 and cclist != None:
                    seplocs.append(i)
                if tag == 5:  # nothing to be done
                    continue
                if tag == 3 and cclist == None:
                    # cclist words which do not have associated spans
                    self.cclists[i] = None
