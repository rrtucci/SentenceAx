from CCNode import *
from numpy import np
from sax_utils import get_words


class CCTree:
    def __init__(self, ccsent):
        # ccsent is a coordinated sentence, the full original sentence
        # before extractions
        self.words = get_words(ccsent)
        self.extra_locs = []

        self.ccnodes = None
        self.set_ccnodes()

        self.root_cclocs = []
        self.par_ccloc_to_child_cclocs = {}
        self.child_ccloc_to_par_cclocs = {}
        self.set_tree_structure()


    def fix_ccnodes(self):
        for ccnode in self.ccnodes:
            if self.words[ccnode.ccloc] in ['nor', '&'] or \
                    ccnode.contains_unbreakable_spans():
                k = self.ccnodes.index(ccnode)
                self.ccnodes.pop(k)

    def set_ccnodes(self, depth_to_tags): # get_ccnodes()
        self.ccnodes = []

        for depth in range(len(depth_to_tags)):
            ccnode = None
            spans = []
            start_loc = -1
            is_conjunction = False
            tags = depth_to_tags[depth]

            for i, tag in enumerate(tags):
                if tag != 1:  # conjunction can end
                    if is_conjunction and ccnode != None:
                        is_conjunction = False
                        spans.append((start_loc, i - 1))
                if tag == 0 or tag == 2:  # ccnode phrase can end
                    if ccnode and \
                            len(spans) >= 2 and \
                            ccloc > spans[0][1] and \
                            ccloc < spans[-1][0]:
                        ccnode = CCNode(self, ccloc, seplocs,
                                       spans, tag=depth)
                        self.ccnodes.append(ccnode)
                        ccnode = None

                if tag == 0:
                    continue
                if tag == 1:  # can start a conjunction
                    if not is_conjunction:
                        is_conjunction = True
                        start_loc = i
                if tag == 2:  # starts a ccnode phrase
                    ccloc, spans, seplocs = -1, [], []
                    is_conjunction = True
                    start_loc = i
                if tag == 3 and ccnode != None:
                    ccloc = i
                if tag == 4 and ccnode != None:
                    seplocs.append(i)
                if tag == 5:  # nothing to be done
                    continue
                if tag == 3 and ccnode == None:
                    # ccnode words which do not have associated spans
                    self.ccnodes[i] = None

        self.fix_ccnodes()

    def set_tree_structure(self):
        # par = parent
        self.root_cclocs = []
        self.par_ccloc_to_child_cclocs = {}
        self.child_ccloc_to_par_cclocs = {}

        for par_ccnode in self.ccnodes:
            par_ccloc = par_ccnode.ccloc
            child_cclocs_list = []
            for child_ccnode in self.ccnodes:
                child_ccloc = child_ccnode.ccloc
                if par_ccnode.is_parent(child_ccnode):
                    child_cclocs_list.append(child_ccloc)
            child_cclocs_list.sort()
            self.par_ccloc_to_child_cclocs[par_ccloc] = child_cclocs_list

        # tree so allow only one parent for each node

        child_cclocs_list = self.par_ccloc_to_child_cclocs.values().sort(
            key=list.__len__)
        # consider i < j
        for i in range(0, len(child_cclocs_list)):
            for j in range(i + 1, len(child_cclocs_list)):
                intersection = \
                    list(set(child_cclocs_list[i]) & set(child_cclocs_list[j]))
                child_cclocs_list[j] -= intersection

        # same as before but with par and child swapped
        for child_ccnode in self.ccnodes:
            child_ccloc = child_ccnode.ccloc
            par_cclocs = []
            for par_ccnode in self.ccnodes:
                par_ccloc = par_ccnode.ccloc
                if child_ccnode.is_childent(par_ccnode):
                    par_cclocs.append(par_ccloc)
            assert len(par_cclocs) == 1  # tree so only one parent per node
            self.child_ccloc_to_par_cclocs[child_ccloc] = par_cclocs


        for ccnode in self.ccnodes:
            ccloc = ccnode.ccloc
            if not self.child_ccloc_to_par_cclocs[ccloc]:
                self.root_cclocs.append(ccloc)


    # get_sentences()
    def get_spanned_locs_list(self, ccnodes_same_level, extra_locs):

        spanned_locs_list = []
        for ccnode in ccnodes_same_level:

            if len(spanned_locs_list) == 0:
                ccnode.set_spanned_locs(extra_locs)
                spanned_locs_list.append(ccnode.spanned_locs)

            else:
                ccnode.set_spanned_locs(None)
                spanned_locs_list.append(ccnode.spanned_locs)

        return spanned_locs_list

                # to_be_added_loc_lists = []
                # to_be_removed_loc_lists = []
                # for spanned_locs in spanned_locs_list:
                #     if ccnode.spans[0][0] in spanned_locs:
                #         spanned_locs.sort()
                #         new_spanned_locs = ccnode.get_spanned_locs()
                #         to_be_added_loc_lists.append(new_spanned_locs)
                #         to_be_removed_loc_lists.append(spanned_locs)
                #
                #
                # for loc_list in to_be_removed_loc_lists:
                #     spanned_locs_list.remove(loc_list)
                # spanned_locs_list.extend(to_be_added_loc_lists)

    def get_spanned_phrases(self):  # ccnodes_to_sentences()

        spanned_words = []
        for ccnode in self.ccnodes:
            for span in ccnode.spans:
                spanned_words.append(' '.join(self.words[span[0]:span[1]]))

        spanned_locs_list = []
        root_count = len(self.root_cclocs)
        new_child_count = 0

        ccnodes_same_level = []

        while len(self.root_cclocs) > 0:

            self.root_cclocs.pop(0)
            root_count -= 1
            ccnodes_same_level.append(self.root_cclocs)

            for child_ccloc in self.par_ccloc_to_child_cclocs[
                self.root_cclocs]:
                self.root_cclocs.append(child_ccloc)
                new_child_count += 1

            if root_count == 0:
                spanned_locs_list = \
                    self.get_spanned_locs_list(ccnodes_same_level,
                                            self.extra_locs)
                root_count = new_child_count
                new_child_count = 0
                ccnodes_same_level = []
        spanned_phrases = [' '.join([self.words[i] for i in
            sorted(spanned_locs)]) for spanned_locs in spanned_locs_list]

        return spanned_phrases

    # def get_shifted_ccnodes(self, arr):  # post_process()
    #     new_ccnodes = []
    #     # arr is 1 dim numpy array
    #     # `np.argwhere(arr)` returns indices, as a column,
    #     # of entries of arr are >0
    #
    #     # np.argwhere([0, 5, 8, 0]) = [1, 2]
    #     # [0, 5, 8, 0].cumsum() = [0, 5, 13, 13]
    #     # np.delete([0, 5, 13, 13], [1, 2]) = [0, 13]
    #     shift = np.delete(arr.cumsum(), index=np.argwhere(arr))
    #     for ccnode in self.ccnodes:
    #         ccloc = ccnode.ccloc + shift[ccnode.ccloc]
    #         spans = [(b + shift[b], e + shift[e])
    #                  for (b, e) in ccnode.spans]
    #         seps_locs = [s + shift[s] for s in ccnode.seps_locs]
    #         ccnode = CCNode(ccloc, spans, seps_locs, ccnode.tag)
    #         new_ccnodes.append(ccnode)
    #     return new_ccnodes
