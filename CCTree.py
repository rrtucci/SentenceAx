from CCNode import *
import numpy as np
from sax_utils import get_words
from copy import deepcopy


class CCTree:
    def __init__(self, orig_sent, ll_ilabel):
        # orig_sent is a coordinated sentence, the full original sentence
        # before extractions
        self.orig_sent = orig_sent
        self.osent_words = get_words(orig_sent)
        self.ll_ilabel = ll_ilabel
        self.osent_locs = range(len(self.osent_words))

        self.ccnodes = None
        # This must be called before calling self.set_tree_structure()
        self.set_ccnodes()

        self.root_cclocs = None
        self.par_ccloc_to_child_cclocs = None
        self.child_ccloc_to_par_cclocs = None
        # this fill the 3 previous None's
        self.set_tree_structure()

        self.cc_sents = None
        self.l_spanned_text_chunk = None
        self.fat_l_spanned_locs = None

        # this fills the 3 previous None's
        self.set_cc_sents()

    @staticmethod
    def get_ccnode_from_ccloc(ccloc, ccnodes):
        unique_k = -1
        one_hot = []
        for k, ccnode in enumerate(ccnodes):
            if ccnode.ccloc == ccloc:
                one_hot.append(1)
                unique_k = k
            else:
                one_hot.append(0)
        if sum(one_hot) == 0:
            print(str(ccloc) + " is not a location of a cc.")
            assert False
        if sum(one_hot) > 1:
            print("more than one ccnode with cc at " + str(ccloc))
            assert False
        return ccnodes[unique_k]

    def fix_ccnodes(self):
        # similar to data.coords_to_sentences
        for ccnode in self.ccnodes:
            if not self.osent_words[ccnode.ccloc] or \
                    self.osent_words[ccnode.ccloc] in ['nor', '&'] or \
                    ccnode.an_unbreakable_word_is_not_spanned():
                k = self.ccnodes.index(ccnode)
                self.ccnodes.pop(k)

    def set_ccnodes(self):
        """
        similar to metric.get_coords()

        Parameters
        ----------
        ll_label

        Returns
        -------

        """
        self.ccnodes = []

        for depth in range(len(self.ll_ilabel)):
            start_loc = -1
            started_CP = False  # CP stands for coordinating phrase
            l_ilabel = self.ll_ilabel[depth]
            ccloc = -1
            seplocs = []
            spans = []

            # cctag_to_int = {
            #   'NONE': 0
            #   'CP': 1,
            #   'CP_START': 2,
            #    'CC': 3,
            #    'SEP': 4,
            #    'OTHERS': 5
            # }

            for i, ilabel in enumerate(l_ilabel):
                if ilabel != 1:  # CP
                    if started_CP:
                        started_CP = False
                        spans.append((start_loc, i))
                if ilabel == 0 or ilabel == 2:  # NONE or CP_START
                    # ccnode phrase can end
                    # two spans at least, split by CC
                    if spans and len(spans) >= 2 and \
                            ccloc >= spans[0][1] and \
                            ccloc < spans[-1][0]:
                        ccnode = CCNode(ccloc,
                                        depth,
                                        self.osent_words,
                                        seplocs,
                                        spans,
                                        self.osent_locs)
                        self.ccnodes.append(ccnode)
                        ccloc = -1
                        seplocs = []
                        spans = []
                elif ilabel == 0:  # NONE
                    continue
                elif ilabel == 1:  # CP
                    if not started_CP:
                        started_CP = True
                        start_loc = i
                elif ilabel == 2:  # CP_START
                    started_CP = True
                    start_loc = i
                elif ilabel == 3:  # CC
                    ccloc = i
                elif ilabel == 4:  # SEP
                    seplocs.append(i)
                elif ilabel == 5:  # OTHERS
                    continue
                else:
                    assert False
        self.fix_ccnodes()
        for ccnode in self.ccnodes:
            ccnode.check_all()

    def set_tree_structure(self):
        """
        similar to data.get_tree(conj) where conj=coords=ccnodes.
        Openie6 normally uses conj=ccloc, but not here.


        Returns
        -------

        """
        # par = parent

        self.par_ccloc_to_child_cclocs = {}
        for par_ccnode in self.ccnodes:
            par_ccloc = par_ccnode.ccloc
            child_cclocs = []
            for child_ccnode in self.ccnodes:
                child_ccloc = child_ccnode.ccloc
                if par_ccnode.is_parent(child_ccnode):
                    child_cclocs.append(child_ccloc)
            self.par_ccloc_to_child_cclocs[par_ccloc] = child_cclocs

        # this is tree so allow only one parent for each node
        self.child_ccloc_to_par_cclocs = {}
        for child_ccloc in self.ccnodes:
            self.child_ccloc_to_par_cclocs[child_ccloc] = []
            for par_ccloc, child_cclocs in self.par_ccloc_to_child_cclocs:
                if child_ccloc in child_cclocs:
                    self.child_ccloc_to_par_cclocs[child_ccloc]. \
                        append(par_ccloc)

        self.root_cclocs = []
        for ccnode in self.ccnodes:
            ccloc = ccnode.ccloc
            if not self.child_ccloc_to_par_cclocs[ccloc]:
                self.root_cclocs.append(ccloc)

    @staticmethod
    def get_fat_spanned_locs(ccnode, spanned_locs):
        spanned_locs.sort()
        min = ccnode.spans[0][0]
        max = ccnode.spans[-1][1] - 1
        fat_spanned_locs = []
        for span in ccnode.spans:
            fat_spanned_locs = []
            for i in spanned_locs:
                if i in range(span[0], span[1]) or \
                        i < min or i > max:
                    fat_spanned_locs.append(i)
        return fat_spanned_locs

    def refresh_eqlevel_ll_spanned_loc(self,
                                       eqlevel_ll_spanned_loc,
                                       eqlevel_ccnodes):
        """
        similar to  data.get_sentences(sentences,
                  conj_same_level,
                  conj_coords,
                  sentence_indices)
        doesn't return anything but changes  sentences

        conj = ccloc, conjunct = spans, coord = ccnode
        sentences = eqlevel_ll_spanned_loc
        sentence = eqlevel_spanned_locs
        conj_same_level = eqlevel_cclocs
        conj_coords = swaps
        sentence_indices = osent_locs

        eqlevel = same/equal level
        li = list


        Parameters
        ----------
        eqlevel_ccnodes
        osent_locs

        Returns
        -------

        """
        for ccnode in eqlevel_ccnodes:
            if len(eqlevel_ll_spanned_loc) == 0:
                spanned_locs = ccnode.get_spanned_locs(self.osent_locs)
                eqlevel_ll_spanned_loc.append(spanned_locs)
            else:
                to_be_added_ll_loc = []
                to_be_removed_ll_loc = []
                for spanned_locs in eqlevel_ll_spanned_loc:
                    if ccnode.spans[0][0] in spanned_locs:
                        fat_spanned_locs = \
                            CCTree.get_fat_spanned_locs(ccnode,
                                                        spanned_locs)

                        to_be_added_ll_loc.append(fat_spanned_locs)

                        to_be_removed_ll_loc.append(spanned_locs)

                for l_loc in to_be_removed_ll_loc:
                    eqlevel_ll_spanned_loc.remove(l_loc)
                for l_loc in to_be_added_ll_loc:
                    eqlevel_ll_spanned_loc.append(l_loc)

        return eqlevel_ll_spanned_loc


    def set_cc_sents(self):
        """
        similar to data.coords_to_sentences()

        Returns
        -------

        """
        # self.fix_ccnodes()  was called at the end of get_ccnodes()

        l_spanned_text_chunk = []
        for ccnode in self.ccnodes:
            for span in ccnode.spans:
                l_spanned_text_chunk.append(
                    ' '.join(self.osent_words[span[0]:span[1]]))

        fat_l_spanned_locs = []
        root_count = len(self.root_cclocs)
        new_child_count = 0

        eqlevel_ccnodes = []

        # self.root_cclocs was filled by __init__
        while len(self.root_cclocs) > 0:

            root_ccloc = self.root_cclocs.pop(0)
            root_ccnode = CCTree.get_ccnode_from_ccloc(root_ccloc,
                                                       self.ccnodes)
            root_count -= 1
            eqlevel_ccnodes.append(root_ccnode)

            for child_ccloc in \
                    self.par_ccloc_to_child_cclocs[root_ccloc]:
                # child becomes new root as tree is pared down
                self.root_cclocs.append(child_ccloc)
                new_child_count += 1

            if root_count == 0:
                fat_l_spanned_locs = \
                    self.refresh_eqlevel_ll_spanned_loc(
                        eqlevel_ll_spanned_loc=[],
                        eqlevel_ccnodes= eqlevel_ccnodes)
                root_count = new_child_count
                new_child_count = 0
                eqlevel_ccnodes = []
        cc_sents = []
        for spanned_locs in fat_l_spanned_locs:
            cc_sent = \
                ' '.join([self.osent_words[i] for i in sorted(
                    spanned_locs)])
            cc_sents.append(cc_sent)
        self.cc_sents = cc_sents
        self.l_spanned_text_chunk = l_spanned_text_chunk
        self.fat_l_spanned_locs = fat_l_spanned_locs


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
