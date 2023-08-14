from CCNode import *
from numpy import np
from sax_utils import get_words
from copy import deepcopy


class CCTree:
    def __init__(self, orig_sent, predictions_for_each_depth):
        # orig_sent is a coordinated sentence, the full original sentence
        # before extractions
        self.orig_sent = orig_sent
        self.extra_locs = []

        self.ccnodes = None
        # This must be called before calling self.set_tree_structure()
        self.set_ccnodes(predictions_for_each_depth)

        self.root_cclocs = []
        self.par_ccloc_to_child_cclocs = {}
        self.child_ccloc_to_par_cclocs = {}
        self.set_tree_structure()

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
        if sum(one_hot)==0:
            print(str(ccloc) + " is not a location of a cc.")
            assert False
        if sum(one_hot)>1:
            print("more than one ccnode with cc at " + str(ccloc))
            assert False
        return ccnodes[unique_k]

    def fix_ccnodes(self):
        words = get_words(self.orig_sent)
        for ccnode in self.ccnodes:
            if not words[ccnode.ccloc] or \
                    words[ccnode.ccloc] in ['nor', '&'] or \
                    ccnode.omits_unbreakable_words():
                k = self.ccnodes.index(ccnode)
                self.ccnodes.pop(k)

    def set_ccnodes(self, predictions_for_each_depth):
        """
        formerly metric.get_coords()

        Parameters
        ----------
        predictions_for_each_depth

        Returns
        -------

        """
        self.ccnodes = []

        for depth in range(len(predictions_for_each_depth)):
            ccnode = None
            start_loc = -1
            is_CP = False  # CP stands for coordinating phrase
            predictions = predictions_for_each_depth[depth]

            # cctag_to_int = {
            #   'NONE': 0
            #   'CP': 1,
            #   'CP_START': 2,
            #    'CC': 3,
            #    'SEP': 4,
            #    'OTHERS': 5
            # }

            for i, prediction in enumerate(predictions):
                if prediction != 1:  # CP
                    if is_CP and ccnode != None:
                        is_CP = False
                        if not ccnode.spans:
                            ccnode.spans = []
                        ccnode.spans.append((start_loc, i - 1))
                if prediction == 0 or prediction == 2:  # NONE or CP_START
                    # ccnode phrase can end
                    if ccnode and \
                            len(ccnode.spans) >= 2 and \
                            ccnode.ccloc >= ccnode.spans[0][1] and \
                            ccnode.ccloc < ccnode.spans[-1][0]:
                        self.ccnodes.append(deepcopy(ccnode))
                        ccnode = None
                if prediction == 0:  # NONE
                    continue
                if prediction == 1:  # CP
                    if not is_CP:
                        is_CP = True
                        start_loc = i
                if prediction == 2:  # CP_START
                    words = get_words(self.orig_sent)
                    ccnode = CCNode(depth)
                    is_CP = True
                    start_loc = i
                if prediction == 3:  # CC
                    if ccnode != None:
                        ccnode.ccloc = i
                    else:
                        # ccnode words which do not have associated spans
                        self.ccnodes[i] = None
                if prediction == 4 and ccnode != None:  # SEP
                    if not ccnode.seplocs:
                        ccnode.seplocs = []
                    ccnode.seplocs.append(i)
                if prediction == 5:  # OTHERS
                    continue
        self.fix_ccnodes()
        for ccnode in self.ccnodes:
            ccnode.check_all()

    def set_tree_structure(self):
        """
        formerly data.get_tree(conj) where conj=coords=ccnodes.
        Openie6 normally uses conj=ccloc, but not here.


        Returns
        -------

        """
        # par = parent

        self.root_cclocs = []
        self.par_ccloc_to_child_cclocs = {}
        self.child_ccloc_to_par_cclocs = {}
        for par_ccnode in self.ccnodes:
            par_ccloc = par_ccnode.ccloc
            child_cclocs = []
            for child_ccnode in self.ccnodes:
                child_ccloc = child_ccnode.ccloc
                if par_ccnode.is_parent(child_ccnode):
                    child_cclocs.append(child_ccloc)
            self.par_ccloc_to_child_cclocs[par_ccloc] = child_cclocs

        # this is tree so allow only one parent for each node

        child_cclocs = sorted(self.par_ccloc_to_child_cclocs.values(),
            key=lambda x: len(x))
        # consider i < j
        for i in range(0, len(child_cclocs)):
            for j in range(i + 1, len(child_cclocs)):
                intersection = \
                    list(set(child_cclocs[i]) & set(child_cclocs[j]))
                child_cclocs[j] -= intersection

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

    def refresh_ll_eqlevel_spanned_loc(self,
                                        ll_eqlevel_spanned_loc,
                                        # eqlevel_cclocs,
                                        eqlevel_ccnodes,
                                        extra_locs):
        """
        formerly  data.get_sentences(sentences,
                  conj_same_level,
                  conj_coords,
                  sentence_indices)
        doesn't return anything but changes  sentences

        conj = ccloc, conjunct = spans, coord = ccnode
        sentences = ll_eqlevel_spanned_loc
        sentence = eqlevel_spanned_locs
        conj_same_level = eqlevel_cclocs
        conj_coords = swaps
        sentence_indices = extra_locs

        eqlevel = same/equal level
        li = list


        Parameters
        ----------
        eqlevel_ccnodes
        extra_locs

        Returns
        -------

        """
        for ccnode in eqlevel_ccnodes:
            if len(ll_eqlevel_spanned_loc) == 0:
                spanned_locs = ccnode.get_spanned_locs(extra_locs)
                ll_eqlevel_spanned_loc.append(spanned_locs)
            else:
                to_be_added_loc_lists = []
                to_be_removed_loc_lists = []
                for spanned_locs in ll_eqlevel_spanned_loc:
                    if ccnode.spans[0][0] in spanned_locs:
                        spanned_locs.sort()
                        min = ccnode.spans[0][0]
                        max = ccnode.spans[-1][-1] - 1

                        for span in ccnode.spans:
                            new_spanned_locs = []
                            for i in spanned_locs:
                                if i in range(span[0], span[1] + 1) or \
                                        i < min or i > max:
                                    new_spanned_locs.append(i)

                            to_be_added_loc_lists.append(new_spanned_locs)

                        to_be_removed_loc_lists.append(spanned_locs)

                for loc_list in to_be_removed_loc_lists:
                    ll_eqlevel_spanned_loc.remove(loc_list)
                for loc_list in to_be_added_loc_lists:
                    ll_eqlevel_spanned_loc.append(loc_list)

        return ll_eqlevel_spanned_loc

    def get_simple_sents(self):
        """
        formerly data.coords_to_sentences()

        Returns
        -------

        """
        # self.fix_ccnodes()  was called at the end of get_ccnodes()

        orig_words = get_words(self.orig_sent)
        spanned_words = []
        for ccnode in self.ccnodes:
            for span in ccnode.spans:
                spanned_words.append(' '.join(orig_words[span[0]:span[1]]))

        li_spanned_locs = []
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
                li_spanned_locs = \
                    self.refresh_ll_eqlevel_spanned_loc(
                        li_spanned_locs,
                        eqlevel_ccnodes,
                        self.extra_locs)
                root_count = new_child_count
                new_child_count = 0
                eqlevel_ccnodes = []
        simple_sents = []
        for spanned_locs in li_spanned_locs:
            simple_sent = \
                ' '.join([orig_words[i] for i in sorted(spanned_locs)])
            simple_sents.append(simple_sent)

        return simple_sents, spanned_words, li_spanned_locs


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
