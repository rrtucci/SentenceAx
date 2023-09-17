from CCNode import *
import numpy as np
from sax_utils import get_words
from copy import deepcopy
import treelib as tr
from words_tags_ilabels_translation import *


class CCTree:
    def __init__(self, orig_sent, ll_ilabel, forced_tree=True):
        # orig_sent is a coordinated sentence, the full original sentence
        # before extractions
        self.orig_sent = orig_sent
        self.osent_words = get_words(orig_sent)
        self.ll_ilabel = ll_ilabel
        self.forced_tree = forced_tree

        # self.osent_locs = range(len(self.osent_words))

        self.ccnodes = None
        # This must be called before calling self.set_tree_structure()
        self.set_ccnodes()

        self.root_cclocs = None
        self.par_ccloc_to_child_cclocs = None
        self.child_ccloc_to_par_cclocs = None
        # this fill the 3 previous None's
        self.set_tree_structure()

        self.cc_sents = None
        # self.l_spanned_text_chunk = None

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
            return None
        elif sum(one_hot) > 1:
            print("more than one ccnode with cc at " + str(ccloc))
            assert False
        else:
            return ccnodes[unique_k]

    def fix_ccnodes(self):
        # similar to Openie6.data.coords_to_sentences
        print("nodes before fixing: ", [str(ccnode) for ccnode in
                                        self.ccnodes])
        for ccnode in self.ccnodes:
            if not self.osent_words[ccnode.ccloc] or \
                    self.osent_words[ccnode.ccloc] in ['nor', '&'] or \
                    ccnode.an_unbreakable_word_is_not_spanned():
                k = self.ccnodes.index(ccnode)
                print("node " + str(self.ccnodes[k]) +
                      " thrown away in fixing ccnodes")
                self.ccnodes.pop(k)

    def set_ccnodes(self, fix_it=False):
        """
        similar to Openie6.metric.get_coords()

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
                # print("lmk90", i, spans, ccloc, started_CP,
                #       "ilabel=", ilabel)
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
                                        spans)
                        # print("vbnk", ccnode)
                        self.ccnodes.append(ccnode)
                        # print("klfgh", len(self.ccnodes))
                        ccloc = -1
                        seplocs = []
                        spans = []
                if ilabel == 0:  # NONE
                    pass
                if ilabel == 1:  # CP
                    if not started_CP:
                        started_CP = True
                        start_loc = i
                if ilabel == 2:  # CP_START
                    # print("hjuk", "was here")
                    started_CP = True
                    start_loc = i
                if ilabel == 3:  # CC
                    ccloc = i
                if ilabel == 4:  # SEP
                    seplocs.append(i)
                if ilabel == 5:  # OTHERS
                    pass
        if fix_it:
            self.fix_ccnodes()
        # print("llm", len(self.ccnodes))
        for ccnode in self.ccnodes:
            ccnode.check_all()

    def set_tree_structure(self):
        """
        similar to Openie6.data.get_tree(conj) where conj=coords=ccnodes.
        Openie6 normally uses conj=ccloc, but not here.


        Returns
        -------

        """
        # par = parent

        # same with child and par interchanged
        self.child_ccloc_to_par_cclocs = {}
        for child_ccnode in self.ccnodes:
            child_ccloc = child_ccnode.ccloc
            par_cclocs = []
            for par_ccnode in self.ccnodes:
                par_ccloc = par_ccnode.ccloc
                if child_ccnode.is_child(par_ccnode) and \
                        par_ccloc not in par_cclocs:
                    par_cclocs.append(par_ccloc)
            self.child_ccloc_to_par_cclocs[child_ccloc] = par_cclocs

        if self.forced_tree:
            for child_ccnode in self.ccnodes:
                child_ccloc = child_ccnode.ccloc
                map = self.child_ccloc_to_par_cclocs
                # force every node to have 0 or 1 parent
                map[child_ccloc] = map[child_ccloc][:1]

        self.par_ccloc_to_child_cclocs = {}
        for par_ccnode in self.ccnodes:
            par_ccloc = par_ccnode.ccloc
            child_cclocs = []
            for child_ccnode in self.ccnodes:
                child_ccloc = child_ccnode.ccloc
                is_parent = (par_ccloc in
                             self.child_ccloc_to_par_cclocs[child_ccloc])
                if is_parent and \
                        child_ccloc not in child_cclocs:
                    child_cclocs.append(child_ccloc)
            self.par_ccloc_to_child_cclocs[par_ccloc] = child_cclocs

        self.root_cclocs = []
        for ccnode in self.ccnodes:
            ccloc = ccnode.ccloc
            if not self.child_ccloc_to_par_cclocs[ccloc]:
                self.root_cclocs.append(ccloc)

    def draw_tree(self):
        """
        important bug that must be fixed in treelib. In your Python
        installation, go to Lib\site-packages\treelib and edit tree.py. Find
        def show. The last line is:

        print(self.reader.encode('utf-8'))

        It should be:

        print(self.reader)



        Returns
        -------

        """
        try:
            # print("ch-to-par", self.child_ccloc_to_par_cclocs)
            # print("par-to-ch", self.par_ccloc_to_child_cclocs)
            tree = tr.Tree()
            for child_ccloc, par_cclocs in self.child_ccloc_to_par_cclocs.items():
                # print("lmkp", str(child_ccloc), str(par_cclocs))
                child_ccnode = self.get_ccnode_from_ccloc(child_ccloc,
                                                          self.ccnodes)
                if child_ccnode:
                    child_name = str(child_ccnode)
                    if not par_cclocs:
                        tree.create_node(child_name, child_name)
                    else:
                        for par_ccloc in par_cclocs:
                            par_ccnode = self.get_ccnode_from_ccloc(par_ccloc,
                                                                    self.ccnodes)
                            # print("hgfd", str(par_ccloc))
                            if par_ccnode:
                                # print("lmjk", child_ccloc, par_ccloc)
                                par_name = str(par_ccnode)
                                # print("hjdf", child_name, par_name)
                                tree.create_node(child_name,
                                                 child_name,
                                                 parent=par_name)

            tree.show()
            return True
        except:
            print("*********************tree not possible")
            print("par_ccloc_to_child_cclocs=",
                  self.par_ccloc_to_child_cclocs)
            return False

    def refresh_ll_spanned_loc(self,
                               ll_spanned_loc,
                               level_ccnodes,
                               level):
        """
        similar to Openie6.data.get_sentences(sentences,
                  conj_same_level,
                  conj_coords,
                  sentence_indices)
        doesn't return anything but changes  sentences

        conj = ccloc, conjunct = spans, coord = ccnode
        sentences = ll_spanned_loc
        sentence = level_spanned_locs
        conj_same_level = level_cclocs
        conj_coords = swaps
        sentence_indices = osent_locs

        level = same/equal level
        li = list


        Parameters
        ----------
        level_ccnodes
        osent_locs

        Returns
        -------

        """
        # print("level=", level)
        # print("ll_spanned_loc", ll_spanned_loc)
        k = 0
        # print("num_nodes", len(self.ccnodes))
        for ccnode in level_ccnodes:
            # print("nnml", "node_id", k)
            k += 1
            fat_spanned_locs = ccnode.get_spanned_locs(fat=True)
            if not ll_spanned_loc:
                ll_spanned_loc.append(fat_spanned_locs)
            else:
                # to_be_added_ll_loc = []
                # to_be_removed_ll_loc = []
                for spanned_locs in ll_spanned_loc:
                    # print("bhk", spanned_locs)
                    # only ccnodes that satisfy this have fat-spanned_locs
                    if ccnode.spans[0][0] in spanned_locs:
                        # to_be_added_ll_loc.append(fat_spanned_locs)
                        # to_be_removed_ll_loc.append(spanned_locs)
                        if spanned_locs in ll_spanned_loc:
                            ll_spanned_loc.remove(spanned_locs)
                        ll_spanned_loc.append(fat_spanned_locs)

                # for l_loc in to_be_removed_ll_loc:
                #     ll_spanned_loc.remove(l_loc)
                # for l_loc in to_be_added_ll_loc:
                #     ll_spanned_loc.append(l_loc)
        print("new_ll_spanned_loc", ll_spanned_loc)
        return ll_spanned_loc

    def set_cc_sents(self):
        """
        similar to Openie6.data.coords_to_sentences()

        Returns
        -------

        """
        # self.fix_ccnodes()  was called at the end of get_ccnodes()

        l_spanned_text_chunk = []
        for ccnode in self.ccnodes:
            for span in ccnode.spans:
                l_spanned_text_chunk.append(
                    ' '.join(self.osent_words[span[0]:span[1]]))

        level_nd_count = len(self.root_cclocs)
        rooty_cclocs = copy(self.root_cclocs)
        next_level_nd_count = 0

        level_ccnodes = []
        ll_spanned_loc = []  # node,num_locs
        level_to_ll_spanned_loc = {}

        # self.root_cclocs was filled by __init__
        level = 0
        while len(rooty_cclocs) > 0:
            print("****************************")
            print("rooty_cclocs", rooty_cclocs)
            print("level, nd_count, next_nd_count", level, "/",
                  level_nd_count,
                  next_level_nd_count)
            print("ll_spanned_loc", ll_spanned_loc)
            print("level_cc_nodes", [str(x) for x in level_ccnodes])

            rooty_ccloc = rooty_cclocs.pop(0)
            rooty_ccnode = CCTree.get_ccnode_from_ccloc(rooty_ccloc,
                                                        self.ccnodes)

            # nd=node
            level_nd_count -= 1
            level_ccnodes.append(rooty_ccnode)
            ll_spanned_loc = [ccnode.spanned_locs for ccnode in level_ccnodes]
            print("level_cc_nodes", [str(x) for x in level_ccnodes])
            print("ll_spanned_loc", ll_spanned_loc)

            for child_ccloc in \
                    self.par_ccloc_to_child_cclocs[rooty_ccloc]:
                # child becomes new root as tree is pared down
                rooty_cclocs.append(child_ccloc)
                next_level_nd_count += 1
            print("level, nd_count, next_nd_count", level, "/",
                  level_nd_count,
                  next_level_nd_count)

            if level_nd_count == 0:
                ll_spanned_loc = \
                    self.refresh_ll_spanned_loc(
                        ll_spanned_loc,
                        level_ccnodes,
                        level)
                if level not in level_to_ll_spanned_loc.keys():
                    level_to_ll_spanned_loc[level] = []
                level_to_ll_spanned_loc[level]+=ll_spanned_loc
                level += 1
                level_nd_count = next_level_nd_count
                next_level_nd_count = 0
                level_ccnodes = []
                ll_spanned_loc = []
                print("level, nd_count, next_nd_count", level, "/",
                      level_nd_count,
                      next_level_nd_count)
        # if level not in level_to_ll_spanned_loc.keys():
        #     level_to_ll_spanned_loc[level] = []
        # level_to_ll_spanned_loc[level] += ll_spanned_loc
        # print("level_to_ll_spanned_loc", level_to_ll_spanned_loc)

        print("bnnnn", level_to_ll_spanned_loc)
        cc_sents = []
        for level, ll_spanned_loc in level_to_ll_spanned_loc.items():
            for spanned_locs in ll_spanned_loc:
                cc_sent = ' '.join([self.osent_words[i]
                                    for i in sorted(spanned_locs)])
                cc_sents.append(cc_sent)
        self.cc_sents = cc_sents
        # self.l_spanned_text_chunk = l_spanned_text_chunk

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


if __name__ == "__main__":
    def main1():
        in_fp = "testing_files/small_cctags.txt"
        out_fp = "testing_files/cc_ilabels.txt"
        file_translate_tags_to_ilabels("cc", in_fp, out_fp)


    def main2(forced_tree=True):
        in_fp = "testing_files/one_sample_cc_ilabels.txt"
        # out_fp = "testing_files/cc_trees.txt"
        with open(in_fp, "r", encoding="utf-8") as f:
            in_lines = f.readlines()

        l_osent = []
        lll_ilabel = []
        ll_ilabel = []
        for in_line in in_lines:
            if in_line:
                if in_line[0].isalpha():
                    l_osent.append(in_line.strip())
                    if ll_ilabel:
                        lll_ilabel.append(ll_ilabel)
                    ll_ilabel = []
                elif in_line[0].isdigit():
                    words = get_words(in_line)
                    # print("lkll", words)
                    ll_ilabel.append([int(x) for x in words])
        # last one
        if ll_ilabel:
            lll_ilabel.append(ll_ilabel)
        #
        # print("lklo", l_osent)
        # print("lklo", lll_ilabel)
        for k in range(len(l_osent)):
            osent = l_osent[k]
            print(osent)
            tree = CCTree(osent, lll_ilabel[k], forced_tree)
            tree.draw_tree()
            for k, sent in enumerate(tree.cc_sents):
                print(str(k + 1) + ". " + sent)
            print()


    main1()
    # main2()
