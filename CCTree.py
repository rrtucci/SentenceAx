from CCNode import *
import numpy as np
from sax_utils import get_words
from copy import deepcopy
import treelib as tr
from words_tags_ilabels_translation import *


class CCTree:
    """
    
    Attributes
    ----------
    calc_tree_struc: bool
    ccnodes: list[CCNode]
    ccsents: list[str]
    child_ccloc_to_par_cclocs: dict[int, list[int]]
    forced_tree: bool
    level_to_ccnodes: dict[int, list[Node]]
    level_to_fat_ll_spanned_loc: dict[int, list[list[int]]]
    ll_ilabel: list[list[int]]
    orig_sent: str
    osent_words: list[str]
    par_ccloc_to_child_cclocs: list[int,list[int]]
    root_cclocs: list[int]
    verbose: str
    
    """

    def __init__(self, orig_sent, ll_ilabel, forced_tree=True,
                 calc_tree_struc=True, verbose=False):
        """
        orig_sent is a coordinated sentence, the full original sentence
        before extractions

        Parameters
        ----------
        orig_sent: str
        ll_ilabel: list[list[int]]
        forced_tree: bool
        calc_tree_struc: bool
        """
        self.orig_sent = orig_sent
        self.osent_words = get_words(orig_sent)
        self.ll_ilabel = ll_ilabel
        self.forced_tree = forced_tree
        self.calc_tree_struc = calc_tree_struc
        self.verbose = verbose

        # self.osent_locs = range(len(self.osent_words))

        self.ccnodes = None
        # This must be called before calling self.set_tree_structure()
        self.set_ccnodes()

        self.root_cclocs = None
        self.par_ccloc_to_child_cclocs = None
        self.child_ccloc_to_par_cclocs = None
        # this fill the 3 previous None's
        if calc_tree_struc:
            self.set_tree_structure()

        self.ccsents = None
        self.level_to_ccnodes = None
        self.level_to_fat_ll_spanned_loc = None

        self.ll_spanned_loc = None
        self.l_spanned_word = None

        # this fills the previous unfilled None's
        if calc_tree_struc:
            self.set_ccsents()

    @staticmethod
    def get_ccnode_from_ccloc(ccloc, ccnodes):
        """

        Parameters
        ----------
        ccloc: int
        ccnodes: list[CCNode]

        Returns
        -------
        CCNode|None, list[CCNode]

        """
        unique_k = -1
        l_hot_k = []
        bad_ccnodes = []
        for k, ccnode in enumerate(ccnodes):
            if ccnode.ccloc == ccloc:
                l_hot_k.append(k)
                unique_k = k
        if not l_hot_k:
            return None
        elif len(l_hot_k) > 1:
            # this normally doesn't happen with training extractions
            # but it can happen with predicted extractions
            print("more than one ccnode with cc at " + str(ccloc))
            print("culprit sent:\n" + str(ccnodes[0].osent_words))
            print("ccnodes[k].spanned_locs:")
            for k in l_hot_k:
                print("k=" + str(k) + ", " + str(ccnodes[k].spanned_locs))
            assert False
        else:
            return ccnodes[unique_k]


    def remove_bad_ccnodes(self):
        """
        similar to Openie6.data.coords_to_sentences

        Returns
        -------
        None

        """
        if self.verbose:
            print("nodes before removals: ", [str(ccnode) for ccnode in
                                            self.ccnodes])
        # one to one mapping between ccnodes and cclocs
        ccloc_to_ccnode = {}
        for ccnode in self.ccnodes:
            if ccnode.ccloc not in ccloc_to_ccnode.keys():
                ccloc_to_ccnode[ccnode.ccloc] = []
            ccloc_to_ccnode[ccnode.ccloc].append(ccnode)
        for ccloc in ccloc_to_ccnode.keys():
            if len(ccloc_to_ccnode[ccloc]) > 1:
                for k, ccnode in enumerate(ccloc_to_ccnode[ccloc]):
                    if k >=1:
                        if self.verbose:
                            print("node " + str(ccnode) + " removed")
                        self.ccnodes.remove(ccnode)

        for ccnode in self.ccnodes:
            if ccnode.ccloc >= len(self.osent_words) or \
                    not self.osent_words[ccnode.ccloc] or \
                    self.osent_words[ccnode.ccloc] in ['nor', '&'] or \
                    ccnode.an_unbreakable_word_is_not_spanned():
                if self.verbose:
                    print("node " + str(ccnode) + " removed")
                self.ccnodes.remove(ccnode)

    def set_ccnodes(self):
        """
        similar to Openie6.metric.get_coords()

        Returns
        -------
        None

        """
        self.ccnodes = []

        for depth in range(len(self.ll_ilabel)):
            start_loc = -1
            started_CP = False  # CP stands for coordinating phrase
            l_ilabel = self.ll_ilabel[depth]
            ccloc = -1
            seplocs = []
            spans = []

            # CCTAG_TO_ILABEL = {
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
                if ilabel in [0, 2]:  # NONE or CP_START
                    # ccnode phrase can end
                    # two spans at least, split by CC
                    if spans and len(spans) >= 2 and \
                            spans[0][1] <= ccloc < spans[-1][0]:
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
                elif ilabel == 1:  # CP
                    if not started_CP:
                        started_CP = True
                        start_loc = i
                elif ilabel == 2:  # CP_START
                    # print("hjuk", "was here")
                    started_CP = True
                    start_loc = i
                elif ilabel == 3:  # CC
                    ccloc = i
                elif ilabel == 4:  # SEP
                    seplocs.append(i)
                elif ilabel == 5:  # OTHERS
                    pass
                elif ilabel == -100:
                    pass
                else:
                    assert False, f"{str(ilabel)} out of range 0:6"
        self.remove_bad_ccnodes()
        # print("llm", len(self.ccnodes))
        for ccnode in self.ccnodes:
            ccnode.check_all()

    def set_tree_structure(self):
        """
        similar to Openie6.data.get_tree(conj) where conj=coords=ccnodes.
        Openie6 normally uses conj=ccloc, but not here.


        Returns
        -------
        None

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
                mapa = self.child_ccloc_to_par_cclocs
                # force every node to have 0 or 1 parent
                mapa[child_ccloc] = mapa[child_ccloc][:1]

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
        None

        """
        try:
            # print("ch-to-par", self.child_ccloc_to_par_cclocs)
            # print("par-to-ch", self.par_ccloc_to_child_cclocs)
            tree = tr.Tree()
            for child_ccloc, par_cclocs in \
                    self.child_ccloc_to_par_cclocs.items():
                # print("lmkp", str(child_ccloc), str(par_cclocs))
                child_ccnode = CCTree.get_ccnode_from_ccloc(child_ccloc,
                                                          self.ccnodes)
                if child_ccnode:
                    child_name = str(child_ccnode)
                    if not par_cclocs:
                        tree.create_node(child_name, child_name)
                    else:
                        for par_ccloc in par_cclocs:
                            par_ccnode = CCTree.get_ccnode_from_ccloc(
                                par_ccloc,
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

    @staticmethod
    def fatten_ll_spanned_loc(ll_spanned_loc,
                              level_ccnodes,
                              level,
                              verbose):
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
        ll_spanned_loc: list[list[int]]
        level_ccnodes: list[CCNode]
        level: int

        Returns
        -------
        list[list[int]]

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
        if verbose:
            print("new_ll_spanned_loc", ll_spanned_loc)
        return ll_spanned_loc

    def set_ccsents(self):
        """
        similar to Openie6.data.coords_to_sentences()

        Returns
        -------
        None

        """
        # self.remove_bad_ccnodes()  was called at the end of get_ccnodes()

        l_spanned_word = []
        for ccnode in self.ccnodes:
            for span in ccnode.spans:
                l_spanned_word.append(
                    ' '.join(self.osent_words[span[0]:span[1]]))

        level_nd_count = len(self.root_cclocs)
        rooty_cclocs = copy(self.root_cclocs)
        next_level_nd_count = 0

        level_ccnodes = []
        ll_spanned_loc = []  # node,num_locs
        self.level_to_ccnodes = {}
        self.level_to_fat_ll_spanned_loc = {}

        # self.root_cclocs was filled by __init__
        level = 0
        while len(rooty_cclocs) > 0:
            if self.verbose:
                print("****************************beginning of while loop")
                print("rooty_cclocs", rooty_cclocs)
                print("level, nd_count, next_nd_count", level, "/",
                      level_nd_count,
                      next_level_nd_count)
                print("ll_spanned_loc", ll_spanned_loc)
                print("level_ccnodes", [str(x) for x in level_ccnodes])

            rooty_ccloc = rooty_cclocs.pop(0)
            rooty_ccnode  = CCTree.get_ccnode_from_ccloc(rooty_ccloc,
                                                        self.ccnodes)

            # nd=node
            level_nd_count -= 1
            level_ccnodes.append(rooty_ccnode)
            ll_spanned_loc = [ccnode.spanned_locs
                              for ccnode in level_ccnodes if ccnode]
            if self.verbose:
                print("level_ccnodes", [str(x) for x in level_ccnodes])
                print("ll_spanned_loc", ll_spanned_loc)

            for child_ccloc in \
                    self.par_ccloc_to_child_cclocs[rooty_ccloc]:
                # child becomes new root as tree is pared down
                rooty_cclocs.append(child_ccloc)
                next_level_nd_count += 1
            if self.verbose:
                print("level, nd_count, next_nd_count", level, "/",
                      level_nd_count,
                      next_level_nd_count)

            if level_nd_count == 0:
                ll_spanned_loc = \
                    CCTree.fatten_ll_spanned_loc(
                        ll_spanned_loc,
                        level_ccnodes,
                        level,
                        self.verbose)
                if level not in self.level_to_fat_ll_spanned_loc.keys():
                    self.level_to_fat_ll_spanned_loc[level] = []
                self.level_to_ccnodes[level] = level_ccnodes
                self.level_to_fat_ll_spanned_loc[level] += ll_spanned_loc
                level += 1
                level_nd_count = next_level_nd_count
                next_level_nd_count = 0
                level_ccnodes = []
                ll_spanned_loc = []
                if self.verbose:
                    print("level, nd_count, next_nd_count", level, "/",
                          level_nd_count,
                          next_level_nd_count)

        if self.verbose:
            print("level_to_ll_spanned_loc", self.level_to_fat_ll_spanned_loc)
        ccsents = []
        for level, ccnodes in self.level_to_ccnodes.items():
            fat_ll_spanned_loc = self.level_to_fat_ll_spanned_loc[level]
            for k, node in enumerate(ccnodes):
                fat_spanned_locs = sorted(fat_ll_spanned_loc[k])
                spanned_locs = sorted(node.spanned_locs)
                left_words = []
                right_words = []
                for i in fat_spanned_locs:
                    if i >= node.ccloc and i in spanned_locs:
                        pass
                    else:
                        left_words.append(self.osent_words[i])
                    if i <= node.ccloc and i in spanned_locs:
                        pass
                    else:
                        right_words.append(self.osent_words[i])
                ccsents.append(' '.join(left_words))
                ccsents.append(' '.join(right_words))
        self.ccsents = ccsents
        self.ll_spanned_loc = ll_spanned_loc
        self.l_spanned_word = l_spanned_word

        # ccsents, l_spanned_word, ll_spanned_loc
        # these 3 variables similar to:
        # word_sentences, conj_words, sentences
        # split_sentences, conj_words, sentence_indices_list


if __name__ == "__main__":
    def main1():
        in_fp = "tests/small_cctags.txt"
        out_fp = "tests/cc_ilabels.txt"
        file_translate_tags_to_ilabels("cc", in_fp, out_fp)


    def main2(forced_tree=True):
        in_fp = "tests/one_sample_cc_ilabels.txt"
        # out_fp = "tests/cc_trees.txt"
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
            tree = CCTree(osent,
                          lll_ilabel[k],
                          forced_tree,
                          verbose=True)
            tree.draw_tree()
            for i, sent in enumerate(tree.ccsents):
                print(str(i + 1) + ". " + sent)
            print()


    main1()
    main2()
