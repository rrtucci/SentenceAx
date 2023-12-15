from CCNode import *
import numpy as np
from utils_gen import get_words
from copy import deepcopy
import treelib as tr
from words_tags_ilabels_translation import *
from itertools import product
from utiils_span import *
from CCTagsLine import *
from utils_tree import *


class CCTree:
    """
    This class builds a polytree (see utils_tree.py for def of polytree)
    from `ll_ilabel`, which is either an output of a run with task "cc",
    or can be derived from a cctags file.

    Think of a CCNode by its __str__. For example, a __str__ for a CCNode
    might be (2, 5)6(7, 23). (2, 5) is its left span `span_pair[0]`,
    6 is its `ccloc` (cc location) and (7, 23) is its right span `span_pair[ 
    1]`. The CCNode's ccloc is always located between but outside the range 
    of its left and right spans.

    The tree nodes are of type CCNode or int (the int corresponding to 
    CCNode.ccloc). Each ccloc can only have one CCNode assigned to it.
    
    
    Attributes
    ----------
    ccnodes: list[CCNode]
        a list of the CCNodes of the polytree `self`.
    ccsents: list[str]
        This class derives from the polytree `self`, a list of cc sentences
        ( i.e, split sentences, obtained by splitting a compound sentence at
        a coordinating conjunction (CC)).
    child_ccloc_to_par_cclocs: dict[int, list[int]]
        dictionary mapping each child's ccloc to a list of the cclocs of the
        parents.
    forced_polytree: bool
        Sometimes ll_ilabel yields more than one parent for a given node,
        but in a polytree every node must have 0 or 1 parent. Iff this is
        True, this tells SentenceAx to discard all but one parent.
    ll_ilabel: list[list[int]]
        a matrix of ints for one sample. The outer dim ranges over depths
        and the inner one over word locations in osent_words.
    orig_sent: str
        the original sentence, before splitting (done first) and extraction (
        done second).
    osent_words: list[str]
        list of words in the original sentence.
    par_ccloc_to_child_cclocs: list[int,list[int]]
        dictionary mapping each parent's ccloc to a list of the cclocs of the
        children.
    root_cclocs: list[int]
        list of the cclocs of the roots of the polytree `self`.
    verbose: str
        True iff want verbose output to be printed out in console.
    
    """

    def __init__(self, orig_sent, ll_ilabel, forced_polytree=True,
                 calc_tree_struc=True, verbose=False):
        """
        Constructor

        Parameters
        ----------
        orig_sent: str
        ll_ilabel: list[list[int]]
        forced_polytree: bool
        calc_tree_struc: bool
            setting this to False makes this class basically a structureless
            lightweight list of ccnodes. Things like the tree dictionary and
            the ccsents are not calculated.
        verbose: bool
        """
        self.orig_sent = orig_sent
        self.osent_words = get_words(orig_sent)
        if verbose:
            print("New cctree:")
            print(orig_sent)
        self.ll_ilabel = ll_ilabel
        self.forced_polytree = forced_polytree
        self.verbose = verbose

        # self.osent_locs = range(len(self.osent_words))

        self.ccnodes = None
        self.l_cctags_line = None
        # This must be called before calling self.set_tree_structure()
        self.set_ccnodes()

        self.root_cclocs = None
        self.par_ccloc_to_child_cclocs = None
        self.child_ccloc_to_par_cclocs = None
        # this fill the 3 previous None's
        if calc_tree_struc:
            self.set_tree_structure()

        self.ccsents = None
        # self.level0_to_ccnodes = None
        # self.level0_to_fat_ll_spanned_loc = None
        #
        # self.ll_spanned_loc = None
        # self.l_spanned_word = None

        # this fills the previous unfilled None's
        if calc_tree_struc:
            # self.set_spanned_attributes()
            self.set_ccsents()

    @staticmethod
    def get_ccnode_from_ccloc(ccloc, ccnodes):
        """
        This static method returns the unique CCNode out of `ccnodes` that
        owns the ccloc `ccloc`.

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

    def get_ccnode(self, ccloc):
        """
        This non-static method returns the unique CCNode out of
        `self.ccnodes` that owns the ccloc `ccloc`.

        Parameters
        ----------
        ccloc: int

        Returns
        -------
        CCNode

        """

        return CCTree.get_ccnode_from_ccloc(ccloc, self.ccnodes)

    def remove_bad_ccnodes(self):
        """
        similar to Openie6.data.coords_to_sentences

        This method removes unacceptable ccnodes from the list self.ccnodes

        Returns
        -------
        None

        """
        if self.verbose:
            print("nodes before removals: ", [str(ccnode) for ccnode in
                                              self.ccnodes])
        # enforce one to one mapping between ccnodes and cclocs
        ccloc_to_l_ccnode = {}
        for ccnode in self.ccnodes:
            if ccnode.ccloc not in ccloc_to_l_ccnode.keys():
                ccloc_to_l_ccnode[ccnode.ccloc] = []
            ccloc_to_l_ccnode[ccnode.ccloc].append(ccnode)
        for ccloc in ccloc_to_l_ccnode.keys():
            if len(ccloc_to_l_ccnode[ccloc]) > 1:
                for k, ccnode in enumerate(ccloc_to_l_ccnode[ccloc]):
                    if k >= 1:
                        if self.verbose:
                            print(f"node {ccnode} removed because there is "
                                  "more than one ccnode with this ccloc")
                        self.ccnodes.remove(ccnode)

        for ccnode in self.ccnodes:
            ccloc = ccnode.ccloc
            num_osent_words = len(self.osent_words)
            if ccloc >= num_osent_words:
                if self.verbose:
                    print(f"node {ccnode} removed because "
                          f"ccloc={ccloc} is >= to len(osent)={num_osent_words}")
                self.ccnodes.remove(ccnode)
                continue
            ccword = self.osent_words[ccloc]
            loc_to_word = ccnode.get_spanned_unbreakable_loc_to_word()
            if ccword in ['nor', '&']:
                if self.verbose:
                    print(f"node {ccnode} removed because "
                          f"{ccword} is not allowed as a CC.")
                self.ccnodes.remove(ccnode)
            elif len(loc_to_word):
                if self.verbose:
                    print(f"node {ccnode} removed because "
                          "its span contains unbreakable words.")
                    print("unbreakable_loc_to_word=", loc_to_word)
                self.ccnodes.remove(ccnode)

    def set_ccnodes(self):
        """
        similar to Openie6.metric.get_coords()

        This method creates the list of ccnodes `self.ccnodes`.

        The method also stores in `l_cctags_line` the info it gleans from
        each line of a sample.

        The method also removes bad nodes by calling remove_bad_ccnodes(),
        and it performs a sanity check of the newly created ccnodes by
        calling ccnode.check_self() for each ccnode.

        Returns
        -------
        None

        """
        self.ccnodes = []
        self.l_cctags_line = []

        num_depths = len(self.ll_ilabel)
        for depth in range(num_depths):
            l_ilabel = self.ll_ilabel[depth]
            cctags_line = CCTagsLine(depth, l_ilabel)
            self.l_cctags_line.append(cctags_line)

        for depth in range(num_depths):
            cctags_line = self.l_cctags_line[depth]
            for ccloc in cctags_line.cclocs:
                spans = cctags_line.spans
                span_pair = CCTagsLine.get_span_pair(spans,
                                                     ccloc,
                                                     throw_if_None=False)
                if span_pair:
                    ccnode = CCNode(ccloc,
                                    depth,
                                    self.osent_words,
                                    span_pair)
                    self.ccnodes.append(ccnode)
        self.remove_bad_ccnodes()
        # print("llm", [str(ccnode) for ccnode in self.ccnodes])
        for ccnode in self.ccnodes:
            ccnode.check_self()

    def set_tree_structure(self):
        """
        similar to Openie6.data.get_tree(conj) where conj=coords=ccnodes.
        Openie6 uses conj=ccloc almost everywhere else!

        This method creates the dictionaries self.child_ccloc_to_par_cclocs
        and self.par_ccloc_to_child_cclocs that define the polytree.

        This method also finds the root nodes of the polytree and stores
        them in the list self.root_nodes.


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

        if self.forced_polytree:
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

    # @staticmethod
    # def get_essential_locs(num_osent_words, ccnodes):
    #     """
    #     This method returns the list of all locations in osent that are not
    #     spanned, or a cc, or a sep_loc, for ANY ccnode.
    #
    #     Parameters
    #     ----------
    #     num_osent_words: int
    #     ccnodes: list[CCNodes]
    #
    #     Returns
    #     -------
    #     list[int]
    #
    #     """
    #     essential_locs = list(range(num_osent_words))
    #     for ccnode in ccnodes:
    #         spans = ccnode.spans
    #         # print("essential_locs, ccloc", essential_locs, ccnode.ccloc)
    #         try:
    #             essential_locs.remove(ccnode.ccloc)
    #         except:
    #             pass
    #         for i in range(*spans[0]):
    #             try:
    #                 essential_locs.remove(i)
    #             except:
    #                 pass
    #         for i in range(*spans[1]):
    #             try:
    #                 essential_locs.remove(i)
    #             except:
    #                 pass
    #         for i in ccnode.sep_locs:
    #             try:
    #                 essential_locs.remove(i)
    #             except:
    #                 pass
    #     return essential_locs

    # @staticmethod
    # def get_ccsents(osent, ccnodes):
    #     """
    #     This method returns a list of the ccsents (conjugate coordination
    #     sentences).
    #
    #     Parameters
    #     ----------
    #     osent: str
    #         original sentence
    #     ccnodes: list[CCNode]
    #
    #     Returns
    #     -------
    #     list[str]
    #
    #     """
    #     osent_words = get_words(osent)
    #     essential_locs = CCTree.get_essential_locs(len(osent_words), ccnodes)
    #     cclocs = [ccnode.ccloc for ccnode in ccnodes]
    #     osent_words = get_words(osent)
    #     ccsents = []
    #     print_list("essential_locs", essential_locs)
    #     for bool_vec in product([0, 1], repeat=len(cclocs)):
    #         ccsent_locs = copy(essential_locs)
    #         for k in range(len(bool_vec)):
    #             ccloc = cclocs[k]
    #             spans = CCTree.get_ccnode_from_ccloc(ccloc, ccnodes).spans
    #             ccsent_locs += list(range(*spans[bool_vec[k]]))
    #         print("llojk", ccsent_locs)
    #         ccsent_locs.sort()
    #         ccsent = " ".join([osent_words[loc] for loc in ccsent_locs])
    #         ccsents.append(ccsent)
    #     return ccsents

    def get_all_ccnode_paths(self):
        """
        This method calls the global function
        utils_tree.get_all_paths_from_any_root() for the polytree `self`.

        This method considers subtrees of the polytree `self` that have as a
        root node one of the root nodes of the polytree, and vary in depth
        from a zero to the depth of the polytree `self`

        The method returns a list of all paths of CCNodes for all such
        subtrees. Each path starts at the root_node of its subtree and ends
        at a (nonempty) leaf node of its subtree.

        Returns
        -------
        list[list[CCNode]]

        """
        if self.verbose:
            print("child_ccloc_to_par_cclocs",
                  self.child_ccloc_to_par_cclocs)
            print("par_ccloc_to_child_cclocs",
                  self.par_ccloc_to_child_cclocs)
        l_ccloc_path = get_all_paths_from_any_root(
            self.par_ccloc_to_child_cclocs,
            self.root_cclocs,
            self.verbose)
        l_ccnode_path = []
        for ccloc_path in l_ccloc_path:
            ccnode_path = [self.get_ccnode(ccloc) for ccloc in ccloc_path]
            l_ccnode_path.append(ccnode_path)
        return l_ccnode_path

    @staticmethod
    def get_inc_exc_span_path(ccnode_path,
                              l_bit,
                              all_span,
                              verbose=False):
        """
        This method starts with a list of N ccnodes `ccnode_path`

        N= len(ccnode_path) = len(l_bit)

        The method then uses that ccnode path to generate the list of 2^N
        span paths obtained by choosing either the left span or the right
        span of each ccnode of that ccnode path. Those 2^N span paths are
        labelled by N-bit vectors (0 if left span is chosen or 1 if right
        span is chosen).

        The method then chooses, from those 2^N span paths, the single one
        `inc_span_path` that is labelled by the input bit vector `l_bit`. At
        the same time, the method stores another span path `exc_span_path`.

        `exc_span_path` is an anti-twin, contrarian twin, to `inc_span_path`
        which chooses the right span every time `inc_span_path` chooses the
        left span (and vice versa).

        inc=included, exc=excluded

        The method then asks if `inc_span_path` satisfies the
        span_path_is_decreasing() condition. If it does, the method returns
        the pair of span paths `inc_span_path, exc_span_path`. If it doesn't
        satisfy that condition, the method returns `None, None`.

        Parameters
        ----------
        ccnode_path: list[CCNode]
            a path (i.e., list) of CCNodes
        l_bit: list[int]
            a list of ints [0, 1]. N=len(l_bit)=len(ccnode_path)
        all_span: tuple[int, int]
            The span (0, number of words in osent)
        verbose: bool

        Returns
        -------
        list[tuple[int, int]], list[tuple[int, int]]
            inc_span_path, exc_span_path

        """

        num_depths = len(l_bit)
        inc_span_path = []
        exc_span_path = []
        for depth in range(num_depths):
            ccnode = ccnode_path[depth]
            bit = l_bit[depth]
            inc_span_path.append(ccnode.span_pair[bit])
            exc_span_path.append(ccnode.span_pair[flip(bit)])
        if span_path_is_decreasing(inc_span_path):
            if verbose:
                print("included_span_path", inc_span_path)
                print("excluded_span_path", exc_span_path)
                draw_inc_exc_span_paths(all_span,
                                        inc_span_path,
                                        exc_span_path)
            return inc_span_path, exc_span_path
        else:
            return None, None

    #
    # @staticmethod
    # def get_donut(span,
    #               sub_spans,
    #               kept_sub_span):
    #     """
    #
    #     Parameters
    #     ----------
    #     span: tuple[int]
    #     sub_spans: list[list[int]]
    #     kept_sub_span: int
    #         either 0 or 1
    #
    #     Returns
    #     -------
    #     list[int] | None
    #
    #     """
    #     # print("subspan0, span", sub_spans[0], span)
    #     # print("subspan1, span", sub_spans[1], span)
    #     if not is_sub_span(sub_spans[0], span) or\
    #         not is_sub_span(sub_spans[1], span):
    #         return None
    #
    #     span_set = set(range(*span))
    #     subset0 = set(range(*sub_spans[0]))
    #     subset1 = set(range(*sub_spans[1]))
    #
    #     diff_set = (span_set - subset0) - subset1
    #     if kept_sub_span == 0:
    #         return list(diff_set | subset0)
    #     elif kept_sub_span == 1:
    #         return list(diff_set | subset1)
    #     else:
    #         return False

    # @staticmethod
    # def get_donut_path(ccnode_path,
    #                    l_bit,
    #                    len_osent_words,
    #                    verbose=False):
    #     """
    #
    #     Parameters
    #     ----------
    #     ccnode_path: list[CCNode]
    #     l_bit: list[int]
    #         list of 0's or 1's
    #     len_osent_words: int
    #
    #     Returns
    #     -------
    #     list[list[int]]
    #
    #     """
    #     span_path = CCTree.get_span_path(ccnode_path, l_bit, verbose)
    #     num_depths = len(l_bit)
    #     for depth in range(num_depths):
    #         if depth < num_depths - 1:
    #             donut = CCTree.get_donut(span_all,
    #                                      ccnode_path[0].spans,
    #                                      kept_sub_span=l_bit[depth])
    #         else:
    #             donut = CCTree.get_donut(span_path[depth],
    #                                      ccnode_path[depth + 1].spans,
    #                                      kept_sub_span=l_bit[depth])
    #         # elif depth == num_depths-1:
    #         #     donut = CCTree.get_donut(span_path[depth],
    #         #                              [[0,0], [1,1]],
    #         #                              kept_sub_span=l_bit[depth])
    #         if not donut:
    #             return None
    #         else:
    #             donut_path.append(donut)
    #     if verbose:
    #         print("donut path: ", donut_path)
    #     return donut_path

    # @staticmethod
    # def get_donut_path(ccnode_path,
    #                    l_bit,
    #                    len_osent_words,
    #                    verbose=False):
    #     """
    #
    #     Parameters
    #     ----------
    #     ccnode_path: list[CCNode]
    #     l_bit: list[int]
    #         list of 0's or 1's
    #     len_osent_words: int
    #
    #     Returns
    #     -------
    #     list[list[int]]
    #
    #     """
    #     span_path = CCTree.get_span_path(ccnode_path, l_bit, verbose)
    #     # add first dummy item to ccnode_path. Won't be used
    #     if not span_path:
    #         return None
    #     donut_path = []
    #     span_all = (0, len_osent_words)
    #     num_depths = len(l_bit)
    #     for depth in range(-1, num_depths-1):
    #         if depth == -1:
    #             donut = CCTree.get_donut(span_all,
    #                                      ccnode_path[0].spans,
    #                                      kept_sub_span=l_bit[depth])
    #         else:
    #             donut = CCTree.get_donut(span_path[depth],
    #                                      ccnode_path[depth+1].spans,
    #                                      kept_sub_span=l_bit[depth])
    #         # elif depth == num_depths-1:
    #         #     donut = CCTree.get_donut(span_path[depth],
    #         #                              [[0,0], [1,1]],
    #         #                              kept_sub_span=l_bit[depth])
    #         if not donut:
    #             return None
    #         else:
    #             donut_path.append(donut)
    #     if verbose:
    #         print("donut path: ", donut_path)
    #     return donut_path

    def get_ccsent(self, exc_span_path):
        """
        This method returns the ccsent (cc sentence) corresponding to the
        span path `exc_span_path`. It calculates that ccsent by removing
        from all_span = (0, length of osent_words), all locs included in
        `exc_span_path` (that is why we call it an excluded span path,
        because it dictates what locs to remove/exclude) from all_span. The
        method also removes from all_span, the cclocs (conjunction
        locations), sep_locs (separator locations) and other_locs.

        Parameters
        ----------
        exc_span_path: list[tuple[int, int]]

        Returns
        -------
        str

        """
        assert exc_span_path
        all_span = (0, len(self.osent_words))
        fin_set = set(range(*all_span))
        for exc_span in exc_span_path:
            fin_set -= set(range(*exc_span))
        fin_locs = sorted(list(fin_set))
        # remove also cclocs, sep_locs and other_locs
        new_fin_locs = copy(fin_locs)
        for i in fin_locs:
            for cctags_line in self.l_cctags_line:
                if i in cctags_line.cclocs or \
                        i in cctags_line.sep_locs or \
                        i in cctags_line.other_locs:
                    if i in new_fin_locs:
                        new_fin_locs.remove(i)
        ccsent = " ".join([self.osent_words[loc] for loc in new_fin_locs])
        return ccsent

    def set_ccsents(self):
        """
        similar to Openie6.data.get_sentences()

        This method sets self.ccsents. It does this by calling
        get_inc_exc_span_path() to get an `exc_span_path`, and then calling
        `get_ccsent(exc_span_path)` to get a ccsent.

        Returns
        -------
        None

        """
        ccnode_paths = self.get_all_ccnode_paths()
        all_span = (0, len(self.osent_words))
        l_ccsent = []
        for path in ccnode_paths:
            if self.verbose:
                print("node path: ", [str(ccnode) for ccnode in path])
        for ccnode_path in ccnode_paths:
            path_len = len(ccnode_path)
            for l_bit in product([0, 1], repeat=path_len):
                l_bit = list(l_bit)
                _, exc_span_path = CCTree.get_inc_exc_span_path(
                    ccnode_path,
                    l_bit,
                    all_span,
                    self.verbose)
                if exc_span_path:
                    ccsent = self.get_ccsent(exc_span_path)
                    l_ccsent.append(ccsent)

        self.ccsents = l_ccsent

        # this mimics Openie6.get_sentences()
        # ccsents = []
        # for level0, ccnodes in self.level0_to_ccnodes.items():
        #     fat_ll_spanned_loc = self.level0_to_fat_ll_spanned_loc[level0]
        #     for k, node in enumerate(ccnodes):
        #         fat_spanned_locs = sorted(fat_ll_spanned_loc[k])
        #         spanned_locs = sorted(node.spanned_locs)
        #         left_words = []
        #         right_words = []
        #         for i in fat_spanned_locs:
        #             if i >= node.ccloc and i in spanned_locs:
        #                 pass
        #             else:
        #                 left_words.append(self.osent_words[i])
        #             if i <= node.ccloc and i in spanned_locs:
        #                 pass
        #             else:
        #                 right_words.append(self.osent_words[i])
        #         ccsents.append(' '.join(left_words))
        #         ccsents.append(' '.join(right_words))
        # self.ccsents = ccsents

    # not used anymore
    # @staticmethod
    # def fatten_ll_spanned_loc(ll_spanned_loc,
    #                           level0_ccnodes,
    #                           level0,
    #                           verbose):
    #     """
    #     similar to Openie6.data.get_sentences(sentences,
    #               conj_same_level0,
    #               conj_coords,
    #               sentence_indices)
    #     doesn't return anything but changes sentences
    #
    #     conj = ccloc, conjunct = spans, coord = ccnode
    #     sentences = ll_spanned_loc
    #     sentence = level0_spanned_locs
    #     conj_same_level0 = level0_cclocs
    #     conj_coords = swaps
    #     sentence_indices = osent_locs
    #
    #     level0 = same/equal level0
    #     li = list
    #
    #
    #     Parameters
    #     ----------
    #     ll_spanned_loc: list[list[int]]
    #     level0_ccnodes: list[CCNode]
    #     level0: int
    #
    #     Returns
    #     -------
    #     list[list[int]]
    #
    #     """
    #     # print("level0=", level0)
    #     # print("ll_spanned_loc", ll_spanned_loc)
    #     k = 0
    #     # print("num_nodes", len(self.ccnodes))
    #     for ccnode in level0_ccnodes:
    #         # print("nnml", "node_id", k)
    #         k += 1
    #         fat_spanned_locs = ccnode.get_spanned_locs(fat=True)
    #         if not ll_spanned_loc:
    #             ll_spanned_loc.append(fat_spanned_locs)
    #         else:
    #             # to_be_added_ll_loc = []
    #             # to_be_removed_ll_loc = []
    #             for spanned_locs in ll_spanned_loc:
    #                 # print("bhk", spanned_locs)
    #                 # only ccnodes that satisfy this have fat-spanned_locs
    #                 if ccnode.spans[0][0] in spanned_locs:
    #                     # to_be_added_ll_loc.append(fat_spanned_locs)
    #                     # to_be_removed_ll_loc.append(spanned_locs)
    #                     if spanned_locs in ll_spanned_loc:
    #                         ll_spanned_loc.remove(spanned_locs)
    #                     ll_spanned_loc.append(fat_spanned_locs)
    #
    #             # for l_loc in to_be_removed_ll_loc:
    #             #     ll_spanned_loc.remove(l_loc)
    #             # for l_loc in to_be_added_ll_loc:
    #             #     ll_spanned_loc.append(l_loc)
    #     if verbose:
    #         print("new_ll_spanned_loc", ll_spanned_loc)
    #     return ll_spanned_loc

    # def set_spanned_attributes(self):
    #     """
    #     similar to Openie6.data.coords_to_sentences()
    #     ccsents ~ Openie6.split_sentences
    #
    #     Returns
    #     -------
    #     None
    #
    #     """
    #     # self.remove_bad_ccnodes()  was called at the end of get_ccnodes()
    #
    #     l_spanned_word = []
    #     for ccnode in self.ccnodes:
    #         for span in ccnode.spans:
    #             l_spanned_word.append(
    #                 ' '.join(self.osent_words[span[0]:span[1]]))
    #
    #     level0_nd_count = len(self.root_cclocs)
    #     rooty_cclocs = copy(self.root_cclocs)
    #     next_level0_nd_count = 0
    #
    #     level0_ccnodes = []
    #     ll_spanned_loc = []  # node,num_locs
    #     self.level0_to_ccnodes = {}
    #     self.level0_to_fat_ll_spanned_loc = {}
    #
    #     # self.root_cclocs was filled by __init__
    #     level0 = 0
    #     while len(rooty_cclocs) > 0:
    #         if self.verbose:
    #             print("****************************beginning of while loop")
    #             print("rooty_cclocs", rooty_cclocs)
    #             print("level0, nd_count, next_nd_count", level0, "/",
    #                   level0_nd_count,
    #                   next_level0_nd_count)
    #             print("ll_spanned_loc", ll_spanned_loc)
    #             print("level0_ccnodes", [str(x) for x in level0_ccnodes])
    #
    #         rooty_ccloc = rooty_cclocs.pop(0)
    #         rooty_ccnode = self.get_ccnode(rooty_ccloc)
    #
    #         # nd=node
    #         level0_nd_count -= 1
    #         level0_ccnodes.append(rooty_ccnode)
    #         ll_spanned_loc = [ccnode.spanned_locs
    #                           for ccnode in level0_ccnodes if ccnode]
    #         if self.verbose:
    #             print("level0_ccnodes", [str(x) for x in level0_ccnodes])
    #             print("ll_spanned_loc", ll_spanned_loc)
    #
    #         for child_ccloc in \
    #                 self.par_ccloc_to_child_cclocs[rooty_ccloc]:
    #             # child becomes new root as tree is pared down
    #             rooty_cclocs.append(child_ccloc)
    #             next_level0_nd_count += 1
    #         if self.verbose:
    #             print("level0, nd_count, next_nd_count", level0, "/",
    #                   level0_nd_count,
    #                   next_level0_nd_count)
    #
    #         if level0_nd_count == 0:
    #             ll_spanned_loc = \
    #                 CCTree.fatten_ll_spanned_loc(
    #                     ll_spanned_loc,
    #                     level0_ccnodes,
    #                     level0,
    #                     self.verbose)
    #             if level0 not in self.level0_to_fat_ll_spanned_loc.keys():
    #                 self.level0_to_fat_ll_spanned_loc[level0] = []
    #             self.level0_to_ccnodes[level0] = level0_ccnodes
    #             self.level0_to_fat_ll_spanned_loc[level0] += ll_spanned_loc
    #             level0 += 1
    #             level0_nd_count = next_level0_nd_count
    #             next_level0_nd_count = 0
    #             level0_ccnodes = []
    #             ll_spanned_loc = []
    #             if self.verbose:
    #                 print("level0, nd_count, next_nd_count", level0, "/",
    #                       level0_nd_count,
    #                       next_level0_nd_count)
    #
    #     if self.verbose:
    #         print("level0_to_ll_spanned_loc", self.level0_to_fat_ll_spanned_loc)
    #
    #     # setting value of self.ccsents done here in Openie6
    #
    #     # self.ll_spanned_loc = ll_spanned_loc
    #     # self.l_spanned_word = l_spanned_word
    #
    #     # ccsents, l_spanned_word, ll_spanned_loc
    #     # these 3 variables similar to:
    #     # word_sentences, conj_words, sentences
    #     # split_sentences, conj_words, sentence_indices_list

    def draw_self(self):
        """
        This method draws self by calling the global function
        utils_tree.draw_polytree().

        Returns
        -------
        None

        """

        def fun(x):
            return str(self.get_ccnode(x))

        polytree = get_mapped_polytree(self.par_ccloc_to_child_cclocs, fun)
        draw_polytree(polytree)


if __name__ == "__main__":
    def main1():
        in_fp = "tests/small_cctags.txt"
        out_fp = "tests/cc_ilabels.txt"
        file_translate_tags_to_ilabels("cc", in_fp, out_fp)


    def main2(forced_polytree=True):
        in_fp = "tests/one_sample_cc_ilabels.txt"
        # out_fp = "tests/cc_trees.txt"
        with open(in_fp, "r", encoding="utf-8") as f:
            in_lines = get_ascii(f.readlines())

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
            tree = CCTree(osent,
                          lll_ilabel[k],
                          forced_polytree,
                          verbose=True)
            tree.draw_self()
            for i, sent in enumerate(tree.ccsents):
                print(str(i + 1) + ". " + sent)
            print()


    main1()
    main2()
