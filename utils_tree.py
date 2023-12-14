"""

This file gives some global functions related to trees. These methods are
used in the class CCTree.

In this file, `tree` stands for a dictionary parent_to_children mapping each
parent node to a list children nodes.

For most of the methods in this file, the nodes can be of any type (Node,
str, etc.). If the nodes need to be of type str, one can use `get_fun_polytree(
)` to map the tree to a "stringified" tree (one whose nodes are all
specified by strings).

Technically a tree has a single root node. If the dictionary
parent_to_children yields more than one root node, we call it a polytree.
The method get_all_trees_of_polytree() can be used to extract all trees  of
the polytree.

We use two types of trees: with and without empty leafs. An empty leaf is an
element of the tree=parent_to_children dictionary of the form "A"->[]. Empty
leafs are necessary for representing the single node tree, so we usually
work with trees with empty leafs when doing tree calculations.


"""
from copy import copy
import treelib as tr


def get_all_nodes(polytree):
    """
    This works whether polytree has empty leafs or not. If it does,
    empty leaf nodes are not included in output list.

    Parameters
    ----------
    polytree

    Returns
    -------

    """
    all_children = get_all_children(polytree)
    all_par = list(polytree.keys())
    return list(set(all_par) | set(all_children))


def get_all_children(polytree):
    """
    This works whether polytree has empty leafs or not. If it does,
    empty children are not included in output list.


    Parameters
    ----------
    polytree

    Returns
    -------

    """
    all_children = []
    for parent, children in polytree.items():
        for child in children:
            if child and child not in all_children:
                all_children.append(child)
    return all_children


def count_leaf_nodes(polytree):
    empty_leaf_count = 0
    nonempty_leaf_count = 0
    for node in polytree:
        if node in polytree and polytree[node]:
            nonempty_leaf_count += 1
        else:
            empty_leaf_count += 1
    return empty_leaf_count, nonempty_leaf_count


def all_leafs_are_nonempty(polytree):
    empty_leaf_count, _ = count_leaf_nodes(polytree)

    if empty_leaf_count == 0:
        return True
    else:
        return False


def all_leafs_are_empty(polytree):
    _, nonempty_leaf_count = count_leaf_nodes(polytree)

    if nonempty_leaf_count == 0:
        return True
    else:
        return False


def get_fun_polytree(polytree, fun):
    """
    This works whether polytree has empty leafs or not. If thev polytree has 
    empty leafs, it maps maps parent->[] to fun(parent)->[]
    
    Parameters
    ----------
    polytree
    fun

    Returns
    -------

    """
    fun_polytree = {}
    for par, children in polytree.items():
        # if children =[], this maps par->[] to fun(par)->[]
        fun_polytree[fun(par)] = [fun(child) for child in children]
    return fun_polytree


def get_root_nodes(polytree):
    """
    This works whether polytree has empty leafs or not.
    
    Parameters
    ----------
    polytree

    Returns
    -------

    """
    all_children = get_all_children(polytree)
    return [node for node in polytree if node not in all_children]


def get_tree_depth(tree, root_node):
    """
    If tree has empty leafs, depth is 1 more than if it doesn't due to last 
    layer of empty tree leafs.

    num_depths = depth + 1
    
    Parameters
    ----------
    tree
    root_node

    Returns
    -------

    """
    if root_node not in tree:
        # If the root is not in the dictionary,
        # it's a leaf node with depth 0
        return 0

    children = tree[root_node]
    if not children:
        # If the root has no children, its depth is 1
        return 1

        # Recursively calculate the depth for each child and find the maximum
    child_depths = [get_tree_depth(tree, child) for child in children]
    return 1 + max(child_depths)


def get_one_tree_of_polytree(polytree,
                             root_node,
                             output_empty_leafs=True):
    """
    This works whether polytree has empty leafs or not.
    
    Parameters
    ----------
    polytree
    root_node
    output_empty_leafs

    Returns
    -------

    """
    tree0 = {root_node: []}
    polytree0 = add_empty_leafs(polytree)
    l_prev_leaf_node = [root_node]
    while True:
        l_leaf_node = []
        for leaf_node in l_prev_leaf_node:
            parents = polytree0[leaf_node]
            tree0[leaf_node] = parents
            l_leaf_node += parents
            if l_leaf_node:
                l_prev_leaf_node = copy(l_leaf_node)
            else:
                if output_empty_leafs:
                    return add_empty_leafs(tree0)
                else:
                    return remove_empty_leafs(tree0)


def get_all_trees_of_polytree(polytree, output_empty_leafs=True):
    """
    This works whether polytree has empty leafs or not.
    
    
    Parameters
    ----------
    polytree
    output_empty_leafs

    Returns
    -------

    """
    root_nodes = get_root_nodes(polytree)
    l_tree = []
    for root_node in root_nodes:
        tree = get_one_tree_of_polytree(
            polytree,
            root_node,
            output_empty_leafs=output_empty_leafs)
        l_tree.append(tree)
    return l_tree


def draw_tree(tree, root_node):
    """
    important bug that must be fixed in treelib. In your Python
    installation, go to Lib\site-packages\treelib and edit tree.py. Find
    def show. The last line is:

    print(self.reader.encode('utf-8'))

    It should be:

    print(self.reader)

    This method draws the same thing ( no empty leafs) whether `tree` has 
    empty leafs or not, but if it does, the method prints out a message 
    warning that "empty leafs present but not drawn".

    Returns
    -------
    None

    """
    try:
        pine_tree = tr.Tree()
        pine_tree.create_node(root_node,
                              root_node)
        for parent, children in tree.items():
            for child in children:
                # print(f"{parent}->{child}")
                if child != root_node:
                    pine_tree.create_node(child,
                                          child,
                                          parent=parent)
        pine_tree.show()
        if not has_zero_empty_leaf_nodes(tree):
            print("WARNING: Empty leafs present but not drawn.")
    except:
        print("*********************tree not possible")
        print(tree)


def draw_polytree(polytree):
    """
    
    This method draws the same thing ( no empty leafs) whether `polytree` 
    has empty leafs or not, but if it does, the method prints out a message 
    warning that "empty leafs present but not drawn".
    
    Parameters
    ----------
    polytree

    Returns
    -------

    """
    root_nodes = get_root_nodes(polytree)
    for root_node in root_nodes:
        tree = get_one_tree_of_polytree(polytree, root_node)
        draw_tree(tree, root_node)


def remove_empty_leafs(x):
    def _remove_empty_leafs(polytree):
        if all_leafs_are_nonempty(polytree):
            return copy(polytree)
        new_polytree = {}
        for par in polytree.keys():
            if polytree[par]:
                new_polytree[par] = polytree[par]
            else:
                pass
        return new_polytree

    if type(x) == dict:
        return _remove_empty_leafs(x)
    elif type(x) == list:
        return [_remove_empty_leafs(a) for a in x]
    else:
        assert False


def add_empty_leafs(x):
    def _add_empty_leafs(polytree):
        if all_leafs_are_empty(polytree):
            return copy(polytree)
        all_parents = list(polytree.keys())
        all_children = get_all_children(polytree)
        all_nodes = set(all_parents + all_children)
        new_polytree = copy(polytree)
        for node in all_nodes:
            if node not in polytree.keys():
                new_polytree[node] = []
        return new_polytree

    if type(x) == dict:
        return _add_empty_leafs(x)
    elif type(x) == list:
        return [_add_empty_leafs(a) for a in x]
    else:
        assert False


def get_different_depth_subtrees(full_tree,
                                 root_node,
                                 output_empty_leafs=True,
                                 verbose=False):
    """
    all subtrees with same root node as full tree
    but different depths.
    
    This works whether `full_tree` has empty leafs or not.
    
    
    Parameters
    ----------
    full_tree
    root_node
    output_empty_leafs
    verbose

    Returns
    -------

    """
    full_tree0 = add_empty_leafs(full_tree)
    num_depths = get_tree_depth(full_tree0, root_node)
    depth = 0
    init_tree = {root_node: []}
    l_subtree = [init_tree]
    l_prev_leaf_node = [root_node]
    tree = copy(init_tree)
    if verbose:
        print("\nEntering get_different_depth_subtrees()")
        print(f"full tree:\n {full_tree}")
    while depth < num_depths:
        if verbose:
            print(f"depth={depth}, tree=", tree)
        l_leaf_node = []
        depth += 1
        for leaf_node in l_prev_leaf_node:
            parents = full_tree0[leaf_node]
            tree[leaf_node] = parents
            l_leaf_node += parents
        l_subtree.append(tree)
        if depth > num_depths - 1:
            return l_subtree
        if l_leaf_node:
            l_prev_leaf_node = copy(l_leaf_node)
        else:
            if output_empty_leafs:
                return add_empty_leafs(l_subtree)
            else:
                return remove_empty_leafs(l_subtree)



def get_all_paths_from_root(polytree,
                            root_nodes,
                            verbose=False):
    """
    This works whether `polytree` has empty leafs or not. 
    Outputed paths do not have a [] at the end.

    Parameters
    ----------
    root_nodes: list[Node] | list[int]
    polytree: dict[Node, list[Node]] | dict[int, list[int]]
    verbose: bool

    Returns
    -------
    list[list[Node]] | list[list[int]]

    """
    l_path_for1root = []
    polytree0 = add_empty_leafs(polytree)

    if verbose:
        print("\nEntering get_all_paths_from_root()")

    # init input:
    # cur_root_node = root_node
    # cur_path =[]
    # init_output
    # l_path_for1root = []
    def get_paths_for_single_root_node(polytree0,
                                       cur_root_node,
                                       cur_path):
        cur_path = cur_path + [cur_root_node]
        if not polytree0[cur_root_node]:
            l_path_for1root.append(cur_path)
        else:
            for child in polytree0[cur_root_node]:
                if child:
                    get_paths_for_single_root_node(
                        polytree0=polytree0,
                        cur_root_node=child,
                        cur_path=cur_path)

        return l_path_for1root

    l_path = []
    for root_node in root_nodes:
        subtrees = get_different_depth_subtrees(polytree0,
                                                root_node,
                                                output_empty_leafs=True,
                                                verbose=verbose)
        for subtree in subtrees:
            l_path_for1root = []
            l_path_for1root = \
                get_paths_for_single_root_node(
                    polytree0=subtree,
                    cur_root_node=root_node,
                    cur_path=[])
            if verbose:
                print(f"paths starting at root node = {root_node}:")
                print(l_path_for1root)
            l_path += l_path_for1root
    return l_path


if __name__ == "__main__":

    def main1():
        # Example tree structure

        #        E
        #       /
        # A->B->C->F
        #     \
        #      D
        # A1->B1
        # 4 paths

        # leafs must be in!
        polytree = {
            'A': ['B'],
            'B': ['C', 'D'],
            'C': ['E', "F"],
            'A1': ['B1'],
            "B1": []
        }
        root_nodes = get_root_nodes(polytree)
        print("root nodes=", root_nodes)
        l_path = get_all_paths_from_root(polytree,
                                         root_nodes,
                                         verbose=True)
        print("l_path:\n", l_path)


    def main2():
        full_tree = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F']}
        root_node = "A"

        print()
        print("full_tree:\n", full_tree)
        new_full_tree = add_empty_leafs(full_tree)
        print("added empty leafs:\n", new_full_tree)
        new_full_tree = remove_empty_leafs(new_full_tree)
        print("added then removed empty leafs:\n", new_full_tree)

        l_subtree = get_different_depth_subtrees(new_full_tree,
                                                 root_node,
                                                 output_empty_leafs=False,
                                                 verbose=True)

        for i, subtree in enumerate(l_subtree):
            print(f"Subtree {i + 1}: ", subtree)


    def main3():
        tree = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F'],
            'D': ['G'],
            'F': ['H', 'I']
        }

        root_node = 'A'
        print("tree without empty nodes:")
        draw_tree(tree, root_node)
        tree_depth = get_tree_depth(tree, root_node)
        print(f"The depth of the tree is: {tree_depth}")
        tree0 = add_empty_leafs(tree)
        print()
        print("tree with empty nodes:")
        draw_tree(tree0, root_node)
        tree0_depth = get_tree_depth(tree0, root_node)
        print(f"The depth of the tree with empty nodes is: {tree0_depth}")


    def main4():
        polytree = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F'],
            'D': ['G'],
            'F': ['H', 'I'],
            'A1': ['B1'],
            "B1": []
        }
        print("draw polytree:")
        draw_polytree(polytree)


    main1()
    main2()
    main3()
    main4()
