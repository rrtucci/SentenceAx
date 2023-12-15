"""

This file contains some global functions related to trees. These methods are 
used in the class CCTree.

In this file, `tree` stands for a dictionary parent_to_children mapping each
parent node to a list of children nodes. 

For most of the methods in this file, the nodes can be of any type (Node, 
str, A, B, etc.), although strings are preferred. If the nodes need to be of 
type B, one can use `get_mapped_polytree( )` to map an A tree (one whose 
nodes are all specified by type A) to a B tree. The opposite translation 
B->A can also be performed with the same method.

Technically a tree has a single root node. If the dictionary 
parent_to_children contains more than one root node, we call it a polytree. 
The method get_all_trees_of_polytree() can be used to extract all trees  of 
a polytree.

We use two types of trees: with and without empty leafs. An empty leaf is an 
element of the tree=parent_to_children dictionary of the form "A"->[]. Trees 
without empty leafs cannot express the single node tree "A"->[]. Hence, 
we usually work with trees with empty leafs when doing tree calculations.


"""
from copy import copy
import treelib as tr


def get_mapped_polytree(polytree, fun):
    """
    This method takes as input a polytree whose nodes are of type A, 
    and returns a polytree whose nodes are of type B, where B = fun(A). For 
    example, str->Node or Node->Str.
    
    This method works whether polytree has empty leafs or not. If thev 
    polytree has empty leafs, it maps maps parent->[] to fun(parent)->[]

    Parameters
    ----------
    polytree: dict[str, list[str]]
    fun: function

    Returns
    -------
    polytree: dict[Any, list[Any]]

    """
    fun_polytree = {}
    for par, children in polytree.items():
        # if children =[], this maps par->[] to fun(par)->[]
        fun_polytree[fun(par)] = [fun(child) for child in children]
    return fun_polytree


def copy_polytree(polytree):
    """
    This method takes as input a polytree and returns a copy of the 
    polytree. The nodes are not copied (in case they are of type Node).
    
    This method works whether polytree has empty leafs or not.

    Parameters
    ----------
    polytree: dict[str, list[str]]

    Returns
    -------
    polytree: dict[str, list[str]]

    """
    return {par: copy(polytree[par]) for par in polytree}


def get_tree_depth(tree, root_node):
    """
    This method returns the depth of the tree `tree` with root node 
    `root_node`.
    
    If tree has empty leafs, depth is 1 more than if it doesn't due to last
    layer of empty leafs.

    num_depths = depth + 1

    Parameters
    ----------
    tree: dict[str, list[str]]
    root_node: str

    Returns
    -------
    int

    """
    if root_node not in tree:
        # If the root is not in the dictionary,
        # it's a leaf node with depth 0
        return 0

    children = tree[root_node]
    if not children:
        # If the root has no children, its depth is 1
        return 1
    child_depths = [get_tree_depth(tree, child) for child in children]
    return 1 + max(child_depths)


def get_all_nodes(polytree):
    """
    This method returns a list of all nodes of polytree `polytree`.
    
    This method works whether polytree has empty leafs or not. If it does,
    empty leaf nodes are not included in output list.

    Parameters
    ----------
    polytree: dict[str, list[str]]

    Returns
    -------
    list[str]

    """
    all_nodes = []
    for parent, children in polytree.items():
        if parent not in all_nodes:
            all_nodes.append(parent)
        for child in children:
            if child and child not in all_nodes:
                all_nodes.append(child)
    return all_nodes


def get_all_children(polytree):
    """
    This method returns a list of all the children of polytree `polytree`.
    
    This method works whether polytree has empty leafs or not. If it does,
    empty children are not included in output list.

    Parameters
    ----------
    polytree: dict[str, list[str]]

    Returns
    -------
    list[str]

    """
    all_children = []
    for parent, children in polytree.items():
        for child in children:
            if child and child not in all_children:
                all_children.append(child)
    return all_children


def get_root_nodes(polytree):
    """
    This method returns a list of all the nodes in polytree `polytree`.
    
    This method works whether polytree has empty leafs or not.

    Parameters
    ----------
    polytree: dict[str, list[str]]

    Returns
    -------
    list[str]

    """
    all_children = get_all_children(polytree)
    return [node for node in polytree if node not in all_children]


def count_leaf_nodes(polytree):
    """
    This method returns a tuple with the number of empty and nonempty leaf 
    nodes in polytree `polytree`.
    
    This method works whether polytree has empty leafs or not.

    Parameters
    ----------
    polytree: dict[str, list[str]]

    Returns
    -------
    tuple[int, int]

    """
    empty_leaf_count = 0
    nonempty_leaf_count = 0
    all_nodes = get_all_nodes(polytree)
    for node in all_nodes:
        if node not in polytree:
            nonempty_leaf_count += 1
        else:
            if polytree[node] == []:
                empty_leaf_count += 1
    return (empty_leaf_count, nonempty_leaf_count)


def all_leafs_are_nonempty(polytree):
    """
    This method returns True iff all the leafs of polytree `polytree` are 
    nonempty.

    Parameters
    ----------
    polytree: dict[str, list[str]]

    Returns
    -------
    bool

    """
    empty_leaf_count, _ = count_leaf_nodes(polytree)

    if empty_leaf_count == 0:
        return True
    else:
        return False


def all_leafs_are_empty(polytree):
    """
    This method returns True iff all the leafs of polytree `polytree` are 
    empty.

    Parameters
    ----------
    polytree: dict[str, list[str]]

    Returns
    -------
    bool

    """
    _, nonempty_leaf_count = count_leaf_nodes(polytree)

    if nonempty_leaf_count == 0:
        return True
    else:
        return False


def remove_empty_leafs(x):
    """
    This method returns a copy of x with all empty leafs removed. x can be 
    either a polytree or a list of polytrees.

    Parameters
    ----------
    x: dict[str, list[str]] | list[dict[str, list[str]]]

    Returns
    -------
    dict[str, list[str]] | list[dict[str, list[str]]]

    """

    def _remove_empty_leafs(polytree):
        if all_leafs_are_nonempty(polytree):
            return copy_polytree(polytree)
        new_polytree = {}
        for par in polytree.keys():
            if polytree[par]:
                new_polytree[par] = copy(polytree[par])
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
    """
    This method returns a copy of x in which all empty leafs have been 
    added. x can be either a polytree or a list of polytrees.

    Parameters
    ----------
    x: dict[str, list[str]] | list[dict[str, list[str]]]

    Returns
    -------
    dict[str, list[str]] | list[dict[str, list[str]]]

    """

    def _add_empty_leafs(polytree):
        if all_leafs_are_empty(polytree):
            return copy_polytree(polytree)
        all_nodes = get_all_nodes(polytree)
        new_polytree = copy_polytree(polytree)
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


def draw_tree(tree, root_node):
    """
    This method draws the tree `tree` with root node `root_node`.

    This method draws the same thing ( no empty leafs) whether `tree` has 
    empty leafs or not, but if it does, the method prints out a message 
    warning that "empty leafs present but not drawn".

    IMPORTANT: bug that must be fixed in treelib. In your Python
    installation, go to Lib\site-packages\treelib and edit tree.py. Find
    def show. The last line is:

    print(self.reader.encode('utf-8'))

    It should be:

    print(self.reader)

    Parameters
    ----------
    tree: dict[str, list[str]]
    root_node: str


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
        if count_leaf_nodes(tree)[0] > 0:
            print("WARNING: Empty leafs present but not drawn.")
            print(tree)
    except:
        print("*********************tree not possible")
        print(tree)


def draw_polytree(polytree):
    """
    This method calls the method draw_tree() for each of the trees contained 
    in the polytree `polytree`.

    This method draws the same thing ( no empty leafs) whether `polytree` 
    has empty leafs or not, but if it does, the method prints out a message 
    warning that "empty leafs present but not drawn".

    Parameters
    ----------
    polytree: dict[str, list[str]]

    Returns
    -------
    None

    """
    root_nodes = get_root_nodes(polytree)
    for root_node in root_nodes:
        tree = get_one_tree_of_polytree(polytree, root_node)
        draw_tree(tree, root_node)


def get_one_tree_of_polytree(polytree,
                             root_node,
                             output_empty_leafs=True):
    """
    This method returns one tree (the one with root node `root_node`) out of 
    presumably several trees of the polytree `polytree`. 
    
    The output tree will have empty leafs iff output_empty_leafs=True.
    
    This method works whether polytree has empty leafs or not.
    
    Parameters
    ----------
    polytree: dict[str, list[str]]
    root_node: str
    output_empty_leafs: bool

    Returns
    -------
    polytree: dict[str, list[str]]

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
    This method returns a list of all the trees of the polytree `polytree`. 
    The output trees will have empty leafs iff output_empty_leafs=True.
    
    The output trees will have empty leafs iff output_empty_leafs=True.
    
    This method works whether polytree has empty leafs or not.
    
    Parameters
    ----------
    polytree: dict[str, list[str]]
    output_empty_leafs: bool

    Returns
    -------
    list[dict[str, list[str]]]

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


def get_different_depth_subtrees(full_tree,
                                 root_node,
                                 output_empty_leafs=True,
                                 verbose=False):
    """
    This method returns a list of subtrees of the tree `full_tree`. The 
    subtrees are constrained to have the same root node as `full_tree` but 
    different depths.
    
    The output trees will have empty leafs iff output_empty_leafs=True.
    
    This method works whether `full_tree` has empty leafs or not.

    Parameters
    ----------
    full_tree: dict[str, list[str]]
    root_node: str
    output_empty_leafs: bool
    verbose: bool

    Returns
    -------
    list[dict[str, list[str]]]

    """
    full_tree0 = add_empty_leafs(full_tree)
    num_depths = get_tree_depth(full_tree0, root_node)
    init_tree = {root_node: []}
    l_tree = [init_tree]
    l_prev_leaf_node = [root_node]
    tree = copy_polytree(init_tree)
    if verbose:
        print("\nEntering get_different_depth_trees()")
        print(f"full tree:\n {full_tree}")
    for depth in range(num_depths):
        l_leaf_node = []
        for leaf_node in l_prev_leaf_node:
            children = full_tree0[leaf_node]
            tree[leaf_node] = children
            l_leaf_node += children
        # if output_empty_leafs:
        #     tree = add_empty_leafs(tree)
        # else:
        #     tree = remove_empty_leafs(tree)
        if depth < num_depths - 1:
            l_tree.append(copy_polytree(tree))
            l_prev_leaf_node = copy(l_leaf_node)
        else:
            if output_empty_leafs:
                l_tree = add_empty_leafs(l_tree)
            else:
                l_tree = remove_empty_leafs(l_tree)
            for depth1, tree1 in enumerate(l_tree):
                print(f"depth={depth1}, tree={tree1}")
            return l_tree


def get_all_paths_from_root(polytree,
                            root_nodes,
                            verbose=False):
    """
    This method returns a list of all the paths in polytree `polytree` that
    start at the root node `root_node` and end with a (nonempty) leaf node.

    This method works whether `polytree` has empty leafs or not. 
    Outputed paths do not have a [] at the end.

    Parameters
    ----------
    polytree: dict[Node, list[Node]]
    root_nodes: list[str]
    verbose: bool

    Returns
    -------
    list[list[str]]

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
    def get_paths_for_single_root_node(polytree1,
                                       cur_root_node,
                                       cur_path):
        cur_path = cur_path + [cur_root_node]
        polytree1 = add_empty_leafs(polytree1)
        if not polytree1[cur_root_node]:
            l_path_for1root.append(cur_path)
        else:
            for child in polytree1[cur_root_node]:
                if child:
                    get_paths_for_single_root_node(
                        polytree1=polytree1,
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
                    polytree1=subtree,
                    cur_root_node=root_node,
                    cur_path=[])
            if verbose:
                print(f"paths starting at root node = {root_node}:")
                print(l_path_for1root)
            l_path += l_path_for1root
    # Some paths will be repeated. Remove repeats.
    l_paso = []
    for path in l_path:
        if path not in l_paso:
            l_paso.append(path)
    return l_paso


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

        l_subtree = get_different_depth_subtrees(
            new_full_tree,
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
