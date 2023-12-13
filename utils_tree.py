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


"""
from copy import copy
import treelib as tr


def get_fun_polytree(polytree, fun):
    fun_polytree = {}
    for par, children in polytree.items():
        fun_polytree[fun(par)] = [fun(child) for child in children]
    return fun_polytree


def get_root_nodes(polytree):
    all_children = get_all_children(polytree)
    return [node for node in polytree if node not in all_children]


def get_tree_depth(tree, root_node):
    """
    If tree has empty leafs, depth is 1 more than if it doesn't.

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
                             empty_leafs_flag=False):
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
                if empty_leafs_flag:
                    return add_empty_leafs(tree0)
                else:
                    return remove_empty_leafs(tree0)


def get_all_trees_of_polytree(polytree):
    root_nodes = get_root_nodes(polytree)
    l_tree = []
    for root_node in root_nodes:
        tree = get_one_tree_of_polytree(polytree, root_node)
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

    Tree draws the same with and without empty leafs.

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
    except:
        print("*********************tree not possible")
        print(tree)


def draw_polytree(polytree):
    root_nodes = get_root_nodes(polytree)
    for root_node in root_nodes:
        tree = get_one_tree_of_polytree(polytree, root_node)
        draw_tree(tree, root_node)


def remove_empty_leafs(polytree):
    new_polytree = {}
    for par in polytree.keys():
        if polytree[par]:
            new_polytree[par] = polytree[par]
        else:
            pass
    return new_polytree


def add_empty_leafs(polytree):
    all_parents = list(polytree.keys())
    all_children = get_all_children(polytree)
    all_nodes = set(all_parents + all_children)
    new_polytree = copy(polytree)
    for node in all_nodes:
        if node not in polytree.keys():
            new_polytree[node] = []
    return new_polytree


def get_all_children(polytree):
    all_children = []
    for parent, children in polytree.items():
        for child in children:
            if child and child not in all_children:
                all_children.append(child)
    return all_children


def get_different_depth_subtrees(full_tree,
                                 root_node,
                                 num_depths,
                                 with_empty_leafs=False,
                                 verbose=False):
    """
    all subtrees with same root node as full tree
    but different depths.
    
    Parameters
    ----------
    full_tree
    root_node
    num_depths
    with_empty_leafs

    Returns
    -------

    """
    depth = 0
    init_tree = {root_node: []}
    l_subtree = [init_tree]
    l_prev_leaf_node = [root_node]
    tree = copy(init_tree)
    while depth < num_depths:
        if verbose:
            print(f"depth={depth}, tree=", tree)
        l_leaf_node = []
        depth += 1
        for leaf_node in l_prev_leaf_node:
            parents = full_tree[leaf_node]
            tree[leaf_node] = parents
            l_leaf_node += parents
        if with_empty_leafs:
            l_subtree.append(remove_empty_leafs(copy(tree)))
        else:
            l_subtree.append(add_empty_leafs(copy(tree)))
        if depth >= num_depths:
            return l_subtree
        if l_leaf_node:
            l_prev_leaf_node = copy(l_leaf_node)
        else:
            return l_subtree


def get_all_paths(polytree,
                  root_nodes=None,
                  verbose=False):
    """
    multiplev root nodes

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
    
    if not root_nodes:
        root_nodes = get_root_nodes(polytree)
    tree0 = add_empty_leafs(polytree)

    # init input:
    # cur_root_node = root_node
    # cur_path =[]
    # init_output
    # l_path_for1root = []
    def get_paths_for_single_root_node(cur_root_node,
                                       cur_path):
        cur_path = cur_path + [cur_root_node]
        if not tree0[cur_root_node]:
            l_path_for1root.append(cur_path)
        else:
            for child in tree0[cur_root_node]:
                get_paths_for_single_root_node(cur_root_node=child,
                                               cur_path=cur_path)
        return l_path_for1root

    l_path = []
    for root_node in root_nodes:
        l_path_for1root = []
        l_path_for1root = \
            get_paths_for_single_root_node(root_node, cur_path=[])
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

        # leaf nodes must be in!
        tree = {
            'A': ['B'],
            'B': ['C', 'D'],
            'C': ['E', "F"],
            'A1': ['B1'],
            "B1": []
        }
        root_nodes = get_root_nodes(tree)
        print("root nodes=", root_nodes)
        l_path = get_all_paths(tree,
                               root_nodes,
                               verbose=True)
        print("l_path:\n", l_path)


    def main2():
        full_tree = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F']}
        root_node = "A"
        num_depths = 2

        print()
        print("full_tree:\n", full_tree)
        new_full_tree = add_empty_leafs(full_tree)
        print("added empty leafs:\n", new_full_tree)
        new_full_tree = remove_empty_leafs(new_full_tree)
        print("added then removed empty leafs:\n", new_full_tree)

        l_subtree = get_different_depth_subtrees(new_full_tree,
                                                 root_node,
                                                 num_depths,
                                                 with_empty_leafs=False,
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
