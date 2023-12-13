"""

This file gives some global functions related to trees. These are used in
the class CCTree.

In this file, `tree` stands for a dictionary parent_to_children mapping each
parent node to a list children nodes.


"""
from copy import copy
import treelib as tr

def get_fun_tree(tree, fun):
    fun_tree = {}
    for par, children in tree.items():
        fun_tree[fun(par)] = [fun(child) for child in children]
    return fun_tree

def get_root_nodes(polytree):
    all_children = get_all_children(polytree)
    return [node for node in polytree if node not in all_children]

def get_trees_of_polytree(polytree,
                             root_node):
    subtree = {root_node: []}
    l_leaf_node = [root_node]
    while l_leaf_node:
        l_new_leaf_node = []
        for leaf_node in l_leaf_node:
            parents = polytree[leaf_node]
            subtree[leaf_node] = parents
            l_leaf_node += parents
            if l_new_leaf_node:
                l_leaf_node = l_new_leaf_node
            else:
                return subtree


def get_tree_depth(tree, root_node):
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

def get_polytree_tree(polytree,
                      root_node):
    

def draw_tree(tree, root_node):
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


def remove_empty_leafs(tree):
    new_tree = {}
    for par in tree.keys():
        if tree[par]:
            new_tree[par] = tree[par]
        else:
            pass
    return new_tree


def add_empty_leafs(tree):
    all_parents = list(tree.keys())
    all_children = get_all_children(tree)
    all_nodes = set(all_parents + all_children)
    new_tree = copy(tree)
    for node in all_nodes:
        if node not in tree.keys():
            new_tree[node] = []
    return new_tree


def get_all_children(tree):
    all_children = []
    for parent, children in tree.items():
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
    l_leaf_node = [root_node]
    tree = copy(init_tree)
    while depth < num_depths:
        if verbose:
            print(f"depth={depth}, tree=", tree)
        l_new_leaf_node = []
        depth += 1
        for leaf_node in l_leaf_node:
            parents = full_tree[leaf_node]
            tree[leaf_node] = parents
            l_new_leaf_node += parents
        if with_empty_leafs:
            l_subtree.append(remove_empty_leafs(copy(tree)))
        else:
            l_subtree.append(add_empty_leafs(copy(tree)))
        if depth >= num_depths:
            return l_subtree
        if l_new_leaf_node:
            l_leaf_node = l_new_leaf_node
        else:
            return l_subtree


def get_all_paths(root_nodes,
                  tree,
                  verbose=False):
    """
    multiplev root nodes

    Parameters
    ----------
    root_nodes: list[Node] | list[int]
    tree: dict[Node, list[Node]] | dict[int, list[int]]
    verbose: bool

    Returns
    -------
    list[list[Node]] | list[list[int]]

    """
    l_path_for1root = []
    tree0 = add_empty_leafs(tree)

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
        l_path = get_all_paths(root_nodes,
                               tree,
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
        tree_depth = get_tree_depth(tree, root_node)

        print(f"The depth of the tree is: {tree_depth}")

    def main4():
        parent_to_children = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F'],
            'D': ['G'],
            'F': ['H', 'I'],
            'A1': ['B1'],
            "B1": []
        }
        for root_node in [ "A", "A1"]:
            draw_tree(parent_to_children, root_node)


    main1()
    main2()
    main3()
    main4()
