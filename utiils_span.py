"""

This file contains various methods that are useful when dealing with spans
found in classes CCNode and CCTree. A span is just a tuple of 2 integers
such as (a, b)= (3, 6). It represents the set of integers range(a, b).

"""
def is_sub_span(span0, span1):
    """

    Parameters
    ----------
    span0: tuple[int, int]
    span1: tuple[int, int]

    Returns
    -------
    bool

    """
    a0, b0 = span0
    a1, b1 = span1
    # sub_span empty
    if b0-a0 < 1:
        return True
    return a1 <= a0 and b1 >= b0

def span_len(span):
    """

    Parameters
    ----------
    span: tuple[int, int]

    Returns
    -------
    int

    """
    a, b = span
    if b-a<=0:
        return 0
    else:
        return b-a

def span_set(span):
    """

    Parameters
    ----------
    span: tuple[int, int]

    Returns
    -------
    set[int]

    """
    if not span_len(span):
        return set()
    else:
        return set(range(*span))

def span_difference(span0, span1):
    """

    Parameters
    ----------
    span0: tuple[int, int]
    span1: tuple[int, int]

    Returns
    -------
    tuple[int, int]

    """
    a0, b0 = span0
    a1, b1 = span1
    assert span_len(span0) > 0
    if span_len(span1) == 0:
        return span0
    if b0 <= a1:
        return span0
    elif a0< a1 < b0:
        return (a0, a1)
    elif a0 == a1 and b1< b0:
        return (b1, b0)
    elif a0 < a1 and b1==b0:
        return (a0, a1)
    elif a0 < a1 and b1 < b0:
        return (a0, a1), (b1, b0)
    elif a1<= a0 and b0<= b1:
        return None
    elif a1<=a0 and b1< b0:
        return (b1, b0)
    elif b1 <= a0:
        return None
    else:
        assert False

def in_span(i, span):
    """

    Parameters
    ----------
    i: int
    span: tuple[int, int]

    Returns
    -------
    bool

    """
    a, b = span
    if i>=a and i<b:
        return True
    else:
        return False

def span_path_is_decreasing(span_path):
    """

    Parameters
    ----------
    span_path: list[tuple[int, int]]

    Returns
    -------
    bool

    """
    len_path = len(span_path)
    for k in range(len_path - 1):
        if not is_sub_span(span_path[k + 1], span_path[k]):
            return False
    return True

def are_disjoint(span0, span1):
    """

    Parameters
    ----------
    span0: tuple[int, int]
    span1: tuple[int, int]

    Returns
    -------
    bool

    """
    a0, b0 = span0
    a1, b1 = span1
    return b0<=a1 or b1<=a0

def draw_inc_exc_spans(all_span, inc_span, exc_span):
    """

    Parameters
    ----------
    all_span: tuple[int, int]
    inc_span: tuple[int, int]
    exc_span: tuple[int, int]

    Returns
    -------
    None

    """
    # print("mmner", all_span, inc_span, exc_span)
    assert is_sub_span(inc_span, all_span)
    assert is_sub_span(exc_span, all_span)
    assert are_disjoint(inc_span, exc_span)
    li = ["_"]*span_len(all_span)
    for i in range(len(li)):
        if in_span(i, exc_span):
            li[i] = "E"
        elif in_span(i, inc_span):
            li[i] = "I"
    print("".join(li))

def draw_inc_exc_span_paths(all_span, inc_span_path, exc_span_path):
    """

    Parameters
    ----------
    all_span: tuple[int, int]
    inc_span_path: list[tuple[int, int]]
    exc_span_path: list[tuple[int, int]]

    Returns
    -------
    None

    """
    assert len(inc_span_path) == len(exc_span_path)
    for i in range(len(inc_span_path)):
        draw_inc_exc_spans(all_span, inc_span_path[i], exc_span_path[i])