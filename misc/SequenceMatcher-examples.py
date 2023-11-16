"""

This file illustrates the use of difflib.SequenceMatcher
Ref.
https://stackoverflow.com/questions/35517353/how-does-pythons-sequencematcher-work

"""
import difflib


def print_example(a, b):
    sm = difflib.SequenceMatcher(None, a, b)
    print()
    print("a=", a)
    print("b=", b)
    print(sm.get_matching_blocks())


print_example(a='ACT', b='ACTGACT')

# output [Match(a=0, b=0, size=3), Match(a=3, b=7, size=0)]

# Match(a=0, b=0, size=3): This indicates that a matching block of size 3
# starts at index 0 in sequence a and index 0 in sequence b.
#
# Match(a=3, b=7, size=0): This indicates that there is a matching block
# of size 0 starting at index 3 in sequence a and index 7 in sequence b

# a and b stand for sequences in the __init__, but for sequence locations in
# get_matching_blocks()

print_example(a='abcdef', b='acdef')

print_example(a=['apple', 'banana', 'orange', 'kiwi'],
              b=['apple', 'grape', 'banana', 'kiwi'])
