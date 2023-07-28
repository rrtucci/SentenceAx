class ClassFromDict(dict):
    """
    dot instead of [] notation access to dictionary attributes.
    Nice to know but won't use.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def update_dict(dict, new_dict, add_new_keys=True):
    for key in dict:
        if key in new_dict: # overlapping keys
            dict[key] = new_dict[key]
    if add_new_keys:
        for key in new_dict:
            if key not in dict: # new keys not in dict yet
                dict[key] = new_dict[key]

if __name__ == "__main__":
    def main():
        h = {"x": 5, "y": 3}
        H = ClassFromDict(h)
        print(H.x)  # Output: 5
        print(H.y)  # Output: 3
        H.y = 5
        print(H.y, h["y"]) # output 5,3

        def F(x, y):
            return x + y

        print(F(**h))  # Output: 8

    main()