with open("openie-hparams0.txt", mode="r") as f0:
    lines = f0.readlines()

def sort_lines():
    block_start_locs =[]
    block_end_locs = []
    for i in range(len(lines)):
        if len(lines[i])>2:
            curr = lines[i][0:2]
            if i==0:
                prior = None
            else:
                if len(lines[i-1])>2:
                    prior = lines[i-1][0:2]
                else:
                    prior = None
            if i == len(lines)-1:
                next = None
            else:
                if len(lines[i+1])>2:
                    next = lines[i+1][0:2]
                else:
                    next=None
            if curr== "--" and next!= "--":
                block_end_locs.append(i-1)
            if prior!= "--" and curr== "--":
                block_start_locs.append(i)
    assert(len(block_start_locs)==len(block_end_locs))
    block_spans = list(zip(block_start_locs, block_end_locs))
    # print("llkl", block_start_locs, block_end_locs)
    for start, end in block_spans:
        # print("cfgh", sorted(lines[start:end+1], key=lambda x:x[2:]))
        lines[start:end+1] = sorted(lines[start:end+1],
                                    key=lambda x:x[2:])

 # comment this out if sorting of parameter names in not desired
sort_lines()

with open("openie-hparams.txt", mode="w") as f:
    for line in lines:
        if lines[0:2] == "--":
            # print("llk", lines[2:])
            splitting = lines[2:].split()
            if len(splitting)==2:
                key, value = lines[2:].split()
                if value[-1] != "}":
                    ending = ",\n"
                else:
                    value = value[:-1]
                    ending = "\n}"
                f.write("    " + key + ": " + value + ending)
            elif len(splitting)==1:
                key = splitting[0]
                if key[-1] != "}":
                    ending = ",\n"
                else:
                    key = key[:-1]
                    ending = "\n}"
                f.write("    " + key + ": True" + ending)
            else:
                assert False
        else:
            f.write(line)
