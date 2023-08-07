from collections import OrderedDict
# cli= command line interface

with open("openie6-hparams0.txt", mode="r") as f0:
    lines = f0.readlines()


def sort_lines(lines):
    block_start_locs = []
    block_end_locs = []
    for i in range(len(lines)):
        if len(lines[i]) > 2:
            curr = lines[i][0:2]
            if i == 0:
                prior = None
            else:
                if len(lines[i - 1]) > 2:
                    prior = lines[i - 1][0:2]
                else:
                    prior = None
            if i == len(lines) - 1:
                next = None
            else:
                if len(lines[i + 1]) > 2:
                    next = lines[i + 1][0:2]
                else:
                    next = None
            if curr == "--" and next != "--":
                block_end_locs.append(i)
            if prior != "--" and curr == "--":
                block_start_locs.append(i)
    assert (len(block_start_locs) == len(block_end_locs))
    block_spans = list(zip(block_start_locs, block_end_locs))
    # print("llkl", block_start_locs, block_end_locs)
    for start, end in block_spans:
        # print("cfgh", sorted(lines[start:end+1], key=lambda x:x[2:]))
        # print("llpp", lines[end])
        lines[end] = lines[end].replace("}", "")
        # print("llpp", lines[end])
        lines[start:end + 1] = sorted(lines[start:end + 1],
                                      key=lambda x: x[2:])
        lines[end] = lines[end].strip() + "}"

    return lines


# comment this out if sorting of parameter names in not desired
lines = sort_lines(lines)
# print("rted", lines)

distinct_params = []
with open("openie6-hparams.txt", mode="w") as f:
    for line in lines:
        if len(line) > 2 and line[0:2] == "--":
            # print("sdty", line[2:])
            words = line[2:].split()
            if len(words) == 2:
                key, value = line[2:].split()
                if value[-1] != "}":
                    ending = ",\n"
                else:
                    value = value[:-1]
                    ending = "\n}"
                key = '"' + key + '"'
                if not value.isdigit():
                    value = '"' + value + '"'

                f.write("    " + key + ": " + value + ending)
                distinct_params.append(key)
            elif len(words) == 1:
                key = words[0]
                if key[-1] != "}":
                    ending = ",\n"
                else:
                    key = key[:-1]
                    ending = "\n}"
                key = '"' + key + '"'
                f.write("    " + key + ": True" + ending)
                distinct_params.append(key)
            else:
                assert False, words
        else:
            f.write(line)

        distinct_params = sorted(list(set(distinct_params)))
        param_to_values = OrderedDict()
        for param in distinct_params:
            param_to_values[param] = set()

with open("openie6-hparams.txt", mode="r") as f:
    for line in f:
        if any([param in line for param in param_to_values.keys()]):
            param, value = line.replace(",", "").split(":")
            param = param.strip()
            value = value.strip()
            param_to_values[param].add(value)

with open("openie6-hparams.txt", mode="a") as f:
    f.write("\n\nparameters to set of possible values:\n")
    f.write("{\n")
    for param in param_to_values:
        values = param_to_values[param]
        str0 = "    " + param + ": ("
        for value in values:
            str0 += value + ", "
        f.write(str0 + "),\n")
    f.write("}\n")
