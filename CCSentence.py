class CCSentence:
    def __init__(self, words):
        self.words = words
        self.cclists = None

    def get_tree(self):
        # par = parent
        par_ccloc_to_child_cclocs = {}
        child_ccloc_to_par_ccloc = {}
        child_cclocs = []

        for par_cclist in self.cclists:
            par_ccloc = par_cclist.ccloc
            child_cclocs.append([])
            for child_cclist in self.cclists:
                child_ccloc = child_cclist.ccloc
                if child_cclist is not None:
                    if par_cclist.is_parent(child_cclist):
                        child_cclocs[-1].append(child_ccloc)

            par_ccloc_to_child_cclocs[par_ccloc] = child_cclocs[-1]

        child_cclocs.sort(key=list.__len__)

        for i in range(0, len(child_cclocs)):
            for child_ccloc in child_cclocs[i]:
                for j in range(i + 1, len(child_cclocs)):
                    if child_ccloc in child_cclocs[j]:
                        child_cclocs[j].remove(child_ccloc)

        for cclist in self.cclists:
            ccloc = cclist.ccloc
            for child_ccloc in par_ccloc_to_child_cclocs[ccloc]:
                child_ccloc_to_par_ccloc[child_ccloc] = ccloc

        root_ccloc = []
        for cclist in self.cclists:
            par_ccloc = cclist.ccloc
            if par_ccloc not in child_ccloc_to_par_ccloc:
                root_ccloc.append(par_ccloc)

        return root_ccloc, child_ccloc_to_par_ccloc, par_ccloc_to_child_cclocs

    def fix_spanned_locs_list(self, spanned_locs_list, cclists_same_level, 
                spanned_locs_locs):
        for cclist in cclists_same_level:

            if len(spanned_locs_list) == 0:

                for span in cclist.spans:
                    spanned_locs = []
                    for i in range(span[0], span[1]):
                        spanned_locs.append(i)
                    spanned_locs_list.append(spanned_locs)

                min = cclist.spans[0][0]
                max = cclist.spans[-1][1] - 1

                for spanned_locs in spanned_locs_list:
                    for i in spanned_locs_locs:
                        if i < min or i > max:
                            spanned_locs.append(i)

            else:
                to_be_added_sents = []
                to_be_removed_sents = []

                for spanned_locs in spanned_locs_list:
                    if cclist.spans[0][0] in spanned_locs:
                        spanned_locs.sort()

                        min = cclist.spans[0][0]
                        max = cclist.spans[-1][1] - 1

                        for span in cclist.spans:
                            new_spanned_locs = []
                            for i in spanned_locs:
                                if i in range(span[0], span[1]) or \
                                        i < min or i > max:
                                    new_spanned_locs.append(i)

                            to_be_added_sents.append(new_spanned_locs)

                        to_be_removed_sents.append(spanned_locs)


                for sent in to_be_removed_sents:
                    spanned_locs_list.remove(sent)
                spanned_locs_list.extend(to_be_added_sents)

    def fix_cclists(self):
        for cclist in self.cclists:
            if self.words[cclist.ccloc] in ['nor', '&'] or\
                    cclist.contains_unbreakable_spans():
                k = self.cclists.index(cclist)
                self.cclists.pop(k)

    def get_spanned_locs_list(self):

        self.fix_cclists()

        spanned_words = []
        for cclist in self.cclists:
            for span in cclist.spans:
                spanned_words.append(' '.join(self.words[span[0]:span[1]]))

        root_cclocs, child_ccloc_to_par_ccloc, par_ccloc_to_child_ccloc = \
            self.get_tree()

        spanned_locs_list = []
        root_count = len(root_cclocs)
        new_child_count = 0

        conj_same_level = []

        while len(root_cclocs) > 0:

            root_cclocs.pop(0)
            root_count -= 1
            conj_same_level.append(root_cclocs)

            for child_ccloc in par_ccloc_to_child_ccloc[root_cclocs]:
                root_cclocs.append(child_ccloc)
                new_child_count += 1

            if root_count == 0:
                self.fix_spanned_locs_list(spanned_locs_list, conj_same_level)
                root_count = new_child_count
                new_child_count = 0
                conj_same_level = []
        spanned_phrases = [' '.join([self.words[i] for i in
                    sorted(spanned_locs)]) for spanned_locs in
                                  spanned_locs_list]

        return spanned_phrases, spanned_words, spanned_locs_list
