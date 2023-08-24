class SampleChild:
    def __init__(self, tokens):
        self.tokens=tokens
    def get_token_str(self):
        return " ".join(self.tokens)
    def get_nontrivial_locs(self):
        locs = []
        for loc, token in enumerate(self.tokens):
            if token != "NONE":
                locs.append(loc)
        return locs