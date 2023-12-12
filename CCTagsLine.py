class CCTagsLine:
    """
    This class is simply a container for the info in a cctags line>
    cctags lines are found under the osent of each sample in a cctags.txt file.

    """

    def __init__(self, depth, l_ilabel):
        """
        Constructor


        Parameters
        ----------
        depth: int
        l_ilabel: list[int]
        """
        self.depth = depth
        self.cclocs = []
        self.sep_locs = []
        self.others_locs = []
        self.spans = []
        # similar to Openie6.metric.get_coords()

        start_loc = -1
        started_CP = False  # CP stands for coordinating phrase

        # CCTAG_TO_ILABEL = {
        #   'NONE': 0
        #   'CP': 1,
        #   'CP_START': 2,
        #    'CC': 3,
        #    'SEP': 4,
        #    'OTHERS': 5
        # }

        for i, ilabel in enumerate(l_ilabel):
            # print("lmk90", i, spans, ccloc, started_CP,
            #       "ilabel=", ilabel)
            if ilabel != 1:  # 1=CP
                if started_CP:
                    started_CP = False
                    self.spans.append((start_loc, i))
            # We reject illegal cclocs later on.
            # if ilabel in [0, 2]:  # NONE or CP_START
            #     ccnode phrase can end
            #     two spans at least, split by CC
            #     if self.spans and len(self.spans) >= 2 and \
            #             self.spans[0][1] <= ccloc < self.spans[-1][0]:
            #         self.cclocs.append(ccloc)
            #         ccloc = -1
            if ilabel == 0:  # 0=NONE
                pass
            elif ilabel == 1:  # 1=CP
                if not started_CP:
                    started_CP = True
                    start_loc = i
            elif ilabel == 2:  # 2=CP_START
                # print("hjuk", "was here")
                started_CP = True
                start_loc = i
            elif ilabel == 3:  # 3=CC
                # ccloc = i
                # not this because want to reject cclocs
                # that are illegal, which may happen during training
                # self.cclocs.append(i)

                # decided to reject illegal cclocs later on
                self.cclocs.append(i)
            elif ilabel == 4:  # 4=SEP
                self.sep_locs.append(i)
            elif ilabel == 5:  # 5=OTHERS
                self.others_locs.append(i)
            elif ilabel == -100:
                pass
            else:
                assert False, f"{str(ilabel)} out of range 0:6"

    @staticmethod
    def get_span_pair(spans, ccloc, throw_if_None=True):
        """
        similar to Openie6.metric.Coordination.get_pair()

        This method returns two **consecutive** spans such that `ccloc` is
        between the 2 spans but outside both of them. If no span is found
        and throw_if_None=False, it raises an exception.

        Parameters
        ----------
        spans: list[tuple[int]]
        ccloc: int
        throw_if_None: bool

        Returns
        -------
        list[tuple[int,int], tuple[int, int]] | None

        """
        span_pair = None
        for i in range(1, len(spans)):
            if ccloc < spans[i][0]:
                span_pair = (spans[i - 1], spans[i])
                # there must be at least one point between the
                # 2 spans, or else the 2 spans would be 1
                if span_pair[0][1] <= ccloc \
                        < span_pair[1][0]:
                    break
                else:
                    span_pair = None
        if throw_if_None and not span_pair:
            raise LookupError(
                f"Could not find any span_pair for index={ccloc}.")
        return span_pair
