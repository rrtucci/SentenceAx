def set_simple_to_complex_sents_dict(self):
    """
    formerly data_processing.load_conj_mapping()

    Returns
    -------

    """
    simple_to_complex_sents = {}
    content = open(PARAMS_D["conj_map"]).read()
    complex_sent = ''
    for sample in content.split('\n\n'):
        for i, line in enumerate(sample.strip('\n').split('\n')):
            if i == 0:
                complex_sent = line
            else:
                simple_to_complex_sents[line] = complex_sent
    return simple_to_complex_sents