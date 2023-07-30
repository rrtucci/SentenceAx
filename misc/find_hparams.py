"""

I inserted this script inside the Openie6 code to get a sorted list of all
params_d names.

"""

import re
import os
from pprint import pprint

def find_params_d_variables(files):
    params_d_variables = set()
    variable_pattern = r'\bparams_d\.\w+\b'

    for file_path in files:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()
            variables = re.findall(variable_pattern, content)
            params_d_variables.update(variables)

    return sorted(list(params_d_variables))

if __name__ == "__main__":

    def main():
        files = ["data.py",
                 "data_preprocessing.py",
                 "metric.py",
                 "model.py",
                 "params.py",
                 "run.py"]
        li = find_params_d_variables(files)
        pprint(li)

    main()

# output of this script:
# ['params_d.accumulate_grad_batches',
#  'params_d.batch_size',
#  'params_d.bos_token_id',
#  'params_d.build_cache',
#  'params_d.checkpoint',
#  'params_d.conj_model',
#  'params_d.constraints',
#  'params_d.cweights',
#  'params_d.debug',
#  'params_d.dev_fp',
#  'params_d.dropout',
#  'params_d.eos_token_id',
#  'params_d.epochs',
#  'params_d.gpus',
#  'params_d.gradient_clip_val',
#  'params_d.inp',
#  'params_d.iterative_layers',
#  'params_d.labelling_dim',
#  'params_d.lr',
#  'params_d.max_steps',
#  'params_d.mode',
#  'params_d.model_str',
#  'params_d.multi_opt',
#  'params_d.no_lt',
#  'params_d.num_extractions',
#  'params_d.num_sanity_val_steps',
#  'params_d.num_tpu_cores',
#  'params_d.oie_model',
#  'params_d.optimizer',
#  'params_d.out',
#  'params_d.predict_fp',
#  'params_d.rescore_model',
#  'params_d.rescoring',
#  'params_d.save',
#  'params_d.save_k',
#  'params_d.split_fp',
#  'params_d.task',
#  'params_d.test_fp',
#  'params_d.track_grad_norm',
#  'params_d.train_fp',
#  'params_d.train_percent_check',
#  'params_d.type',
#  'params_d.use_tpu',
#  'params_d.val_check_interval',
#  'params_d.wreg',
#  'params_d.write_allennlp',
#  'params_d.write_async']
#

