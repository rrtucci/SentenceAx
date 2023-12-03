"""

I inserted this script inside the Openie6 code to get a sorted list of all
`hparams.` and `batch.`

"""
import re
import os
from pprint import pprint

def find_hparams_variables(files):
    hparams_variables = set()
    variable_pattern = r'\bhparams\.\w+\b'
    # variable_pattern = r'\bbatch\.\w+\b'

    for file_path in files:
        with open(file_path, "r") as file:
            content = file.read()
            variables = re.findall(variable_pattern, content)
            hparams_variables.update(variables)

    return sorted(list(hparams_variables))

if __name__ == "__main__":

    def main():
        files = ["data.py",
                 "data_preprocessing.py",
                 "metric.py",
                 "model.py",
                 "params.py",
                 "run.py"]
        li = find_hparams_variables(files)
        pprint(li)

    main()

# output.meta_data for hparams.
# ['hparams.accumulate_grad_batches',
#  'hparams.batch_size',
#  'hparams.bos_token_id',
#  'hparams.build_cache',
#  'hparams.checkpoint',
#  'hparams.conj_model',
#  'hparams.constraints',
#  'hparams.cweights',
#  'hparams.debug',
#  'hparams.dev_fp',
#  'hparams.dropout',
#  'hparams.eos_token_id',
#  'hparams.epochs',
#  'hparams.gpus',
#  'hparams.gradient_clip_val',
#  'hparams.inp',
#  'hparams.iterative_layers',
#  'hparams.labelling_dim',
#  'hparams.lr',
#  'hparams.max_steps',
#  'hparams.mode',
#  'hparams.model_str',
#  'hparams.multi_opt',
#  'hparams.no_lt',
#  'hparams.num_extractions',
#  'hparams.num_sanity_val_steps',
#  'hparams.num_tpu_cores',
#  'hparams.oie_model',
#  'hparams.optimizer',
#  'hparams.out',
#  'hparams.predict_fp',
#  'hparams.rescore_model',
#  'hparams.rescoring',
#  'hparams.save',
#  'hparams.save_k',
#  'hparams.split_fp',
#  'hparams.task',
#  'hparams.test_fp',
#  'hparams.track_grad_norm',
#  'hparams.train_fp',
#  'hparams.train_percent_check',
#  'hparams.type',
#  'hparams.use_tpu',
#  'hparams.val_check_interval',
#  'hparams.wreg',
#  'hparams.write_allennlp',
#  'hparams.write_async']

# output for batch.
# ['batch.labels',
#  'batch.meta_data',
#  'batch.pos_index',
#  'batch.text',
#  'batch.verb',
#  'batch.verb_index',
#  'batch.word_starts']
