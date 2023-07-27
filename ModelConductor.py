import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import shutil
from glob import glob
from Model import *
from my_globals import *



class ModelConductor:
    def __init__(self, task):

        assert task in ["ex", "cc"]
        self.task = task
        self.saved = False
        self.save_dir = "data/save"
        self.has_cuda = torch.cuda.is_available()

        if task == 'cc':
            self.train_fp = 'data/openie-data/ptb-train.labels'
            self.dev_fp = 'data/openie-data/ptb-dev.labels'
            self.test_fp = 'data/openie-data/ptb-test.labels'
        elif task == 'ex':
            self.train_fp = 'data/openie-data/openie4_labels'
            self.dev_fp = 'data/carb-data/dev.txt'
            self.test_fp = 'data/carb-data/test.txt'

        train_dataset, val_dataset, test_dataset, \
            meta_data_vocab, all_sentences = data.process_data(hparams)
        self.train_dataloader = DataLoader(train_dataset,
                                      batch_size=hparams.batch_size,
                                      collate_fn=data.pad_data,
                                      shuffle=True,
                                      num_workers=1)
        self.val_dataloader = DataLoader(val_dataset,
                                    batch_size=hparams.batch_size,
                                    collate_fn=data.pad_data,
                                    num_workers=1)
        self.test_dataloader = DataLoader(test_dataset,
                                     batch_size=hparams.batch_size,
                                     collate_fn=data.pad_data,
                                     num_workers=1)

    def set_checkpoint_callback(self):
        if self.saved:
            self.checkpoint_callback = ModelCheckpoint(
                filepath=self.save_dir + '/{epoch:02d}_{eval_acc:.3f}',
                verbose=True,
                monitor='eval_acc',
                mode='max',
                save_top_k=hparams.save_k if not hparams.debug else 0,
                period=0)
        else:
            self.checkpoint_callback = None

    def get_all_checkpoint_paths(self):
        return glob(self.save_dir + '/*.ckpt')

    def get_checkpoint_path(self):
        all_paths = glob(self.save_dir + '/*.ckpt')
        assert len(all_paths) == 1
        return all_paths[0]

    def get_logger(self, mode):
        log_dir = hparams.save + '/logs/'
        if os.path.exists(log_dir + f'{mode}'):
            mode_logs = list(glob(log_dir + f'/{mode}_*'))
            new_mode_index = len(mode_logs) + 1
            print('Moving old log to...')
            print(shutil.move(hparams.save + f'/logs/{mode}',
                        hparams.save + f'/logs/{mode}_{new_mode_index}'))
        logger = TensorBoardLogger(
            save_dir=hparams.save,
            name='logs',
            version=mode + '.part')
        return logger

    def get_trainer(self, logger, checkpoint_path=None):
        # trainer = Trainer(
        #     accumulate_grad_batches = int(hparams.accumulate_grad_batches),
        #     checkpoint_callback = self.checkpoint_callback,
        #     gpus = hparams.gpus,
        #     gradient_clip_val = hparams.gradient_clip_val,
        #     logger = logger,
        #     max_epochs = hparams.epochs,
        #     min_epochs = hparams.epochs,
        #     num_sanity_val_steps = hparams.num_sanity_val_steps,
        #     num_tpu_cores = hparams.num_tpu_cores,
        #     resume_from_checkpoint = checkpoint_path,
        #     show_progress_bar = True,
        #     track_grad_norm = hparams.track_grad_norm,
        #     train_percent_check = hparams.train_percent_check,
        #     use_tpu = hparams.use_tpu,
        #     val_check_interval = hparams.val_check_interval)

        trainer = Trainer(
            checkpoint_callback = self.checkpoint_callback,
            logger = logger,
            resume_from_checkpoint = checkpoint_path,
            show_progress_bar = True,
            **hparams)
        return trainer

    def override_args(loaded_hparams_dict, current_hparams_dict,
                      cline_sys_args):
        # override the values of loaded_hparams_dict with
        # the values i current_hparams_dict
        # (only the keys in cline_sys_args)
        for arg in cline_sys_args:
            if '--' in arg:
                key = arg[2:]
                loaded_hparams_dict[key] = current_hparams_dict[key]

        for key in current_hparams_dict:
            if key not in loaded_hparams_dict:
                loaded_hparams_dict[key] = current_hparams_dict[key]

        return loaded_hparams_dict

    def train(self):
        self.set_checkpoint_callback()
        model = Model()
        logger = self.get_logger('train')
        trainer = self.get_trainer(logger)
        trainer.fit(model, train_dataloader=self.train_dataloader,
                    val_dataloaders=self.val_dataloader)
        shutil.move(hparams.save + f'/logs/train.part',
                    hparams.save + f'/logs/train')

    def resume(self):
        self.set_checkpoint_callback()
        checkpoint_path = self.get_checkpoint_path()
        if self.has_cuda:
            loaded_hparams_dict = torch.load(checkpoint_path)['hparams']
        else:
            loaded_hparams_dict = torch.load(
                checkpoint_path, map_location=torch.device('cpu'))['hparams']

    # def resume(hparams, checkpoint_callback, meta_data_vocab,
    #            train_dataloader, val_dataloader, test_dataloader,
    #            all_sentences):

        current_hparams_dict = vars(hparams)
        loaded_hparams_dict = data.override_args(
            loaded_hparams_dict, current_hparams_dict, sys.argv[1:])
        loaded_hparams = data.convert_to_namespace(loaded_hparams_dict)

        model = Model()
        logger = self.get_logger('resume')
        trainer = self.get_trainer(logger, checkpoint_path)
        trainer.fit(model, train_dataloader=self.train_dataloader,
                    val_dataloaders=self.val_dataloader)
        shutil.move(hparams.save + f'/logs/resume.part',
                    hparams.save + f'/logs/resume')

    def test(self, train):
        self.set_checkpoint_callback()
        checkpoint_path = self.get_checkpoint_path()
        if not train:
            if self.has_cuda:
                loaded_hparams_dict = torch.load(checkpoint_path)[
                    ....]
            else:
                loaded_hparams_dict = \
                    torch.load(checkpoint_path,
                        map_location=torch.device('cpu'))[
                        ....]

    def predict(self):
        self.set_checkpoint_callback()
        
    def splitpredict(self):
        self.set_checkpoint_callback()
