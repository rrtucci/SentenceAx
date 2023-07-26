from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from glob import glob
import torch


class Conductor:
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

        self.gradient_clip_val = 5

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

    def train(self):
        self.set_checkpoint_callback()

    def resume(self):
        self.set_checkpoint_callback()
        checkpoint_path = self.get_checkpoint_path()
        if self.has_cuda:
            loaded_hparams_dict = torch.load(checkpoint_path)['hparams']
        else:
            loaded_hparams_dict = torch.load(
                checkpoint_path, map_location=torch.device('cpu'))['hparams']

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
