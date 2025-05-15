import torch
import soundfile as sf
from typing import Union
from pathlib import Path
import numpy as np
import random
import pytorch_lightning as pl
from copy import deepcopy
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor

import queue
import threading


class DataLoadIter:
    def __init__(self, data_src_dir: Union[str, Path], data_tgt_dir: Union[str, Path], 
                is_train: bool, batch_size: int = 1, cut_len: int = 32000, num_workers: int = 1, prefetch: int = 0):
        self.is_train = is_train
        self.batch_size = batch_size
        self.cut_len = cut_len
        self.data_src_dir = Path(data_src_dir)
        self.data_tgt_dir = Path(data_tgt_dir)
        self.wav_names = [p.stem for p in self.data_src_dir.glob('*.wav')]
        self.num_workers = num_workers
        self.prefetch = prefetch
        
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    
    def normalize_src_tgt(self, src, tgt, eps=1e-8):
        norm_factor = np.max(np.abs(src)) + eps
        src = src / norm_factor
        tgt = tgt / norm_factor
        return src, tgt

    def load_wav(self, path):
        wav, sr = sf.read(path, dtype='float32')
        assert sr == 16000 and wav.ndim == 1
        return wav
    
    def process_one_sample(self, name):
        if self.is_train:
            src = self.load_wav(self.data_src_dir / (name + '.wav'))
            tgt = self.load_wav(self.data_tgt_dir / (name + '.wav'))
            length = src.shape[-1]
            if length < self.cut_len:
                src = np.pad(src, (0, self.cut_len - length), mode='wrap')
                tgt = np.pad(tgt, (0, self.cut_len - length), mode='wrap')
            else:
                # randomly cut segment
                wav_start = random.randint(0, length - self.cut_len)
                src = src[wav_start: wav_start + self.cut_len]
                tgt = tgt[wav_start: wav_start + self.cut_len]
            src, tgt = self.normalize_src_tgt(src, tgt)
            length = self.cut_len
        else:
            assert self.batch_size == 1
            src = self.load_wav(self.data_src_dir / (name + '.wav'))
            tgt = self.load_wav(self.data_tgt_dir / (name + '.wav'))
            src, tgt = self.normalize_src_tgt(src, tgt)
            length = src.shape[-1]
        return src, tgt, length, name

    def data_iter_fn(self, q, event):
        wav_names = deepcopy(self.wav_names)
        if self.is_train:  # training
            random.shuffle(wav_names)
            use_sample = len(wav_names) // (self.world_size * self.batch_size) * (self.world_size * self.batch_size)
            wav_names = wav_names[:int(use_sample)]
        else:  # testing
            assert self.batch_size == 1
        
        executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for sample_idx in range(self.rank * self.batch_size, len(wav_names), self.world_size * self.batch_size):
            
            batch_src = []
            batch_tgt = []
            lengths = []
            for result in executor.map(self.process_one_sample, wav_names[sample_idx:sample_idx + self.batch_size]):
                src, tgt, length, name = result
                batch_src.append(src)
                batch_tgt.append(tgt)
                lengths.append(length)
            batch_src = np.stack(batch_src, axis=0)
            batch_tgt = np.stack(batch_tgt, axis=0)
            q.put((torch.from_numpy(batch_src), torch.from_numpy(batch_tgt), torch.LongTensor(lengths), wav_names[sample_idx:sample_idx + self.batch_size]))
        event.set()

    def __iter__(self):
        q = queue.Queue(maxsize=self.prefetch + 1)
        event = threading.Event()
        worker = threading.Thread(target=self.data_iter_fn, args=(q, event))
        worker.start()
        while not event.is_set() or not q.empty():
            try:
                yield q.get(timeout=1.0)
            except queue.Empty:
                continue

    def __len__(self):
        """
        :return: number of batches in dataset
        """
        num_batches = int(len(self.wav_names) // (self.world_size * self.batch_size))
        if self.is_train:
            return num_batches
        else:
            if self.rank < len(self.wav_names) // self.batch_size - num_batches * self.world_size:
                return num_batches + 1
            else:
                return num_batches


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_src_dir,
        train_tgt_dir,
        val_src_dir,
        val_tgt_dir,
        test_src_dir,
        test_tgt_dir,
        batch_size, 
        cut_len, 
        num_workers,
    ):
        super().__init__()
        self.train_src_dir = train_src_dir
        self.train_tgt_dir = train_tgt_dir
        self.val_src_dir = val_src_dir
        self.val_tgt_dir = val_tgt_dir
        self.test_src_dir = test_src_dir
        self.test_tgt_dir = test_tgt_dir

        self.batch_size = batch_size
        self.cut_len = cut_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_iter = DataLoadIter(self.train_src_dir, self.train_tgt_dir, is_train=True, 
                                           batch_size=self.batch_size, cut_len=self.cut_len, num_workers=self.num_workers)
            self.val_iter = DataLoadIter(self.val_src_dir, self.val_tgt_dir, is_train=False, batch_size=1, cut_len=self.cut_len)
        if stage == 'test' or stage is None:
            self.test_iter = DataLoadIter(self.test_src_dir, self.test_tgt_dir, is_train=False, batch_size=1, cut_len=self.cut_len)

    def train_dataloader(self):
        return self.train_iter

    def val_dataloader(self):
        return self.val_iter

    def test_dataloader(self):
        return self.test_iter

