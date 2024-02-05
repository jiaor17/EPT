import numpy as np
import torch
from tqdm import tqdm
import pdb

import torch.distributed as dist

class MixDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, *datasets) -> None:
        super().__init__()
        self.datasets = datasets
        self.cum_len = []
        self.total_len = 0
        for dataset in datasets:
            self.total_len += len(dataset)
            self.cum_len.append(self.total_len)
        self.collate_fn = self.datasets[0].collate_fn

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        last_cum_len = 0
        for i, cum_len in enumerate(self.cum_len):
            if idx < cum_len:
                return self.datasets[i].__getitem__(idx - last_cum_len)
            last_cum_len = cum_len
        return None
    
class PretrainDatasetWrapper(MixDatasetWrapper):
    def __init__(self, max_n_vertex, *datasets) -> None:
        super().__init__(*datasets)
        self.max_n_vertex = max_n_vertex
        self._lengths = []
        for dataset in datasets:
            cur_approx_lengths = [int(x[0]) * dataset.approx_length for x in dataset._properties]
            cur_preserve = (np.array(cur_approx_lengths) <= max_n_vertex).sum()
            cur_save_ratio = cur_preserve / len(cur_approx_lengths)
            print(f"Dataset: {dataset.name}\t| Total: {len(cur_approx_lengths)}\t| Used: {cur_preserve}\t| Ratio: {cur_save_ratio * 100:.2f}% ")
            self._lengths.extend(cur_approx_lengths)


        self._lengths = np.array(self._lengths)
        self._valid_indices = np.where(self._lengths <= max_n_vertex)[0]
        self._lengths = self._lengths[self._valid_indices]
        self.valid_len = len(self._lengths)

    def __len__(self):
        return self.valid_len

    def __getitem__(self, idx):
        real_idx = self._valid_indices[idx]
        return super(PretrainDatasetWrapper, self).__getitem__(int(real_idx))

class DynamicBatchWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, max_n_vertex_per_batch) -> None:
        super().__init__()
        self.dataset = dataset
        self.indexes = [i for i in range(len(dataset))]
        self.max_n_vertex_per_batch = max_n_vertex_per_batch
        self.total_size = None
        self.batch_indexes = []
        self._form_batch()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.dataset, attr):
            return getattr(self.dataset, attr)
        else:
            raise AttributeError(f"'DynamicBatchWrapper'(or '{type(self.dataset)}') object has no attribute '{attr}'")

    ########## overload with your criterion ##########
    def _form_batch(self):

        np.random.shuffle(self.indexes)
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        cur_vertex_cnt = 0
        batch = []

        for i in tqdm(self.indexes):
            if hasattr(self.dataset, '_lengths'):
                item_len = self.dataset._lengths[i]
            else:
                data = self.dataset[i]
                item_len = len(data['B']) if 'B' in data else data['len']
            if item_len > self.max_n_vertex_per_batch:
                # self.batch_indexes.append([i])
                continue
            cur_vertex_cnt += item_len
            if cur_vertex_cnt > self.max_n_vertex_per_batch:
                self.batch_indexes.append(batch)
                batch = []
                cur_vertex_cnt = item_len
            batch.append(i)
        self.batch_indexes.append(batch)

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        else:
            # control the lengths of the dataset, otherwise the dataloader will raise error
            if len(self.batch_indexes) < self.total_size:
                num_add = self.total_size - len(self.batch_indexes)
                self.batch_indexes = self.batch_indexes + last_batch_indexes[:num_add]
            else:
                self.batch_indexes = self.batch_indexes[:self.total_size]

    def __len__(self):
        return len(self.batch_indexes)
    
    def __getitem__(self, idx):
        return [self.dataset[i] for i in self.batch_indexes[idx]]
    
    def collate_fn(self, batched_batch):
        batch = []
        for minibatch in batched_batch:
            batch.extend(minibatch)
        return self.dataset.collate_fn(batch)