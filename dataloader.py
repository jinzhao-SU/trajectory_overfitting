from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys


class UAVDatasetTuple(Dataset):
    def __init__(self, task_path, label_path):
        self.task_path = task_path
        self.label_path = label_path
        self.label_md = []
        self.task_md = []
        self._get_tuple()

    def __len__(self):
        return len(self.label_md)

    def _get_tuple(self):
        self.task_md = np.load(self.task_path).astype(float)
        self.label_md = np.load(self.label_path).astype(float)
        assert len(self.task_md) == len(self.label_md), "not identical"

    def __getitem__(self, idx):
        sample = {}
        try:
            task = self._prepare_task(idx)
            label = self._get_label(idx)
        except Exception as e:
            print('error encountered while loading {}'.format(idx))
            print("Unexpected error:", sys.exc_info()[0])
            print(e)
            raise

        sample = {'task': task, 'label': label}

        return sample

    def _prepare_task(self, idx):
        task_md = self.task_md[idx]
        return task_md

    def _get_label(self, idx):
        label_md = self.label_md[idx]
        return label_md
