from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys


class UAVDatasetTuple(Dataset):
    def __init__(self, task_path, label_path):
        self.tasks_path = task_path
        self.label_path = label_path
        self._get_tuple()
        self.label_md = []
        self.task_md = []

    def __len__(self):
        return len(self.label_md)

    def _get_tuple(self):
        task_collection = np.load(self.task_path)
        label_collection = np.load(self.label_path)
        assert len(task_collection) == len(label_collection), "not identical"

        for idx, _ in enumerate(task_collection):
            self.task_md.append(task_collection[idx].float())
            self.label_md.append(label_collection[idx].float())

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

    def get_class_count(self):
        total = len(self.label_md) * self.label_md[0].shape[0] * self.label_md[0].shape[1]
        label = self.label_md
        positive_class = 0
        for label_md in label:
            positive_class += np.sum(label_md)
        print("The number of positive image pair is:", positive_class)
        print("The number of negative image pair is:", total - positive_class)
        positive_ratio = positive_class / total
        negative_ratio = (total - positive_class) / total

        return positive_ratio, negative_ratio
