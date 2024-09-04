import torch
from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

vocab_list = ["[BOS]", "[EOS]", "createTextNode", 'a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n',
             'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']
char_list = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n',
             'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']
bos_token = "[BOS]"
eos_token = "[EOS]"
pad_token = "[PAD]"

def process_date(source, target):
    max_length = 12
    if len(source) > max_length:
        source = source[:max_length]
    if len(target) > max_length-1:
        target = target[:max_length-1]
    source_id = [vocab_list.index(p) for p in source]
    target_id = [vocab_list.index(p) for p in target]
    target_id = [vocab_list.index("[BOS]")] + target_id + [vocab_list.index("[EOS]")]
    source_m = np.array([1] * max_length)
    target_m = np.array([1] * (max_length + 1))
    if len(source_id) < max_length:
        pad_len = max_length - len(source_id)
        source_id += [vocab_list.index("[PAD]")] * pad_len
        source_m[-pad_len:] = 0
    if len(target_id) < max_length + 1:
        pad_len = max_length - len(target_id) + 1
        target_id += [vocab_list.index("[PAD]")] * pad_len
        target_m[-pad_len:] = 0
    return source_id, target_id

class MyDataset(Dataset):
    def __init__(self, source_path, target_path) -> None:
        super(MyDataset, self).__init__()
        self.sourceList = []
        self.targetList = []
        with open(source_path) as f:
            content = f.readlines()
            for i in content:
                self.sourceList.append(deepcopy(i.strip())) # deepcopy to avoid modifying the original list 去掉换行符

        with open(target_path) as f:
            content = f.readlines()
            for i in content:
                self.targetList.append(deepcopy(i.strip()))

    def __len__(self):
        return len(self.sourceList)

    def __getitem__(self, index):
        return self.sourceList[index], self.targetList[index] # 输入和输出必须相等

if __name__ == '__main__':
    test_data = MyDataset("source.txt", "target.txt")