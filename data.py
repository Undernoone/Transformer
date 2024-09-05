import torch
from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np

vocab_list = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n',
             'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']
char_list = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n',
             'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']
bos_token = "[BOS]"
eos_token = "[EOS]"
pad_token = "[PAD]"

def process_date(source, target):
    maxLength = 12
    if len(source) > maxLength: # generateData有13的
        source = source[:maxLength]
    if len(target) > maxLength-1: # 一会看看为什么-1
        target = target[:maxLength-1]
    # 转为编号
    source_id = [vocab_list.index(p) for p in source]
    target_id = [vocab_list.index(p) for p in target]
    target_id = [vocab_list.index("[BOS]")] + target_id + [vocab_list.index("[EOS]")]
    source_m = np.array([1] * maxLength)
    target_m = np.array([1] * (maxLength + 1))
    if len(source_id) < maxLength:
        pad_len = maxLength - len(source_id)
        source_id += [vocab_list.index("[PAD]")] * pad_len
        source_m[-pad_len:] = 0
        # source_id += [vocab_list.index(pad_token)] * (maxLength - len(source_id))
    if len(target_id) < maxLength + 1:
        pad_len = maxLength - len(target_id) + 1
        target_id += [vocab_list.index("[PAD]")] * pad_len
        target_m[-pad_len:] = 0
        # target_id += [vocab_list.index(pad_token)] * (maxLength - len(target_id) + 1)
    return source_id, source_m, target_id, target_m

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
        source_id, source_m, target_id, target_m = process_date(self.sourceList[index], self.targetList[index])
        return (torch.tensor(source_id, dtype=torch.long), torch.tensor(source_m, dtype=torch.long),
                torch.tensor(target_id, dtype=torch.long), torch.tensor(target_m, dtype=torch.long))

# 生成的数据是3-13之间，但是Transformermodel的输入长度是12，所以需要对数据进行处理

if __name__ == '__main__':
    test_data = MyDataset("source.txt", "target.txt")
    source_id, source_m, target_id, target_m = test_data[2]
    pass


