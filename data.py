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

# def pad_sequence(seq, max_length, pad_value):
#     pad_len = max_length - len(seq)
#     if pad_len > 0:
#         seq += [pad_value] * pad_len
#     return seq

'''    
string need to transfer to id , because Transformer only accept id as input
add [BOS] and [EOS] token to target from source
if source length is less than max_length, add pad_token to source and target_mask to 0
if target length is less than max_length+1, add pad_token to target and target_mask to 0
'''
def process_data(source, target, max_length=None):
    maxLength = 12
    if len(source) > maxLength: # Because in generateData.py, the length of source from 3-13 was generated.
        source = source[:maxLength]
    if len(target) > maxLength-1: # Need to add 1 for [BOS] token
        target = target[:maxLength-1]
    source_id = [vocab_list.index(p) for p in source]
    target_id = [vocab_list.index(p) for p in target]
    target_id = [vocab_list.index("[BOS]")] + target_id + [vocab_list.index("[EOS]")]
    source_mask = np.array([1] * maxLength)
    target_mask = np.array([1] * (maxLength + 1))
    # source_id = pad_sequence(source_id, max_length, vocab_list.index("[PAD]"))
    # target_id = pad_sequence(target_id, max_length + 1, vocab_list.index("[PAD]"))
    # if len(source_id) < max_length:
    #     source_m[-(max_length - len(source_id)):] = 0
    # if len(target_id) < max_length + 1:
    #     target_m[-(max_length - len(target_id) + 1):] = 0
    if len(source_id) < maxLength:
        pad_len = maxLength - len(source_id)
        source_id += [vocab_list.index("[PAD]")] * pad_len
        source_mask[-pad_len:] = 0
        # source_id += [vocab_list.index(pad_token)] * (maxLength - len(source_id))
    if len(target_id) < maxLength + 1:
        pad_len = maxLength - len(target_id) + 1
        target_id += [vocab_list.index("[PAD]")] * pad_len
        target_mask[-pad_len:] = 0
        # target_id += [vocab_list.index(pad_token)] * (maxLength - len(target_id) + 1)
    return source_id, source_mask, target_id, target_mask

class Dataset(Dataset):
    def __init__(self, source_path, target_path) -> None:
        super(Dataset, self).__init__()
        self.sourceList = []
        self.targetList = []
        with open(source_path) as f:
            content = f.readlines()
            for i in content:
                self.sourceList.append(deepcopy(i.strip())) # deepcopy to avoid modifying the original list,remove line break
        with open(target_path) as f:
            content = f.readlines()
            for i in content:
                self.targetList.append(deepcopy(i.strip()))

    def __len__(self):
        return len(self.sourceList)

    def __getitem__(self, index):
        source_id, source_mask, target_id, target_mask = process_data(self.sourceList[index], self.targetList[index])
        return (torch.tensor(source_id, dtype=torch.long), torch.tensor(source_mask, dtype=torch.long),
                torch.tensor(target_id, dtype=torch.long), torch.tensor(target_mask, dtype=torch.long))

if __name__ == '__main__':
    test_data = Dataset("source.txt", "target.txt")
    source_id, source_mask, target_id, target_mask = test_data[2]
    pass