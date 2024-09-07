from torch.utils.data import DataLoader
from data import TestDataset  # 你的测试数据集
from model import Transformer
import torch
import numpy as np
from sklearn.metrics import accuracy_score

# Initialize model and load saved weights
model = Transformer().cuda()
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set model to evaluation mode

# Prepare test data
test_dataset = TestDataset("test_source.txt", "test_target.txt")  # 假设你有一个测试数据集
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Lists to store predictions and true labels
all_predictions = []
all_targets = []

# Evaluate model
with torch.no_grad():
    for input_id, input_m, output_id, output_m in test_dataloader:
        # Perform inference
        output = model(input_id.cuda(), input_m.cuda(), output_id[:, :-1].cuda(), output_m[:, :-1].cuda())
        predictions = output.argmax(dim=-1)  # Get predicted class

        # Append results
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(output_id[:, 1:].cpu().numpy())

# Flatten lists and convert to numpy arrays
all_predictions = np.concatenate(all_predictions)
all_targets = np.concatenate(all_targets)

# Calculate accuracy
accuracy = accuracy_score(all_targets.flatten(), all_predictions.flatten())
print(f'Test Accuracy: {accuracy:.4f}')
