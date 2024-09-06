from model import Transformer
from torch.utils.data import DataLoader
from data import MyDataset
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# Initialize model, dataset, dataloader, loss function, and optimizer
my_model = Transformer().cuda()
dataset = MyDataset("source.txt", "target.txt")
dataloader = DataLoader(dataset, 32, shuffle=True)
loss_func = nn.CrossEntropyLoss(ignore_index=2)
trainer = AdamW(params=my_model.parameters(), lr=0.0005)

# Lists to store epoch and loss values
epochs = []
losses = []

# Training loop
for epoch in range(100):
    t = tqdm(dataloader)
    epoch_loss = 0
    for input_id, input_m, output_id, output_m in t:
        output = my_model(input_id.cuda(), input_m.cuda(), output_id[:, :-1].cuda(), output_m[:, :-1].cuda())
        target = output_id[:, 1:].cuda()
        loss = loss_func(output.reshape(-1, 29), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1)
        trainer.step()
        trainer.zero_grad()

        # Accumulate loss
        epoch_loss += loss.item()

        # Update tqdm description
        t.set_description(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

    # Average loss for the epoch
    avg_loss = epoch_loss / len(dataloader)
    epochs.append(epoch + 1)
    losses.append(avg_loss)

# Save model state
torch.save(my_model.state_dict(), "model.pth")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()
