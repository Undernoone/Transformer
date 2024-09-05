from model import Transformer
from torch.utils.data import DataLoader
from data import MyDataset
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import torch

model = Transformer().cuda()
dataset = MyDataset("source.txt", "target.txt")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

lossFunction = nn.CrossEntropyLoss(ignore_index=2)
optimizer = AdamW(params=model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-9)

for epoch in range(200):
    model.train()
    epoch_loss = 0
    t = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for input_id, input_m, output_id, output_m in t:
        input_id, input_m = input_id.cuda(), input_m.cuda()
        output_id, output_m = output_id.cuda(), output_m.cuda()

        output = model(input_id, input_m, output_id[:, :-1], output_m[:, :-1])

        target = output_id[:, 1:]

        output_reshaped = output.reshape(-1, output.size(-1))
        target_reshaped = target.reshape(-1)
        loss = lossFunction(output_reshaped, target_reshaped)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        epoch_loss += loss.item()
        t.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
