from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import albumentations as A

def train(
    model: "torch.nn.Module",
    train_data_loader: "torch.utils.data.DataLoader",
    criterion: "torch.nn.functional.nll_loss",  # 'torch.nn.NLLLoss'
    optimizer: "torch.optim.Optimizer",
    epoch: int,
    train_losses: list,
    train_acc: list,
    device: "torch.device" = torch.device("cpu"),
):
    model.to(device)
    model.train()
    pbar = tqdm(train_data_loader)
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data['image'].to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        # loss = F.nll_loss(y_pred, target)
        loss = criterion(y_pred, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Epoch={epoch} Batch_id={batch_idx} Loss={loss.item()} Accuracy={100*correct/processed:0.2f}"
        )
        train_acc.append(100 * correct / processed)


def test(
    model: "torch.nn.Module",
    test_data_loader: "torch.utils.data.DataLoader",
    epoch: int,
    test_losses: list,
    test_acc: list,
    device: "torch.device" = torch.device("cpu"),
):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_data_loader:
            data, target = data['image'].to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_data_loader.dataset)
    test_losses.append(test_loss)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_data_loader.dataset),
            100.0 * correct / len(test_data_loader.dataset),
        )
    )

    test_acc.append(100.0 * correct / len(test_data_loader.dataset))

# ref: https://github.com/albumentations-team/albumentations/issues/879#issuecomment-824771225
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))
