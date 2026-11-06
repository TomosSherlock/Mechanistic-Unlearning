import copy
import torch
from utils import evaluate, display_progress, report

# adapted from https://github.com/Happy2Git/Unlearning_Inference_IAM/blob/13f385b62e7cbc85c8c59d0912839e756db51cde/unlearning_lib/methods/fisher_forgetting.py

def finetune_epoch(retain_loader, forget_loader, test_retain_loader, test_forget_loader, net, optimizer, scheduler, criterion, epoch, run, args, DEVICE: str = 'cpu', epochs: int = -1):
    if epochs == -1:
        epochs = epoch + 1

    for inputs, targets in retain_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    scheduler.step()

    forgotten_accuracy = evaluate(net, None, None, forget_loader, args)
    report([net, None], None, test_retain_loader, test_forget_loader, epoch, run, args, forgotten_accuracy = forgotten_accuracy)
    display_progress("Finetuning", epoch, epochs)

    return forgotten_accuracy

def finetune(net_input, retain_loader, forget_loader, test_retain_loader, test_forget_loader, run, args,
            DEVICE: str = 'cpu', epochs: int = 5, lr: float = 0.01, target_unlearned_performance: float = 999):
    """Unlearning by fine-tuning. (unfederated)

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
      net : nn.Module.
        pre-trained model to use as base of unlearning.
      retain_loader : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget_loader : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget.
      test_retain_loader : torch.utils.data.DataLoader.
        Dataset loader for access to the test set corresponding to the retained class.
      test_forget_loader : torch.utils.data.DataLoader.
        Dataset loader for access to the test set corresponding to the forgotten class.
      run : wandb.run.
        WandB run object for logging.
      args : argparse.Namespace.
        Arguments for the unlearning process.
      epochs : int.
        Number of epochs to train the model.
      lr : float.
        Learning rate for the optimizer.
      DEVICE : str.
        Device to use for training.
      target_unlearned_performance : float.
        Target performance on the forgotten samples to unlearn until. 
        (if unspecified or set to 999, unlearning will occur for the specified number of epochs instead.)
    Returns:
      net : updated model
    """
    net = copy.deepcopy(net_input)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    net.train()

    epoch = -1 
    report([net, None], None, test_retain_loader, test_forget_loader, epoch, run, args)

    if (target_unlearned_performance == 999):
        for epoch in range(epochs):
            finetune_epoch(retain_loader, forget_loader, test_retain_loader, test_forget_loader, net, optimizer, scheduler, criterion, epoch, run, args, DEVICE=DEVICE, epochs=epochs)
    else:
        forgotten_sample_performance = 100
        epoch = 0
        while (forgotten_sample_performance > target_unlearned_performance):
            forgotten_sample_performance = finetune_epoch(retain_loader, forget_loader, test_retain_loader, test_forget_loader, net, optimizer, scheduler, criterion, epoch, run, args, DEVICE=DEVICE, epochs=epochs)
            epoch += 1


    net.eval()
    return net