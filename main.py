import torch
from torch import nn
from tqdm import tqdm as progress_bar
from arguments import params
from utils import setup_gpus, set_seed
from modelPrediction import PredictionModel
from torch.utils.data import DataLoader
import loadData
from transformers import AutoImageProcessor, NatConfig, NatModel, NatForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_prediction(args, model, img_processor):
    print('training')
    criterion = nn.KLDivLoss(reduction='batchmean')  # Changed to KLDivLoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    num_classes =1000
    radius = 16
    radius_decay = 0.707107
    lst = []
    losses = []
    acc = []
    for epoch_count in range(args.n_epochs):
        print("epoch", epoch_count)
        total_loss = 0
        model.train()

        for iter, (inputs, labels) in progress_bar(enumerate(train_loader), total=len(train_loader)):
            model.zero_grad()
            inputs = img_processor(inputs, return_tensors="pt")['pixel_values'].to(device)
            labels = labels.to(device).long()

            logits = model(inputs).logits

            # Apply top-n label softening
            soft_labels = nn.functional.one_hot(labels, num_classes=num_classes)
            for label in soft_labels:
                idx = torch.argmax(label)
                for i in range(idx - int(radius), idx + int(radius) + 1):
                    if 0 <= i < num_classes:
                        label[i] = 1
            soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)  # Normalize

            log_softmax_logits = nn.functional.log_softmax(logits, dim=1)
            loss = criterion(log_softmax_logits, soft_labels.float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        radius *= radius_decay  # Update the radius for top-n loss decay

        val_acc, val_loss = run_eval(args, model, img_processor, test_loader)
        lst.append(val_loss)
        acc.append(val_acc)
        print(f'epoch {epoch_count}, loss: {total_loss / len(train_loader)}, val_loss: {val_loss}, val_acc: {val_acc}')

    return lst, acc

def run_eval(args, model, img_processor, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_acc, num_samples = 0, 0, 0
    for step, (inputs, labels) in progress_bar(enumerate(loader), total=len(loader)):
        inputs = img_processor(inputs, return_tensors="pt")['pixel_values'].to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(inputs).logits

        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        total_acc += (preds == labels).sum().item()
        total_loss += loss.item()
        num_samples += labels.size(0)

    return total_acc / num_samples, total_loss / len(loader)

if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    set_seed(args)

    train_dataset = loadData.Dataset(args, 'train')
    test_dataset = loadData.Dataset(args, 'test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.task == 'prediction':
        model = PredictionModel(args).to(device)
        image_processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
        model = NatForImageClassification.from_pretrained("shi-labs/nat-mini-in1k-224").to(device)

        lst, acc = train_prediction(args, model, image_processor)
        test_acc, test_loss = run_eval(args, model, image_processor, test_loader)
        print(f'Test accuracy: {test_acc}')
