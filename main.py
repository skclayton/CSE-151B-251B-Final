import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from arguments import params  # Ensure this matches the name of the file and function for your parameters.
from utils import setup_gpus, set_seed  # Adjust according to your utilities for setting up GPUs and seeds.
import loadData  # Adjust to your module for loading data.
from modelPrediction import ConvNextClassification
from transformers import AutoImageProcessor
from modelPrediction import ConvNextClassification
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def train_prediction(args, model, img_processor, train_loader, device):
    print('Training...')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    model.train()
    for epoch in range(args.n_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{args.n_epochs}'):
            inputs, labels = batch
            inputs = img_processor(inputs, return_tensors="pt")['pixel_values'].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch}: Loss {total_loss/len(train_loader)}')
        scheduler.step()

def run_eval(args, model, img_processor, test_loader, device):
    print('Evaluating...')
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss, total_correct = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluation'):
            inputs = img_processor(inputs, return_tensors="pt")['pixel_values'].to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / len(test_loader.dataset)
    print(f'Evaluation - Loss: {avg_loss}, Accuracy: {accuracy}')
    return accuracy, avg_loss

if __name__ == "__main__":
    args = params()
    setup_gpus(args)
    set_seed(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = loadData.Dataset(args, 'train')  # Adjust loadData.Dataset to match your data loading approach.
    test_dataset = loadData.Dataset(args, 'test')   # Adjust loadData.Dataset accordingly.
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
    model = ConvNextClassification(1000, "facebook/convnext-tiny-224").to(device)

    train_prediction(args, model, image_processor, train_loader, device)
    test_acc, test_loss = run_eval(args, model, image_processor, test_loader, device)
    print(f'Final Test Accuracy: {test_acc}, Final Test Loss: {test_loss}')
