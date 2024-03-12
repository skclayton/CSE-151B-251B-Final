import torch
from torch import nn
from tqdm import tqdm as progress_bar
from arguments import params
from utils import setup_gpus, set_seed
from modelPrediction import PredictionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_prediction(args, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    
    for epoch_count in range(args.n_epochs):
        loss = 0
        model.train()

        for:
            inputs, labels = 
            logits = model(inputs)
            loss = criterion(logits,labels)
            loss.backward()

            optimizer.step()
            model.zero_grad()
            losses += loss.item()

        scheduler.step()
        print('epoch', epoch_count, 'loss:', loss)
        
        
if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  set_seed(args)
  
  if args.task == 'prediction':
    model = PredictionModel(args).to(device)
    train_prediction(args, model)