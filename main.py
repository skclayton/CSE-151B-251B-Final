import torch
from torch import nn
from tqdm import tqdm as progress_bar
from arguments import params
from utils import setup_gpus, set_seed
from modelPrediction import PredictionModel
from torch.utils.data import DataLoader
import loadData
from transformers import AutoImageProcessor, NatConfig,NatModel,NatForImageClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_prediction(args, model,img_processor):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    
    lst = []
    acc = []
    for epoch_count in range(args.n_epochs):
        loss = 0
        
        model.train()

        for iter, (inputs, labels) in enumerate(train_loader):
            model.zero_grad()
            inputs = img_processor(inputs,return_tensors="pt").to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            losses += loss.item()

            optimizer.step()

        scheduler.step()
        
        val_acc,val_loss = run_eval(args,model,img_processor)
        lst += [val_loss]
        acc += [val_acc]
        print('epoch', epoch_count, 'loss:', loss)
    
    return lst,acc
        
def run_eval(args, model,img_processor):
    model.eval()
    criterion = nn.CrossEntropyLoss()
        
    losses ,acc = 0
    for step, (inputs,labels) in progress_bar(enumerate(train_loader)):
        inputs = img_processor(inputs,return_tensors="pt").to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        loss = criterion(logits,labels)
        tem = (logits.argmax(1)==labels).float().sum()
        acc += tem.item()
        losses += loss.item()
    
    return acc/len(train_loader), losses/len(train_loader)
            
        
if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    set_seed(args)
    
    train_dataset = loadData.Dataset(args, 'train')
    test_dataset = loadData.Dataset(args, 'test')
    

    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.task == 'prediction':
        #model = PredictionModel(args).to(device)
        
        # baseline NAT model
        image_processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
        model = NatForImageClassification.from_pretrained("shi-labs/nat-mini-in1k-224")
        train_prediction(args, model)