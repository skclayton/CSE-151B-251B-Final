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
    print('training')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    
    radius = 16
    radius_decay = 0.707107
    
    lst = []
    losses =[]
    acc = []
    for epoch_count in range(args.n_epochs):
        print("epoch",epoch_count)
        loss = 0
        
        model.train()

        for iter, (inputs, labels) in progress_bar(enumerate(train_loader),total=len(train_loader)):
            model.zero_grad()
            inputs = img_processor(inputs,return_tensors="pt")['pixel_values'].to(device)
            labels = labels.to(device)
            
            logits = model(inputs).logits
            
            labels = labels.long()
            labels = nn.functional.one_hot(labels, num_classes=10)
            for label in labels:
                idx = torch.argmax(label)
                for i in range(idx - int(radius), idx + int(radius) + 1):
                    if i >= 0 and i < len(label):
                        label[i] = 1

            labels = labels / labels.sum(dim = 1, keepdim=True)
            
            loss = criterion(logits, labels)
            loss.backward()
            losses += [loss.item()]

            optimizer.step()

        scheduler.step()
        radius *= radius_decay
        
        val_acc,val_loss = run_eval(args,model,img_processor,test_loader)
        lst += [val_loss]
        acc += [val_acc]
        print('epoch', epoch_count, 'loss:', loss)
    
    return lst,acc
        
def run_eval(args, model,img_processor,loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
        
    losses ,acc = 0,0
    for step, (inputs,labels) in progress_bar(enumerate(loader),total=len(loader)):
        #print("inputs",inputs.size())
        #print("labels",labels)
        inputs = img_processor(inputs,return_tensors="pt")['pixel_values'].to(device)
        labels = labels.to(device)
        
        #print("inputs 2",inputs['pixel_values'].size())
        
        with torch.no_grad():
            logits = model(inputs).logits
        
        loss = criterion(logits,labels)
        print("logits:", logits, "labels",labels,"preds",logits.argmax(1))
        tem = (logits.argmax(1)==labels).float().sum()
        acc += tem.item()
        losses += loss.item()
    
    #print("loader size",len(loader))
    return acc/len(loader), losses/len(loader)
            
        
if __name__ == "__main__":
    args = params()
    args = setup_gpus(args)
    set_seed(args)
    
    train_dataset = loadData.Dataset(args, 'train')
    test_dataset = loadData.Dataset(args, 'test')
    #print('train_dataset', train_dataset[0])

    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    print(len(train_loader))
    if args.task == 'prediction':
        
        # baseline NAT model
        image_processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
        
        # model = PredictionModel(args).to(device)
        model = NatForImageClassification.from_pretrained("shi-labs/nat-mini-in1k-224").to(device)
        
        train_prediction(args, model,image_processor)
        
    #     # top 1 accuracy
        test_acc,test_loss = run_eval(args,model,image_processor,test_loader)
        print(test_acc)
    #     print("top 1 accuracy ",)