from torch import nn
from transformers import AutoImageProcessor, NatForImageClassification

class PredictionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
        self.encoding = NatForImageClassification.from_pretrained("shi-labs/nat-mini-in1k-224")
        
        self.drop_rate = args.drop_rate
        self.dropout = nn.Dropout(p=self.drop_rate)
        self.classifier = Classifier(args=args)
 
    def forward(self, inputs):
        processed_inputs = self.processor(inputs, return_tensors="pt")
        output = self.encoding(**processed_inputs)
        print(output.shape())
  
class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_dim = args.embed_dim
        self.top = nn.Linear(input_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.bottom = nn.Linear(args.hidden_dim, args.n_classes)

    def forward(self, hidden):
        middle = self.relu(self.top(hidden))
        logit = self.bottom(middle)
        return logit

