from torch import nn
from transformers import AutoImageProcessor, NatForImageClassification

class PredictionModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.encoding = NatForImageClassification.from_pretrained("shi-labs/nat-mini-in1k-224")
        self.classifier = Classifier(args=args)
 
    def forward(self, inputs):
        outputs = self.encoding(**inputs)
        return self.classifier(outputs)
    
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

