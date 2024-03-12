from torch import nn

class PredictionModel(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.drop_rate = args.drop_rate
    self.dropout = nn.Dropout(p=self.drop_rate)
    self.classifier = Classifier(args=args)
 
  def forward(self, inputs):
    pass
  
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

