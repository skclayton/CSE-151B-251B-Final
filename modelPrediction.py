import torch
from torch import nn
from transformers import ConvNextModel

class ConvNextClassification(nn.Module):
    def __init__(self, num_labels, pretrained_model="facebook/convnext-tiny-224"):
        super().__init__()
        # Initialize the ConvNext model
        self.convnext = ConvNextModel.from_pretrained(pretrained_model)
        # Create a classifier layer
        # Note: Adjust the input features of nn.Linear based on the output features of your ConvNext model variant
        self.classifier = nn.Linear(37632, num_labels)


    def forward(self, pixel_values):
        outputs = self.convnext(pixel_values)
        # Flatten the output for the classifier:
        x = torch.flatten(outputs.last_hidden_state, start_dim=1)  # This flattens all dimensions except the batch
        # Ensure the flattened size matches the linear layer's input expectation
        logits = self.classifier(x)
        return logits
