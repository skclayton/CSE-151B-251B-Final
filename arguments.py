import argparse
import os

def params():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--task", default="prediction", type=str,\
                help="prediction is image age prediction;\n\
                      segmentation is image semantic segmentation")
    parser.add_argument("--seed", default=42, type=int,
                help="Seed for random number generator.")
    parser.add_argument("--n-classes", default=100, type=int,
                help="Number of classes for classifier.")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=64, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=(768+60)//2, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.4, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=1, type=int,
                help="Total number of training epochs to perform.")

    args = parser.parse_args()
    return args
