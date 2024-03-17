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
    
    parser.add_argument("--embed-dim", default=1000, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--hidden-dim", default=325, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.4, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=5, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='number of label classes (Model default if None)')
    parser.add_argument('--embed_dim', type=int, default=150, metavar='N',
                        help='number of label classes (Model default if None)')
    #Optimizer 
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    #scheduler
    parser.add_argument('--warmup-lr', type=float, default=0.000001, metavar='LR',
                        help='warmup learning rate (default: 0.000001)')
    parser.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (5e-6)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    

    args = parser.parse_args()
    return args
