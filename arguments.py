import argparse

def params():
    parser = argparse.ArgumentParser(description='Training and model parameters.')

    # Task-related arguments
    parser.add_argument("--task", default="prediction", type=str,
                        help="Task type: prediction or segmentation.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Seed for random number generator.")
    parser.add_argument("--n-classes", default=100, type=int,
                        help="Number of classes for the classifier.")

    # Training hyperparameters
    parser.add_argument("--batch-size", default=64, type=int,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--learning-rate", default=1e-4, type=float,
                        help="Initial learning rate.")
    parser.add_argument("--hidden-dim", default=(768+60)//2, type=int,
                        help="Dimension of the hidden layer.")
    parser.add_argument("--drop-rate", default=0.4, type=float,
                        help="Dropout rate for the network.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                        help="Epsilon parameter for the Adam optimizer.")
    parser.add_argument("--n-epochs", default=4, type=int,
                        help="Number of epochs to train for.")

    # Optimizer parameters
    parser.add_argument("--opt", default="adamw", type=str,
                        help="Optimizer type (default: 'adamw').")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Momentum for the optimizer (if applicable).")
    parser.add_argument("--weight-decay", default=0.05, type=float,
                        help="Weight decay (L2 penalty).")

    # Scheduler parameters
    parser.add_argument("--warmup-lr", default=1e-6, type=float,
                        help="Warmup learning rate.")
    parser.add_argument("--min-lr", default=5e-6, type=float,
                        help="Minimum learning rate after decay.")
    parser.add_argument("--epochs", default=300, type=int,
                        help="Total number of epochs for training.")
    parser.add_argument("--decay-epochs", default=100, type=float,
                        help="Number of epochs between each learning rate decay.")
    parser.add_argument("--warmup-epochs", default=20, type=int,
                        help="Number of warmup epochs.")
    parser.add_argument("--decay-rate", default=0.1, type=float,
                        help="Rate at which the learning rate decays.")

    # Parse and return arguments
    args = parser.parse_args()
    return args
