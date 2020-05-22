import argparse
import os

from . import cnn
from . import feedforward
from . import svm

def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=16,
        help='number of times to go through the data, default=20')
    
    parser.add_argument(
        '--batch-size',
        default=100,
        type=int,
        help='number of records to read during each training step, default=100')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    
    parser.add_argument(
        '--nw-type',
        default='feedforward',
        type=str,
        help='type of model to run'
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # TODO: Switch based on --nw-type arg

    model_to_use = args.nw_type
    model_to_use.train(args.num_epochs, args.batch_size, args.learning_rate, args.job_dir)
