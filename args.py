
import argparse

def get_args():
    '''
    :return: the parameter of the train process
    '''
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_slide', default=5, type=int, help='slide number of the input data')
    parser.add_argument('--output_slide', default=5, type=int, help='slide number of the output data')
    parser.add_argument('--total_step', default=8000, type=int, help='number of iterations in the training process')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gamma', default=0.15, type=float, help='parameter of the kernel function')
    parser.add_argument('--print_every', default=10, type=int, help='number of iteration steps between prints')
    parser.add_argument('--device', default='cpu', type=str, help='device of thr process')
    parser.add_argument('--sample', default=100, type=int, help='the sampling number of each iteration')
    parser.add_argument('--learning_rate', default=0.02, type=float, help='the learning rate of the training process')
    parser.set_defaults(feature=True)
    return parser.parse_args()
