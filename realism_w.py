import argparse
import os
from realism_fuction import realism

Winv_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'truncation_0_7')
W_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images1024x1024')


def parse_args():
    parser = argparse.ArgumentParser(description="calcualte realism metric using W rather feature vector")
    # parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size', type=int, default=100)

    parser.add_argument('--Winv_dir', default=Winv_DIRECTORY)
    parser.add_argument('--W_dir', default=W_DIRECTORY)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    realism = realism(args)
    realism.run()
    print("finished")
