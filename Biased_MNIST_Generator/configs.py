import argparse

def config():
    parser = argparse.ArgumentParser(description="Biased-MNIST Generator")

    parser.add_argument('--images-path', default="data/raw/train-images.idx3-ubyte", 
                        type=str, help='path to MNIST train images file')
    parser.add_argument('--labels-path', default="data/raw/train-labels.idx1-ubyte", 
                            type=str, help='path to MNIST train labels file')
    parser.add_argument('--data-output-path', default='data/processed/',
                        type=str, help='folder to store Biased-MNIST idx files')
    parser.add_argument('--sample-path', default='samples/', type=str,
                        help='folder to store preview images')
    parser.add_argument('--samples', type=int, default=20, 
                        help='number of preview images to generate')
    parser.add_argument('--rho', default=1, type=float,
                        help='probability of biased classes to have a coloured background')
    parser.add_argument('--n-biased-classes', default=1, type=int,
                        help='number of classes to make biased')

    return parser.parse_args()
