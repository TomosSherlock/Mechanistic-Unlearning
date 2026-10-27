import argparse

def config():
    parser = argparse.ArgumentParser(description="Vertical federated Unlearning Experiments")

    parser.add_argument('--print_decimal_digits', default=6, type=int,
                        help='How many decimal places print out in logger.')
    parser.add_argument('--verbose', default=1, type=int,
                        help='Whether to print verbose logging info')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_gpu', default=True,
                        help='Whether to use GPU or not')
    ################################### Experiment ###############################################
    parser.add_argument('--outdir', default='exp', type=str,
                        help='output directory')
    parser.add_argument('--expname', default='---', type=str,
                        help='detailed exp name to distinguish different sub-exp')
    parser.add_argument('--evals', default=20, type=int,
                        help='Number of times to evaluate each model')
    # parser.add_argument('--expname_tag', default='', type=str,
    #                     help=' detailed exp tag to distinguish different sub-exp with the same expname')

    ##################################### Dataset #################################################
    parser.add_argument('--data', default='custom', type=str,
                        help='name of dataset',
                        choices=['cifar10', 'mnist', 'cifar100', 'modelnet', 'yahoo', 'mri', 'ct', 'custom', 'custom-1'])
    parser.add_argument('--data_path', default='data', type=str,
                        help='path of training data')
    parser.add_argument('--test_path', default='data', type=str,
                        help='path of testing data')
    parser.add_argument('--batch_size', default=32, type=int,
                        help = 'Number of data per batch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_classes', default=10, type=int,
                        help = 'Number of classes in data')
    parser.add_argument('--test_prefix', default='t10k', type=str,
                        help='file name prefix for the test data when using a custom dataset')
    parser.add_argument('--train_prefix', default='train', type=str,
                        help='file name prefix for the train data when using a custom dataset')
    parser.add_argument('--data_suffix', default='-images.idx3-ubyte', type=str,
                        help='file name suffix for the x datas when using a custom dataset')
    parser.add_argument('--label_suffix', default='-labels.idx1-ubyte', type=str,
                        help='file name suffix for the y labels when using a custom dataset')

    #################################### Training ################################################
    parser.add_argument('--epochs', default=100, type=int,
                        help = 'Number of epochs in training')
    parser.add_argument('--mode', default='full', type=str,
                        help = 'Train with full dataset or retrain with remain dataset',
                        choices=['full', 'retrain'])
    parser.add_argument('--std', default=0.1, type=float)
    parser.add_argument('--percent', default=0.999, type=float)
    parser.add_argument('--vfl', default=2, type=int,
                        help = 'Number of active parties in vfl (set to 0 for non-federated learning)')

    ##################################### Unlearn #################################################
    parser.add_argument('--unlearn_method', default='LUV', type=str,
                        help='Unlearning method')
    parser.add_argument('--unlearn_client_method', default='ours', type=str,
                        help='Unlearning client method')
    parser.add_argument('--unlearn_class', default=0, type=int,
                        help='Unlearning class')
    parser.add_argument('--unlearn_class_num', default=1, type=int,
                        help='Number of unlearn class, eg. 1 class, 2 classes, 4 classes unlearning')
    parser.add_argument('--unlearn_lr', default=0.0000002, type=float,
                        help='Learning rate in unlearning')
    parser.add_argument('--unlearn_samples', default=40, type=int,
                        help='Samples of data use in EE unlearning')
    parser.add_argument('--unlearn_epochs', default=16, type=int,
                        help='Unlearning epochs')
    parser.add_argument('--unlearn_bias', default=0, type=int,
                        help="Whether or not to unlearn the biased samples instead of a class")
    parser.add_argument('--u_batch_size', default=32, type=int,
                        help='Batch size for any unlearning method')
    parser.add_argument('--f_method', default='EWC', type=str,
                        help='Method for more complex unlearning methods')
    parser.add_argument('--alpha', default=1, type=float,
                        help='Alpha value for manifold mixup')
    parser.add_argument('--target_unlearned_performance', default=999, type=float,
                        help='Percent unlearned sample performance to stop unlearning at. ' \
                        'If unspecified, unlearning will occur for the specified number of epochs instead.')
    parser.add_argument('--alpha_momentum', default=1.5, type=float,
                        help='Factor to multiply manifold mixup alpha by after each unlearning epoch')
    parser.add_argument('--quick', type=bool, default=False)

    ###################################### Model ##################################################
    parser.add_argument('--model_type', default='resnet18', type=str,
                        help='Type of model used')
    parser.add_argument('--optimizer_lr', default=1e-3, type=float,
                        help='Type of model used')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--model_path', default='exp/', type=str,
                            help='Path of trained model to unlearn')

    return parser.parse_args()
