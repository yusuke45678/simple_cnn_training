import argparse


def get_args():
    """generate argparse object

    Returns:
        args: [description]
    """
    parser = argparse.ArgumentParser(description='simple CNN model')

    # dataset
    parser.add_argument('-r', '--root', type=str, default='./downloaded_data',
                        help='root of dataset. default to ./downloaded_data')
    parser.add_argument('-d', '--dataset_name', type=str, default='CIFAR10',
                        help='name of dataset. default to CIFAR10')

    # model
    parser.add_argument('--torch_home', type=str, default='./pretrained_models',
                        help='TORCH_HOME environment variable '
                        'where pre-trained model weights are stored. '
                        'default to ./pretrained_models')
    parser.add_argument('-m', '--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='CNN model; resnet18/50. default to resnet18')
    parser.add_argument('--use_pretrained', dest='use_pretrained',
                        action='store_true',
                        help='use pretrained model weights')
    parser.add_argument('--scratch', dest='use_pretrained',
                        action='store_false',
                        help='do not use pretrained model weights '
                        '(train from scratch)')
    parser.set_defaults(use_pretrained=True)

    # training
    parser.add_argument('-b', '--batch_size', type=int, default=8,
                        help='batch size. default to 8')
    parser.add_argument('-w', '--num_workers', type=int, default=2,
                        help='number of workers. default to 2')
    parser.add_argument('-e', '--num_epochs', type=int, default=25,
                        help='number of epochs. default to 25')
    parser.add_argument('--val_epochs', type=int, default=2,
                        help='validation interval in epochs. default to 2')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['SGD', 'Adam'],
                        help='optimizer. SGD or Adam. default to SGD')
    parser.add_argument('--grad_upd', type=int, default=1,
                        help='step interval to update gradient. default to 1')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='learning rate. default to 0.0001')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of SGD. default to 0.9')
    parser.add_argument('--betas', nargs='+', type=float, default=[0.9, 0.999],
                        help='betas of Adam. default to (0.9, 0.999).'
                        'specify like --betas 0.9 0.999')
    parser.add_argument('--use_scheduler', dest='use_scheduler',
                        action='store_true',
                        help='use scheduler')
    parser.add_argument('--no_scheduler', dest='use_scheduler',
                        action='store_false',
                        help='do not use scheduler')
    parser.set_defaults(use_scheduler=False)

    parser.add_argument('--use_dp', dest='use_dp',
                        action='store_true',
                        help='use multi GPUs with data parallel')
    parser.add_argument('--single_gpu', dest='use_dp',
                        action='store_false',
                        help='use single GPU')
    parser.set_defaults(use_dp=True)


    args = parser.parse_args()
    print(args)

    return args
