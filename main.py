import argparse
import os
import importlib
import utils.models as models, utils.datasets as datasets
from torch.utils.data import DataLoader
import torchvision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        help='the datasets for evaluation;',
                        type=str,
                        choices=datasets.datasets_choices,
                        default='mnist')
    parser.add_argument('--epochs',
                        help='the number of epochs;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size;',
                        type=int,
                        default=128)
    parser.add_argument('--lr_passive',
                        help='learning rate for the passive parties;',
                        type=float,
                        default=0.01)
    parser.add_argument('--lr_active',
                        help='learning rate for the active party;',
                        type=float,
                        default=0.01)
    parser.add_argument('--lr_attack',
                        help='learning rate for the attacker;',
                        type=float,
                        default=0.01)
    parser.add_argument('--attack_epoch',
                        help='set epoch for attacking, greater than or equal to 2;',
                        type=int,
                        default=2)
    parser.add_argument('--attack_id',
                        help="the ID list of the attacker, like ``--attack_id 0 1'' for [0,1];",
                        nargs='*',
                        type=int,
                        default=[0])
    parser.add_argument('--num_passive',
                        help='number of passive parties;',
                        type=int,
                        default=1)
    parser.add_argument('--division',
                        help='choose the data division mode;',
                        type=str,
                        choices=['vertical', 'random', 'imbalanced'],
                        default='random')
    parser.add_argument('--round',
                        help='round for log;',
                        type=int,
                        default=0)
    parser.add_argument('--target_label',
                        help='target label, which aim to change to;',
                        type=int,
                        default=0)
    parser.add_argument('--source_label',
                        help='source label, which aim to change;',
                        type=int,
                        default=1)
    parser.add_argument('--trigger',
                        help='set trigger type;',
                        type=str,
                        choices=['our', 'villain', 'badvfl', 'basl'],
                        default='our')
    parser.add_argument('--add_noise',
                        help='add noise to embeddings for perturbation;',
                        action='store_true',
                        default=False)
    parser.add_argument('--update_centers',
                        help='update cluster center;',
                        action='store_true',
                        default=False)
    parser.add_argument('--defense',
                        help='choose the defense strategy;',
                        type=str,
                        choices=['none', 'dp', 'compression', 'detection', 'clip'],
                        default='none')
    parser.add_argument('--detection_rate',
                        help="``--detection_rate 0.8'' means that there is a 80 percent probability of detecting the trigger;",
                        type=float,
                        default=0.8)
    parser.add_argument('--compression_rate',
                        help='compression rate for the gradient compression defense;',
                        type=float,
                        default=0.5)
    parser.add_argument('--dp_epsilon',
                        help='privacy budget for the differential privacy defense;',
                        type=float,
                        default=0.01)
    parser.add_argument('--clip_rate',
                        help='clip rate for the gradient clipping defense;',
                        type=float,
                        default=0.8)
    
    # determine whether the arguments are legal or not
    args = parser.parse_args()
    # check the arguments
    if args.attack_epoch > args.epochs:
        raise ValueError('--attack_epoch should be smaller than or equals to --epochs')
    for i in args.attack_id:
        if i >= args.num_passive:
            raise ValueError('--attack_id should be smaller than --num_passive')
    if args.num_passive != 1:
        if args.dataset in ['mnist', 'fashionmnist'] and args.num_passive not in [2, 4, 7]:
            raise ValueError("The number of passive parties for {} must be 1, 2, 4 or 7.".format(datasets.datasets_name[args.dataset]))
        elif args.dataset in ['cifar10', 'cinic10', 'cifar100'] and args.num_passive not in [2, 4, 8]:
            raise ValueError("The number of passive parties for {} must be 1, 2, 4 or 8.".format(datasets.datasets_name[args.dataset]))
        elif args.dataset == "criteo" and args.num_passive != 3:
            raise ValueError("The number of passive parties for Criteo must be 1 or 3.")
    if args.division in ['random', 'imbalanced'] and args.dataset not in ['mnist', 'fashionmnist', 'cifar10', 'cinic10']:
        raise ValueError("Dataset {} can not use division={}, please use ``vertical'' instead.".format(datasets.datasets_name[args.dataset], args.division))
    if args.trigger == 'badvfl' and args.epochs < 3:
        raise ValueError("--epochs should be greater than or equal to 3 when --trigger='badvfl'")
    if args.dataset == "cinic10" and not os.path.exists("./dataset/CINIC10"):
            raise ValueError("You should download and unzip the CINIC10 dataset to ./dataset/CINIC10 first! Download: https://github.com/BayesWatch/cinic-10")
    
    # change the arguments to dictionary and print
    print('Arguments:')
    args_vars = vars(args)
    format_args = '\t%' + str(max([len(i) for i in args_vars.keys()])) + 's : %s'
    for pair in sorted(args_vars.items()): print(format_args % pair)

    # create a log directory
    dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    log_dir = os.path.join(dir, "log", args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create a data directory
    data_dir = os.path.join(dir, "data", args.dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # load dataset
    dataset_path = os.path.join(dir, 'dataset')
    if args.dataset == "cinic10":
        dataloader_train = DataLoader(torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/train',
            transform=datasets.transforms_default[args.dataset]), batch_size=args.batch_size, shuffle=True)
    else:
        data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True, download=True, transform=datasets.transforms_default[args.dataset])
        dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=False)

    if args.dataset == "cinic10":
        dataloader_test = DataLoader(torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/test',
            transform=datasets.transforms_default[args.dataset]), batch_size=args.batch_size, shuffle=True)
    else:
        data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False, transform=datasets.transforms_default[args.dataset])
        dataloader_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False)

    # load model
    entire_model = models.entire[args.dataset](num_passive=args.num_passive, division=args.division)

    # load attacker
    attacker_path = 'attackers.%s' % args.trigger
    attacker = getattr(importlib.import_module(attacker_path), 'Attacker')

    # call trainer
    t = attacker(args, entire_model, dataloader_train, dataloader_test)  # passive_model, active_model,
    t.train()
    t.backdoor()
    t.test()


if __name__ == '__main__':
    main()