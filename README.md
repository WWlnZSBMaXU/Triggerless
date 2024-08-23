This repository contains the code for the paper "Is the Trigger Essential? A Feature-Based Triggerless Backdoor Attack in Vertical Federated Learning".

# Requirements

```text
python==3.8.18
pytorch==1.11.0
scikit-learn==1.3.0
```

Or you can use `requirements.txt` and the following command to create the environment:

```bash
conda install --yes --file requirements.txt
```

# How to use?

```
usage: main.py [-h] [--dataset {mnist,fashionmnist,cifar10,cifar100,criteo,cinic10}] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr_passive LR_PASSIVE] [--lr_active LR_ACTIVE] [--lr_attack LR_ATTACK]
               [--attack_epoch ATTACK_EPOCH] [--attack_id [ATTACK_ID [ATTACK_ID ...]]] [--num_passive NUM_PASSIVE] [--division {vertical,random,imbalanced}] [--round ROUND] [--target_label TARGET_LABEL]
               [--source_label SOURCE_LABEL] [--trigger {our,villain,badvfl,basl}] [--add_noise] [--update_centers] [--defense {none,dp,compression,detection,clip}] [--detection_rate DETECTION_RATE]
               [--compression_rate COMPRESSION_RATE] [--dp_epsilon DP_EPSILON] [--clip_rate CLIP_RATE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {mnist,fashionmnist,cifar10,cifar100,criteo,cinic10}
                        the datasets for evaluation;
  --epochs EPOCHS       the number of epochs;
  --batch_size BATCH_SIZE
                        batch size;
  --lr_passive LR_PASSIVE
                        learning rate for the passive parties;
  --lr_active LR_ACTIVE
                        learning rate for the active party;
  --lr_attack LR_ATTACK
                        learning rate for the attacker;
  --attack_epoch ATTACK_EPOCH
                        set epoch for attacking, greater than or equal to 2;
  --attack_id [ATTACK_ID [ATTACK_ID ...]]
                        the ID list of the attacker, like ``--attack_id 0 1'' for [0,1];
  --num_passive NUM_PASSIVE
                        number of passive parties;
  --division {vertical,random,imbalanced}
                        choose the data division mode;
  --round ROUND         round for log;
  --target_label TARGET_LABEL
                        target label, which aim to change to;
  --source_label SOURCE_LABEL
                        source label, which aim to change;
  --trigger {our,villain,badvfl,basl}
                        set trigger type;
  --add_noise           add noise to embeddings for perturbation;
  --update_centers      update cluster center;
  --defense {none,dp,compression,detection,clip}
                        choose the defense strategy;
  --detection_rate DETECTION_RATE
                        ``--detection_rate 0.8'' means that there is a 80 percent probability of detecting the trigger;
  --compression_rate COMPRESSION_RATE
                        compression rate for the gradient compression defense;
  --dp_epsilon DP_EPSILON
                        privacy budget for the differential privacy defense;
  --clip_rate CLIP_RATE
                        clip rate for the gradient clipping defense;
```

## Backdoor attack with trigger VILLAIN

The trigger is from the paper:

> Yijie Bai, Yanjiao Chen et al. "{VILLAIN}: Backdoor Attacks Against Vertical Split Learning." 32nd USENIX Security Symposium. 2743-2760, 2023.

Some examples:

```bash
python main.py --trigger villain --dataset cifar10 --epochs 50 --attack_epoch 40 
python main.py --add_noise --dataset fashionmnist --num_passive 4
python main.py --trigger villain --defense clip --clip_rate 0.8
```

## Backdoor attack with trigger BadVFL

The trigger is from the paper:

> Mohammad Naseri, Yufei Han, and Emiliano De Cristofaro. "BadVFL: Backdoor Attacks in Vertical Federated Learning." arXiv preprint arXiv:2304.08847. 2023.

Some examples:

```bash
python main.py --trigger badvfl --dataset cinic10 --epochs 50 --attack_epoch 40 --num_passive 4
python main.py --trigger badvfl --defense compression --compression_rate 0.8
python main.py --trigger badvfl
```

## Backdoor attack with trigger BASL

The trigger is from the paper:

> Ying He, Zhili Shen et al. "Backdoor Attack Against Split Neural Network-Based Vertical Federated Learning," in IEEE Transactions on Information Forensics and Security. 19: 748-763, 2024.

Some examples:

```bash
python main.py --trigger basl --dataset criteo --division vertical
python main.py --trigger basl --defense dp --dp_epsilon 0.0001
python main.py --trigger basl --num_passive 4
```

## Our triggerless backdoor attack

Some examples:

```bash
python main.py --add_noise --dataset cinic10 --epochs 50
python main.py --num_passive 3 --dataset criteo --division vertical
python main.py --add_noise --defense detection --detection_rate 0.8
```

# Organization of code files

```bash
.
├── attackers
│   ├── badvfl.py
│   ├── basl.py
│   ├── our.py
│   ├── vflbase.py
│   └── villain.py
├── data  # auto-created: data for the backdoor attack will be auto sotred here
├── dataset  # auto-created: dataset will be auto downloaded here
├── log  # auto-created: training, testing and attacking record will be stored here
├── main.py
├── README.md
└── utils
    ├── datasets.py
    ├── metrics.py
    └── models.py

5 directories, 10 files
```

## main.py

The main file and program entry. Use it to load dataset, model, attacker, and to implement the baseline backdoor attacks and our proposed triggerless attack.

## attackers/vflbase.py

The base VFL model, where all attacker classes are inherited from this class. In this file, the following generic functions are included:

- **_process_data()**: Processing the dataset into the form required by the VFL.
- **train()** and **backdoor()**: Overridden by different attackers.
- **_evaluate()**: Evaluating the performance of VFL models in the training phase.
- **cluster()** and **_kmeans()**: For label inference module.
- **defense_grad()**: Used to implement three defense strategies: gradient clipping, gradient compression, and differential privacy.

## attackers/[Others]

Different backdoor attack methods with distinct triggers.

## utils/*

- **datasets.py**: Storing dataset information, which can be used to load and process different datasets.
- **metrics.py**: Class `Metrics` to record information to `log` folder. The logs for different datasets will be stored in different children folders with distinct file names.
- **models.py**: Neural network models for different datasets.
  
    | Dataset | Model |
    | --- | --- |
    | MNIST and FashionMNIST | FC1-FC1 (fully connected layer neural network) |
    | CIFAR-10 and CINIC-10 | Conv4-FC2 (convolutional neural network) |
    | CIFAR-100 | ResNet |
    | Criteo | DeepFM |