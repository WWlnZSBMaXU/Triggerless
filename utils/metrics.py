import os
import json

class Metrics(object):
    def __init__(self, args):
        self.args = args        
        self.test_acc = []
        self.test_loss = []
        self.train_acc = []
        self.train_loss = []
        self.attack_acc = []
        self.targeted_BIR = []
        self.manipulated_BIR = []
        self.real_BIR = []
        self.manipulated_ASR = []
        self.real_ASR = []
        self.LISR = None
        self.dir = './log'
        self.rate = None

    def write(self):
        '''write existing history records into a json file'''
        metrics = {}
        metrics['dataset'] = self.args.dataset
        metrics['epochs'] = self.args.epochs
        metrics['batch_size'] = self.args.batch_size
        metrics['lr_passive'] = self.args.lr_passive
        metrics['lr_active'] = self.args.lr_active
        metrics['lr_attack'] = self.args.lr_attack
        metrics['attack_epoch'] = self.args.attack_epoch
        metrics['attack_id'] = self.args.attack_id
        metrics['num_passive'] = self.args.num_passive
        metrics['division'] = self.args.division
        metrics['round'] = self.args.round
        metrics['target_label'] = self.args.target_label
        metrics['source_label'] = self.args.source_label
        metrics['trigger'] = self.args.trigger
        metrics['add_noise'] = self.args.add_noise
        metrics['update_centers'] = self.args.update_centers
        metrics['defense'] = self.args.defense
        metrics['detection_rate'] = self.args.detection_rate
        metrics['compression_rate'] = self.args.compression_rate
        metrics['dp_epsilon'] = self.args.dp_epsilon
        metrics['clip_rate'] = self.args.clip_rate

        metrics['test_acc'] = self.test_acc
        metrics['test_loss'] = self.test_loss
        metrics['train_acc'] = self.train_acc
        metrics['train_loss'] = self.train_loss
        metrics['attack_acc'] = self.attack_acc
        metrics['targeted_BIR'] = self.targeted_BIR
        metrics['manipulated_BIR'] = self.manipulated_BIR
        metrics['real_BIR'] = self.real_BIR
        metrics['manipulated_ASR'] = self.manipulated_ASR
        metrics['real_ASR'] = self.real_ASR
        metrics['LISR'] = self.LISR
        metrics['rate'] = self.rate

        if self.args.defense == 'none':
            defense_rate = 0
        elif self.args.defense == 'dp':
            defense_rate = self.args.dp_epsilon
        elif self.args.defense == 'clip':
            defense_rate = self.args.clip_rate
        elif self.args.defense == 'compression':
            defense_rate = self.args.compression_rate
        elif self.args.defense == 'detection':
            defense_rate = self.args.detection_rate

        filename = "metrics_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.json".format(self.args.num_passive,
            self.args.batch_size,
            self.args.epochs,
            self.args.lr_passive,
            self.args.lr_attack, 
            self.args.attack_epoch, 
            self.args.attack_id,
            self.args.division,
            self.args.update_centers,
            self.args.add_noise,
            self.rate,
            self.args.trigger,
            self.args.defense,
            defense_rate,
            self.args.round)
        metrics_path = os.path.join(self.dir, self.args.dataset, filename)

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
