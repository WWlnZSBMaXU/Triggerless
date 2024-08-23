import torch
from utils.metrics import Metrics
import os
import numpy as np
import utils.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class BaseVFL(object):
    def __init__(self, args, entire_model, train_dataset, test_dataset):  # passive_model, active_model
        # setup arguments
        self.args = args

        # get data file path
        self.data_dir = os.path.join('./data', self.args.dataset,
            'data_{}_{}_{}_{}_{}_{}'.format(self.args.num_passive,
                self.args.batch_size,
                self.args.epochs,
                self.args.attack_epoch, 
                self.args.attack_id,
                self.args.division))
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # process dataset
        self._process_data(train_dataset, test_dataset)

        # setup entire model and optimizer
        self.model = entire_model
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer_entire = torch.optim.SGD(self.model.parameters(), lr=args.lr_active)
        self.optimizer_active = torch.optim.SGD(self.model.active.parameters(), lr=args.lr_active)
        self.optimizer_passive = []
        for i in range(args.num_passive):
            lr = args.lr_attack if i in args.attack_id else args.lr_passive
            self.optimizer_passive.append(torch.optim.SGD(self.model.passive[i].parameters(), lr=lr))

        # setup metrics
        self.metrics = Metrics(args)

        # record iteration
        self.iteration = None

        # record cluster centers
        self.cluster_centers = None

        # init cluster LIA        
        self.total_acc = [0] * self.args.num_passive
        self.round = 0


    def _process_data(self, train_dataset, test_dataset):
        print("Processing dataset...")

        if self.args.division == 'vertical':
            self.train_dataset = []
            self.train_dataset_len = 0
            for batch_data in train_dataset:
                data, labels = batch_data  # len(data) = batch_size 128
                # for mnist and fashionmnist: torch.Size([128, 1, 28, 28])
                # for cifar10: torch.Size([128, 3, 32, 32])
                if self.args.dataset == "criteo":
                    self.train_dataset.append([torch.chunk(data, self.args.num_passive, dim=1), labels])
                else:
                    self.train_dataset.append([torch.chunk(data, self.args.num_passive, dim=3), labels])
                self.train_dataset_len += len(data)

            self.test_dataset = []
            self.test_dataset_len = 0
            for batch_data in test_dataset:
                data, labels = batch_data
                if self.args.dataset == "criteo":
                    self.test_dataset.append([torch.chunk(data, self.args.num_passive, dim=1), labels])
                else:
                    self.test_dataset.append([torch.chunk(data, self.args.num_passive, dim=3), labels])
                self.test_dataset_len += len(data)
        elif self.args.division == 'random':
            if self.args.dataset not in ['mnist', 'fashionmnist', 'cifar10', "cifar100", 'cinic10']:
                raise ValueError("Random division only supports MNIST and CIFAR-10.")

            sample_list = []
            if self.args.num_passive == 1:
                sample_list.append(list(range(28)) if self.args.dataset in ['mnist', 'fashionmnist'] else list(range(32)))
            elif self.args.num_passive == 2:
                if self.args.dataset in ["mnist", "fashionmnist"]:
                    list_0 = [1, 4, 5, 7, 9, 11, 13, 14, 16, 18, 19, 23, 24, 26]
                    list_1 = [0, 2, 3, 6, 8, 10, 12, 15, 17, 20, 21, 22, 25, 27]
                elif self.args.dataset in ["cifar10", "cifar100", "cinic10"]:
                    list_0 = [0, 3, 4, 5, 6, 7, 9, 14, 15, 16, 22, 23, 28, 29, 30, 31]
                    list_1 = [1, 2, 8, 10, 11, 12, 13, 17, 18, 19, 20, 21, 24, 25, 26, 27]
                sample_list.append(list_0)
                sample_list.append(list_1)
            elif self.args.num_passive == 4:
                if self.args.dataset in ["mnist", "fashionmnist"]:
                    list_0 = [1, 2, 3, 17, 20, 23, 27]
                    list_1 = [6, 8, 11, 12, 13, 14, 24]
                    list_2 = [5, 7, 9, 10, 22, 25, 26]
                    list_3 = [0, 4, 15, 16, 18, 19, 21]
                elif self.args.dataset in ["cifar10", "cifar100", "cinic10"]:
                    list_0 = [0, 4, 6, 14, 20, 24, 25, 28]
                    list_1 = [1, 3, 9, 15, 17, 19, 22, 27]
                    list_2 = [5, 7, 11, 16, 18, 21, 23, 31]
                    list_3 = [2, 8, 10, 12, 13, 26, 29, 30]
                sample_list.append(list_0)
                sample_list.append(list_1)
                sample_list.append(list_2)
                sample_list.append(list_3)
            elif self.args.num_passive == 7 and self.args.dataset in ["mnist", "fashionmnist"]:
                sample_list = [[ 5, 20,  2, 17],
                               [23, 15,  1, 18],
                               [11, 24, 26, 16],
                               [ 0, 14,  8,  4],
                               [22, 25,  7,  3],
                               [10, 19,  9, 27],
                               [21, 13, 12,  6]]
            elif self.args.num_passive == 8 and self.args.dataset in ["cifar10", "cifar100", "cinic10"]:
                sample_list = [[ 4,  6,  0, 28],
                               [20, 24,  7, 30],
                               [14, 25, 23, 31],
                               [ 9, 15,  1, 19],
                               [17, 27,  3, 21],
                               [ 8, 16,  2, 22],
                               [ 5, 13, 11, 31],
                               [12, 26, 10, 29]]

            self.train_dataset = []
            self.train_dataset_len = 0
            for batch_data in train_dataset:
                data, labels = batch_data
                data_list = []
                for i in range(self.args.num_passive):
                    data_list.append(data.index_select(3, torch.tensor(sample_list[i])))
                self.train_dataset.append([data_list, labels])
                self.train_dataset_len += len(data)

            self.test_dataset = []
            self.test_dataset_len = 0
            for batch_data in test_dataset:
                data, labels = batch_data
                data_list = []
                for i in range(self.args.num_passive):
                    data_list.append(data.index_select(3, torch.tensor(sample_list[i])))
                self.test_dataset.append([data_list, labels])
                self.test_dataset_len += len(data)
        elif self.args.division == 'imbalanced':
            if self.args.dataset not in ['mnist', 'cifar10', "cifar100", 'cinic10']:
                raise ValueError("Imbalance division only supports MNIST, CIFAR-10, and CINIC-10.")

            sample_list = []
            if self.args.num_passive == 1:
                sample_list.append(list(range(28)) if self.args.dataset == "mnist" else list(range(32)))
            elif self.args.num_passive == 2:
                if self.args.dataset in ["mnist", "fashionmnist"]:
                    # 20 & 8
                    list_0 = [0, 2, 3, 4, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26, 27]
                    list_1 = [1, 5, 6, 8, 13, 17, 20, 24]
                elif self.args.dataset in ["cifar10", "cifar100", "cinic10"]:
                    # 20 & 12
                    list_0 = [0, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16, 18, 21, 22, 24, 25, 26, 28, 30, 31]
                    list_1 = [1, 2, 5, 6, 11, 12, 17, 19, 20, 23, 27, 29]
                sample_list.append(list_0)
                sample_list.append(list_1)
            elif self.args.num_passive == 4:
                if self.args.dataset in ["mnist", "fashionmnist"]:
                    # 12 & 6 & 3 & 7
                    list_0 = [1, 3, 4, 5, 7, 11, 14, 15, 19, 21, 23, 27]
                    list_1 = [2, 6, 9, 10, 12, 22]
                    list_2 = [0, 13, 17]
                    list_3 = [8, 16, 18, 20, 24, 25, 26]
                elif self.args.dataset in ["cifar10", "cifar100", "cinic10"]:
                    # 13 & 7 & 4 & 8
                    list_0 = [0, 3, 6, 7, 12, 14, 15, 16, 23, 27, 29, 30, 31]
                    list_1 = [1, 2, 10, 13, 19, 22, 24]
                    list_2 = [8, 9, 11, 26]
                    list_3 = [4, 5, 17, 18, 20, 21, 25, 28]
                sample_list.append(list_0)
                sample_list.append(list_1)
                sample_list.append(list_2)
                sample_list.append(list_3)

            self.train_dataset = []
            self.train_dataset_len = 0
            for batch_data in train_dataset:
                data, labels = batch_data
                data_list = []
                for i in range(self.args.num_passive):
                    data_list.append(data.index_select(3, torch.tensor(sample_list[i])))
                self.train_dataset.append([data_list, labels])
                self.train_dataset_len += len(data)

            self.test_dataset = []
            self.test_dataset_len = 0
            for batch_data in test_dataset:
                data, labels = batch_data
                data_list = []
                for i in range(self.args.num_passive):
                    data_list.append(data.index_select(3, torch.tensor(sample_list[i])))
                self.test_dataset.append([data_list, labels])
                self.test_dataset_len += len(data)

        print("Finish processing dataset!")

    
    def train(self):
        pass


    def backdoor(self):
        pass


    def test(self):
        print("\n============== Test ==============")
        self.iteration = len(self.test_dataset)

        # test entire model and show test loss and accuracy
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for _, batch_data in enumerate(self.test_dataset):
                data, labels = batch_data
                _, _, pred = self.model(data)

                test_loss += self.loss(pred, labels).item()
                pred = pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(self.test_dataset)
        test_acc = 100. * correct / self.test_dataset_len
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, self.test_dataset_len, test_acc))        
        
        self.metrics.test_loss.append(test_loss)
        self.metrics.test_acc.append(test_acc)
        self.metrics.write()

        return test_acc
    

    def _evaluate(self):
        # evaluate entire model and show training loss and accuracy
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        train_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_data in self.train_dataset:
                data, labels = batch_data  # data is tuple, len(data[0]) = batch_size
                _, _, pred = self.model(data)
                train_loss += self.loss(pred, labels).item()
                pred = pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        train_loss /= len(self.train_dataset)
        train_acc = 100. * correct / self.train_dataset_len
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            train_loss, correct, self.train_dataset_len, train_acc))        
        
        self.metrics.train_loss.append(train_loss)
        self.metrics.train_acc.append(train_acc)
        self.metrics.write()

        return train_acc
    

    def cluster(self, data, labels, batch_idx, phase='train'):        
        num_classes = datasets.datasets_classes[self.args.dataset]
        cluster_centers_list = []

        if labels.shape[0] != self.args.batch_size:
            pass
        else:
            self.round += 1  # inplement attack once

            # process emb
            passive_emb = []
            for passive_id in range(len(data)):
                passive_emb.append(data[passive_id].reshape(data[passive_id].shape[0], -1).detach().numpy())  # KMeans expected dim <= 2.
            # Using K-means to cluster embeddings for different passive parties.
            acc_list, cluster_centers_list = self._kmeans(num_classes, passive_emb, labels)

            # update total accuracy
            for passive_id in range(self.args.num_passive):
                self.total_acc[passive_id] += acc_list[passive_id]

        # calculate average accuracy and write metrics
        if batch_idx == self.iteration - 1:
            avg_acc = []
            for passive_id in range(self.args.num_passive):
                avg_acc.append(self.total_acc[passive_id] / self.round)
                print('Average LIA Attack Accuracy of Passive {} (each epoch): {:.2f}%'.format(passive_id, avg_acc[passive_id]))
            if phase == 'train':
                self.metrics.attack_acc.append(avg_acc)
            elif phase == 'backdoor':
                self.metrics.LISR = avg_acc
            self.metrics.write()

            self.total_acc = [0] * self.args.num_passive
            self.round = 0
        
        return cluster_centers_list


    def _kmeans(self, num_classes, data, labels):
        '''
        K-means clustering.
        '''
        # initialize the attack predicted labels
        cluster_labels = torch.randint(0, 9, labels.shape, dtype=torch.long)

        acc_list = [0] * self.args.num_passive
        cluster_centers_list = []
        for passive_id in range(self.args.num_passive):
            # algorithm{'lloyd', 'elkan', 'auto', 'full'}, default='lloyd'
            kmeans = KMeans(n_clusters=num_classes, random_state=0, n_init='auto')
            kmeans.fit(data[passive_id])
            kmeans_labels = kmeans.predict(data[passive_id])

            # calculate the closest point to the center
            dis = kmeans.transform(data[passive_id]).min(axis=1)  # n_samples * n_clusters
            always_correct = 0
            for i in range(num_classes):
                i_idx = np.where(kmeans_labels == i)  # tuple, size=1
                if len(i_idx[0]) == 0:
                    continue
                always_correct += 1
                closest_idx = i_idx[0][dis[i_idx].argmin()]

                # update labels according to the closest point
                cluster_labels[i_idx] = labels[closest_idx]

            # calculate the accuracy
            correct = cluster_labels.eq(labels).sum().item()
            attack_acc = 100. * (correct - always_correct) / (labels.shape[0] - always_correct)
            acc_list[passive_id] = attack_acc

            # save the cluster centers
            cluster_centers = []
            for i in range(num_classes):
                i_idx = np.where(cluster_labels == i)
                if len(i_idx[0]) == 0:
                    cluster_centers.append(np.random.rand(data[passive_id][0].shape[0]))
                    continue
                cluster_centers.append(data[passive_id][i_idx].mean(axis=0))
            cluster_centers_list.append(cluster_centers)

        return acc_list, cluster_centers_list


    def defense_grad(self, grad):
        grad_new = []
        for i in range(self.args.num_passive):            
            grad_copy = grad[i].clone().detach()
            if self.args.defense == 'dp':
                sensitivity = grad_copy.flatten().norm(dim=0, p=2)
                grad_clipped = torch.clamp(grad_copy, -sensitivity, sensitivity)
                noise = torch.tensor(np.random.laplace(loc=0, scale=1.0 * sensitivity / self.args.dp_epsilon, size=grad_clipped.shape))
                grad_dp = grad_clipped + noise
                grad_new.append(grad_dp)
            elif self.args.defense == 'compression':
                # get threshold
                survival_values = torch.topk(grad_copy.abs().reshape(1, -1), int(grad_copy.abs().reshape(1, -1).shape[1] * self.args.compression_rate)).values
                threshold = survival_values.reshape(1, -1).min()
                # compress
                grad_zeros = torch.zeros_like(grad_copy)
                grad_compressed = torch.where(grad_copy.abs() > threshold, grad_copy, grad_zeros)
                grad_new.append(grad_compressed)
            elif self.args.defense == 'clip':
                clip_value = grad_copy.abs().max() * self.args.clip_rate
                grad_clipped = torch.clamp(grad_copy, -clip_value, clip_value)
                grad_new.append(grad_clipped)
        return grad_new