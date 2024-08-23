import torch
import numpy as np
from .vflbase import BaseVFL


class Attacker(BaseVFL):
    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        print("LIA: clustering")

        if self.args.num_passive - len(self.args.attack_id) == 0:
            self.rate = 1
        else:
            self.rate = (self.args.num_passive - len(self.args.attack_id)) / len(self.args.attack_id)
        self.metrics.rate = self.rate


    def train(self):
        self.iteration = len(self.train_dataset)

        cluster_centers = [[] for _ in range(self.args.num_passive)]
        
        for epoch in range(self.args.epochs):
            # train entire model
            self.model.train()
            self.model.active.train()
            for i in range(self.args.num_passive):
                self.model.passive[i].train()

            # start train and attack
            for batch_idx, batch_data in enumerate(self.train_dataset):
                data, labels = batch_data

                emb, _, pred = self.model(data)

                # save the emb
                if epoch == self.args.epochs - 1:
                    centers = self.cluster(emb, labels, batch_idx)
                    if centers:
                        for i in range(self.args.num_passive):
                            cluster_centers[i].append(centers[i])

                loss = self.loss(pred, labels)

                # zero grad for all optimizers
                self.optimizer_entire.zero_grad()
                self.optimizer_active.zero_grad()
                for i in range(self.args.num_passive):
                    self.optimizer_passive[i].zero_grad()

                # backward propagation
                if self.args.defense in ['dp', 'compression', 'clip']:
                    loss.backward(retain_graph=True)
                    grad = torch.autograd.grad(loss, emb, create_graph=True)
                    grad = self.defense_grad(grad)
                    for i in range(self.args.num_passive):
                        emb[i].backward(grad[i])
                else:
                    loss.backward()

                # update parameters for all optimizers
                self.optimizer_entire.step()
                self.optimizer_active.step()
                for i in range(self.args.num_passive):
                    self.optimizer_passive[i].step()

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == self.iteration:
                    print('Epoch:{}/{}, Step:{} \tLoss: {:.6f}'.format(epoch+1, self.args.epochs, batch_idx+1, loss.item()))

            # evaluate the model each epoch
            self._evaluate()
        
        self.cluster_centers = [[] for _ in range(self.args.num_passive)]
        for i in range(self.args.num_passive):
            self.cluster_centers[i] = torch.tensor(np.array(cluster_centers[i]).mean(axis=0))

    
    def backdoor(self):
        print("\n============== Backdoor ==============")
        self.iteration = len(self.test_dataset)

        # test entire model and show test loss and accuracy
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        # manipulated backdoor attack accuracy
        manipulated_correct = [0 for _ in range(self.args.num_passive)]
        manipulated_data_count = [0 for _ in range(self.args.num_passive)]
        # real backdoor attack accuracy
        real_correct = [0 for _ in range(self.args.num_passive)]
        real_data_count = [0 for _ in range(self.args.num_passive)]
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_dataset):
                data, labels = batch_data
                data = list(data)
                emb = []
                for i in range(self.args.num_passive):
                    emb.append(self.model.passive[i](data[i]))
                
                # culculate the accuracy of LIA
                self.cluster(emb, labels, batch_idx)
                
                # add trigger
                manipulated_idx_list = [[] for _ in range(self.args.num_passive)]
                for attacker in self.args.attack_id:
                    # regognize the targeted emb
                    for i in range(emb[attacker].shape[0]):
                        sim = torch.cosine_similarity(emb[attacker][i].flatten(), self.cluster_centers[attacker], dim=1)
                        label_sim = sim.argmax()
                        if label_sim == self.args.source_label:
                            manipulated_idx_list[attacker].append(i)
                            emb[attacker][i] = self.add_noise(self.cluster_centers[attacker][self.args.target_label].to(torch.float32) * self.rate).reshape(emb[attacker][0].shape)
                # forward propagation
                agg_emb = self.model._aggregate(emb)
                logit = self.model.active(agg_emb)
                pred = self.model.softmax(logit)

                pred = pred.argmax(dim=1, keepdim=True)

                # get the number of correct predictions
                for attacker in self.args.attack_id:
                    # manipulated backdoor attack accuracy
                    manipulated_idx = manipulated_idx_list[attacker]
                    manipulated_data_count[attacker] += len(manipulated_idx)
                    if len(manipulated_idx) == 0:
                        manipulated_correct[attacker] += 0
                    else:
                        target_labels = torch.full_like(labels[manipulated_idx], self.args.target_label)
                        manipulated_correct[attacker] += pred[manipulated_idx].eq(target_labels.view_as(pred[manipulated_idx])).sum().item()
                    # real backdoor attack accuracy
                    source_idx = np.where(labels == self.args.source_label)[0]
                    real_idx = list(set(manipulated_idx) & set(source_idx))
                    real_data_count[attacker] += len(real_idx)
                    if len(real_idx) == 0:
                        real_correct[attacker] += 0
                    else:
                        target_labels = torch.full_like(labels[real_idx], self.args.target_label)
                        real_correct[attacker] += pred[real_idx].eq(target_labels.view_as(pred[real_idx])).sum().item()

        for attacker in self.args.attack_id:
            # manipulated backdoor attack accuracy
            if manipulated_data_count[attacker] == 0:
                backdoor_acc = 0
            else:
                backdoor_acc = 100. * manipulated_correct[attacker] / manipulated_data_count[attacker]
            print('Attacker {}\'s Manipulated Backdoor Attack Accuracy: {}/{} ({:.2f}%)'.format(attacker, manipulated_correct[attacker], manipulated_data_count[attacker], backdoor_acc))
            self.metrics.manipulated_ASR.append(backdoor_acc)
            # real backdoor attack accuracy
            if real_data_count[attacker] == 0:
                backdoor_acc = 0
            else:
                backdoor_acc = 100. * real_correct[attacker] / real_data_count[attacker]
            print('Attacker {}\'s Real Backdoor Attack Accuracy: {}/{} ({:.2f}%)'.format(attacker, real_correct[attacker], real_data_count[attacker], backdoor_acc))
            self.metrics.real_ASR.append(backdoor_acc)
        self.metrics.write()
    

    def add_noise(self, x):
        if self.args.add_noise:
            mu = x.mean()
            sigma = x.std()
            return x + torch.tensor(np.random.normal(mu, sigma, x.shape))
        else:
            return x


    def test(self):
        super().test()