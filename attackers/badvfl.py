import torch
import numpy as np
from .vflbase import BaseVFL
import os


class Attacker(BaseVFL):
    def __init__(self, args, model, train_dataset, test_dataset):
        super(Attacker, self).__init__(args, model, train_dataset, test_dataset)
        self.args = args
        print("LIA: clustering")

        if self.args.num_passive - len(self.args.attack_id) == 0:
            self.rate = 0.4
        else:
            self.rate = 0.4 * (self.args.num_passive - len(self.args.attack_id)) / len(self.args.attack_id)

        self.std_devs = None
        
        self.trigger_row = [None] * self.args.num_passive
        self.trigger_col = [None] * self.args.num_passive
        self.trigger = [None] * self.args.num_passive
        self.trigger_pos = [None] * self.args.num_passive


    def train(self):
        self.iteration = len(self.train_dataset)

        cluster_centers = [[] for _ in range(self.args.num_passive)]
        std_devs = [[] for _ in range(self.args.num_passive)]
        trigger_positions = [{} for _ in range(self.args.num_passive)]
        min_sim = -1
        final_cluster_centers = [[] for _ in range(self.args.num_passive)]
        
        for epoch in range(self.args.epochs):
            # for calculating the backdoor implantation rate
            # targeted BIR
            targeted_correct = 0
            targeted_data_count = 0
            # manipulated BIR
            manipulated_correct = [0 for _ in range(self.args.num_passive)]
            manipulated_data_count = [0 for _ in range(self.args.num_passive)]
            # real BIR
            real_correct = [0 for _ in range(self.args.num_passive)]
            real_data_count = [0 for _ in range(self.args.num_passive)]

            # train entire model
            self.model.train()
            self.model.active.train()
            for i in range(self.args.num_passive):
                self.model.passive[i].train()

            # start train and attack
            for batch_idx, batch_data in enumerate(self.train_dataset):
                data, labels = batch_data

                if epoch < self.args.attack_epoch - 2:
                    emb, _, pred = self.model(data)
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
                # save the emb
                elif epoch == self.args.attack_epoch - 2:
                    emb, _, pred = self.model(data)
                    loss = self.loss(pred, labels)
                    
                    # save the emb
                    centers = self.cluster(emb, labels, batch_idx)
                    if centers:
                        for i in range(self.args.num_passive):
                            cluster_centers[i].append(centers[i])
                    # save std_dev
                    for i in range(self.args.num_passive):
                        std_dev = emb[i].std(dim=0).mean()
                        std_devs[i].append(std_dev)

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
                elif epoch == self.args.attack_epoch - 1:
                    for i in range(self.args.num_passive):
                        data[i].requires_grad = True
                    emb, _, pred = self.model(data)                
                    loss = self.loss(pred, labels)

                    # find the trigger positions
                    for attacker in self.args.attack_id:
                        grad_passive = torch.autograd.grad(loss, data[attacker], create_graph=True)[0]
                        # regognize the targeted emb
                        manipulated_data_list = []
                        for i in range(emb[attacker].shape[0]):
                            sim = torch.cosine_similarity(emb[attacker][i].flatten(), self.cluster_centers[attacker], dim=1)
                            label_sim = sim.argmax()
                            if label_sim == self.args.source_label:
                                if sim[label_sim] > min_sim:
                                    torch.save(data[attacker][i], os.path.join(self.data_dir,'targeted_data_{}.pt'.format(attacker)))
                                    min_sim = sim[label_sim]
                                position = abs(grad_passive[i]).argmax()
                                if position not in trigger_positions[attacker]:
                                    trigger_positions[attacker][position] = 0
                                else:
                                    trigger_positions[attacker][position] += 1
                            elif label_sim == self.args.target_label:
                                manipulated_data_list.append(i)
                        torch.save(manipulated_data_list, os.path.join(self.data_dir,'manipulated_data_list_{}_{}.pt'.format(attacker,batch_idx)))

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
                    elif self.args.defense == 'detection':
                        # record the target data
                        for attacker in self.args.attack_id:
                            target_idx_list = np.where(labels == self.args.target_label)
                            if len(target_idx_list[0]) == 0:
                                random_emb = torch.randn_like(emb[attacker]) * emb[attacker].std(dim=0).mean()
                                torch.save(random_emb, os.path.join(self.data_dir,'target_data_list_{}_{}.pt'.format(attacker,batch_idx)))
                            else:
                                torch.save(emb[attacker][target_idx_list[0]].detach().clone(), os.path.join(self.data_dir,'target_data_list_{}_{}.pt'.format(attacker,batch_idx)))
                    else:
                        loss.backward(retain_graph=True)
                    # update parameters for all optimizers
                    self.optimizer_entire.step()
                    self.optimizer_active.step()
                    for i in range(self.args.num_passive):
                        self.optimizer_passive[i].step()
                    
                    for i in range(self.args.num_passive):
                        data[i].requires_grad = False
                else:
                    # add trigger to data
                    real_source_idx = np.where(labels == self.args.source_label)
                    for attacker in self.args.attack_id:
                        manipulated_data_list = torch.load(os.path.join(self.data_dir,'manipulated_data_list_{}_{}.pt'.format(attacker,batch_idx)))
                        j_manipulate = 0
                        for i in manipulated_data_list:
                            if len(real_source_idx[0]) == 0 or j_manipulate >= len(real_source_idx[0]):
                                data[attacker][i] = self.trigger[attacker]
                            else:
                                source_sample_idx = real_source_idx[0][j_manipulate]
                                j_manipulate += 1
                                if self.args.dataset in ['mnist', 'fashionmnist']:
                                    data[attacker][i] = data[attacker][source_sample_idx]
                                    data[attacker][i][0][self.trigger_row-1:self.trigger_row+2, self.trigger_col-1:self.trigger_col+2] = torch.tensor([[0,1,0],[1,1,1],[0,1,0]])
                                elif self.args.dataset in ['cifar10', 'cifar100', 'cinic10']:
                                    data[attacker][i] = data[attacker][source_sample_idx]
                                    for channal in range(3):
                                        data[attacker][i][channal][self.trigger_row-1:self.trigger_row+2, self.trigger_col-1:self.trigger_col+2] = torch.tensor([[255,0,255],[0,0,0],[255,0,255]])
                                elif self.args.dataset == 'criteo':
                                    data[attacker][i] = data[attacker][source_sample_idx]
                                    data[attacker][i][self.trigger_pos-1:self.trigger_pos+2] = torch.tensor([0.0,1.0,0.0])

                    if self.args.defense == 'detection':
                        data = list(data)
                        emb = []
                        emb_old = []
                        for i in range(self.args.num_passive):
                            tmp_emb = self.model.passive[i](data[i])
                            emb_old.append(tmp_emb)
                            emb.append(tmp_emb.detach().clone())
                        
                        for attacker in self.args.attack_id:
                            manipulated_data_list = torch.load(os.path.join(self.data_dir,'manipulated_data_list_{}_{}.pt'.format(attacker,batch_idx)))
                            target_emb_list = torch.load(os.path.join(self.data_dir,'target_data_list_{}_{}.pt'.format(attacker,batch_idx)))
                            j_detect = 0
                            for i in manipulated_data_list:
                                detected_flag = np.random.rand()
                                if detected_flag <= self.args.detection_rate:  # and i < emb[attacker].shape[0]:
                                    if len(target_emb_list) == 0 or j_detect >= len(target_emb_list):
                                        emb[attacker][i] = torch.randn_like(emb[attacker][i]) * emb[attacker].std(dim=0).mean()
                                    else:
                                        emb[attacker][i] = target_emb_list[j_detect]
                                        j_detect += 1

                        for i in range(self.args.num_passive):
                            emb[i].requires_grad = True
                        # forward propagation
                        agg_emb = self.model._aggregate(emb)
                        logit = self.model.active(agg_emb)
                        pred = self.model.softmax(logit)
                        loss = self.loss(pred, labels)
                        # zero grad for all optimizers
                        self.optimizer_entire.zero_grad()
                        self.optimizer_active.zero_grad()
                        for i in range(self.args.num_passive):
                            self.optimizer_passive[i].zero_grad()
                        # backward propagation
                        loss.backward(retain_graph=True)
                        grad = torch.autograd.grad(loss, emb, create_graph=True)
                        for i in range(self.args.num_passive):
                            emb_old[i].backward(grad[i])
                        # update parameters for all optimizers
                        self.optimizer_entire.step()
                        self.optimizer_active.step()
                        for i in range(self.args.num_passive):
                            self.optimizer_passive[i].step()
                    else:
                        emb, _, pred = self.model(data)
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

                    pred = pred.argmax(dim=1, keepdim=True)
                    # get the number of correct predictions
                    # targeted BIR
                    target_idx = np.where(labels == self.args.target_label)
                    targeted_data_count += len(target_idx[0])
                    if len(target_idx[0]) == 0:
                        targeted_correct += 0
                    else:
                        target_labels = labels[target_idx[0]]
                        targeted_correct += pred[target_idx].eq(target_labels.view_as(pred[target_idx])).sum().item()
                    # manipulated BIR
                    for attacker in self.args.attack_id:
                        manipulated_data_list = torch.load(os.path.join(self.data_dir,'manipulated_data_list_{}_{}.pt'.format(attacker,batch_idx)))
                        manipulated_data_count[attacker] += len(manipulated_data_list)
                        if len(manipulated_data_list) == 0:
                            manipulated_correct[attacker] += 0
                        else:
                            target_labels = torch.full_like(labels[manipulated_data_list], self.args.target_label)
                            manipulated_correct[attacker] += pred[manipulated_data_list].eq(target_labels.view_as(pred[manipulated_data_list])).sum().item()
                    # real BIR
                    for attacker in self.args.attack_id:
                        manipulated_data_list = torch.load(os.path.join(self.data_dir,'manipulated_data_list_{}_{}.pt'.format(attacker,batch_idx)))
                        real_idx = list(set(target_idx[0]) & set(manipulated_data_list))
                        real_data_count[attacker] += len(real_idx)
                        if len(real_idx) == 0:
                            real_correct[attacker] += 0
                        else:
                            target_labels = torch.full_like(labels[real_idx], self.args.target_label)
                            real_correct[attacker] += pred[real_idx].eq(target_labels.view_as(pred[real_idx])).sum().item()
                    
                    # update cluster centers
                    if epoch == self.args.epochs - 1 and self.args.update_centers:
                        centers = self.cluster(emb, labels, batch_idx)
                        if centers:
                            for i in range(self.args.num_passive):
                                final_cluster_centers[i].append(centers[i])

                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == self.iteration:
                    print('Epoch:{}/{}, Step:{} \tLoss: {:.6f}'.format(epoch+1, self.args.epochs, batch_idx+1, loss.item()))

            if epoch == self.args.attack_epoch - 2:
                # statistics for cluster centers
                self.cluster_centers = [[] for _ in range(self.args.num_passive)]
                for i in range(self.args.num_passive):
                    self.cluster_centers[i] = torch.tensor(np.array(cluster_centers[i]).mean(axis=0))
                print("Finish updating cluster centers.")
                # statistics for std_devs
                self.std_devs = []
                for i in range(self.args.num_passive):
                    self.std_devs.append(torch.tensor(std_devs[i]).mean())
            elif epoch == self.args.attack_epoch - 1:
                for attacker in self.args.attack_id:
                    position = max(trigger_positions[attacker], key=trigger_positions[attacker].get)
                    if self.args.dataset in ['mnist', 'fashionmnist']:
                        width = 28 / self.args.num_passive
                        if position % 28 == 0:
                            position += 1
                        elif position % 28 == 27:
                            position -= 1
                        if int(position / 28) == 0:
                            position += 28
                        elif int(position / 28) == width - 1:
                            position -= 28
                        self.trigger_row = position % 28
                        self.trigger_col = int(position / 28)
                        self.trigger[attacker] = torch.load(os.path.join(self.data_dir,'targeted_data_{}.pt'.format(attacker)))
                        self.trigger[attacker].requires_grad = False
                        self.trigger[attacker][0][self.trigger_row-1:self.trigger_row+2, self.trigger_col-1:self.trigger_col+2] = torch.tensor([[0,1,0],[1,1,1],[0,1,0]])
                    elif self.args.dataset in ['cifar10', 'cifar100', 'cinic10']:
                        width = int(32 / self.args.num_passive)
                        position %= 32 * width
                        if position % 32 == 0:
                            position += 1
                        elif position % 32 == 31:
                            position -= 1
                        if int(position / 32) == 0:
                            position += 32
                        elif int(position / 32) == width - 1:
                            position -= 32
                        self.trigger_row = position % 32
                        self.trigger_col = int(position / 32)
                        self.trigger[attacker] = torch.load(os.path.join(self.data_dir,'targeted_data_{}.pt'.format(attacker)))
                        self.trigger[attacker].requires_grad = False
                        for channal in range(3):
                            self.trigger[attacker][channal][self.trigger_row-1:self.trigger_row+2, self.trigger_col-1:self.trigger_col+2] = torch.tensor([[255,0,255],[0,0,0],[255,0,255]])
                    elif self.args.dataset == 'criteo':                        
                        width = 39 / self.args.num_passive
                        if position == 0:
                            position += 1
                        elif position == width - 1:
                            position -= 1
                        self.trigger_pos = position
                        self.trigger[attacker] = torch.load(os.path.join(self.data_dir,'targeted_data_{}.pt'.format(attacker)))
                        self.trigger[attacker].requires_grad = False
                        self.trigger[attacker][self.trigger_pos-1:self.trigger_pos+2] = torch.tensor([0.0,1.0,0.0])
            elif epoch > self.args.attack_epoch - 1:
                # targeted BIR
                backdoor_acc = 100. * targeted_correct / targeted_data_count
                print('Targeted Backdoor Implantation Rate: {}/{} ({:.2f}%)'.format(targeted_correct, targeted_data_count, backdoor_acc))
                self.metrics.targeted_BIR.append(backdoor_acc)
                # manipulated BIR
                oBIR = []
                for attacker in self.args.attack_id:
                    backdoor_acc = 100. * manipulated_correct[attacker] / manipulated_data_count[attacker]
                    print('Attacker {}\'s Manipulated Backdoor Implantation Rate: {}/{} ({:.2f}%)'.format(attacker, manipulated_correct[attacker], manipulated_data_count[attacker], backdoor_acc))
                    oBIR.append(backdoor_acc)
                self.metrics.manipulated_BIR.append(oBIR)
                # real BIR
                rBIR = []
                for attacker in self.args.attack_id:
                    if real_data_count[attacker] == 0:
                        backdoor_acc = 0
                    else:
                        backdoor_acc = 100. * real_correct[attacker] / real_data_count[attacker]
                    print('Attacker {}\'s Real Backdoor Implantation Rate: {}/{} ({:.2f}%)'.format(attacker, real_correct[attacker], real_data_count[attacker], backdoor_acc))
                    rBIR.append(backdoor_acc)
                self.metrics.real_BIR.append(rBIR)

                # update cluster centers
                if epoch == self.args.epochs - 1 and self.args.update_centers:
                    self.cluster_centers = [[] for _ in range(self.args.num_passive)]
                    for i in range(self.args.num_passive):
                        self.cluster_centers[i] = torch.tensor(np.array(final_cluster_centers[i]).mean(axis=0))
                    print("Finish updating final cluster centers.")
            
            self.metrics.write()

            # evaluate the model each epoch
            self._evaluate()

    
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
                ### for calculating the LIA attack accuracy ###
                emb_LIA = []
                for i in range(self.args.num_passive):
                    emb_LIA.append(self.model.passive[i](data[i]))
                self.cluster(emb_LIA, labels, batch_idx)
                ###############################################

                manipulated_idx_list = [[] for _ in range(self.args.num_passive)]
                for attacker in self.args.attack_id:
                    # regognize the targeted emb
                    emb = self.model.passive[attacker](data[attacker])
                    for i in range(emb.shape[0]):
                        sim = torch.cosine_similarity(emb[i].flatten(), self.cluster_centers[attacker], dim=1)
                        label_sim = sim.argmax()
                        if label_sim == self.args.source_label:
                            manipulated_idx_list[attacker].append(i)
                            if self.args.dataset in ['mnist', 'fashionmnist']:
                                data[attacker][i][0][self.trigger_row-1:self.trigger_row+2, self.trigger_col-1:self.trigger_col+2] = torch.tensor([[0,1,0],[1,1,1],[0,1,0]])
                            elif self.args.dataset in ['cifar10', 'cifar100', 'cinic10']:
                                for channal in range(3):
                                    data[attacker][i][channal][self.trigger_row-1:self.trigger_row+2, self.trigger_col-1:self.trigger_col+2] = torch.tensor([[255,0,255],[0,0,0],[255,0,255]])
                            elif self.args.dataset == 'criteo':
                                data[attacker][i][self.trigger_pos-1:self.trigger_pos+2] = torch.tensor([0.0,1.0,0.0])
                # forward propagation                
                _, _, pred = self.model(data)
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


    def test(self):
        super().test()