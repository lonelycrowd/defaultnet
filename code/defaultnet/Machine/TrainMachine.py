import logging as log
import torch
from torchvision import transforms as tf
from statistics import mean
import os

from ..Data.Format.dataloader import DataLoader,list_collate
from ..Data.Datasets.custom_dataset import CustomDataset
from ..Models import Sketch as MyModel
from .Machine import BaseMachine



class TrainMachine(BaseMachine):

    def __init__(self, hyper_params):
        self.hyper_params = hyper_params
        # all in args
        self.batch_size = hyper_params.batch
        self.mini_batch_size = hyper_params.mini_batch
        self.max_batches = hyper_params.max_batches

        self.classes = hyper_params.classes

        self.cuda = hyper_params.cuda
        self.backup_dir = hyper_params.backup_dir

        log.debug('Creating network')
        model_name = hyper_params.model_name
        net = MyModel(hyper_params.classes, hyper_params.weights, train_flag=1, clear=hyper_params.clear)
        log.info('Net structure\n\n%s\n' % net)
        if self.cuda:
            net.cuda()

        log.debug('Creating optimizer')
        learning_rate = hyper_params.learning_rate
        momentum = hyper_params.momentum
        decay = hyper_params.decay
        batch = hyper_params.batch
        log.info(f'Adjusting learning rate to [{learning_rate}]')
        optim = torch.optim.SGD(net.parameters(), lr=learning_rate/batch, momentum=momentum, dampening=0, weight_decay=decay*batch)

        log.debug('Creating dataloader')
        dataset = CustomDataset(hyper_params.dataset_params)
        dataloader = data.DataLoader(
            dataset,
            batch_size = self.mini_batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = hyper_params.nworkers if self.cuda else 0,
            pin_memory = hyper_params.pin_mem if self.cuda else False,
            collate_fn = list_collate,
        )

        super(TrainMachine, self).__init__(net, optim, dataloader)

        self.nloss = self.network.nloss

        self.train_loss = [{'tot': [], 'coord': [], 'conf': [], 'cls': []} for _ in range(self.nloss)]
        
    def start(self):
        log.debug('Creating additional logging objects')
        hyper_params = self.hyper_params

        lr_steps = hyper_params.lr_steps
        lr_rates = hyper_params.lr_rates

        bp_steps = hyper_params.bp_steps
        bp_rates = hyper_params.bp_rates
        backup = hyper_params.backup

        rs_steps = hyper_params.rs_steps
        rs_rates = hyper_params.rs_rates
        resize = hyper_params.resize

        self.add_rate('learning_rate', lr_steps, [lr/self.batch_size for lr in lr_rates])
        self.add_rate('backup_rate', bp_steps, bp_rates, backup)
        self.add_rate('resize_rate', rs_steps, rs_rates, resize)

        self.dataloader.change_input_dim()

        
    def process_batch(self, data):
        data, target = data
        # to(device)
        if self.cuda:
            data = data.cuda()
        #data = torch.autograd.Variable(data, requires_grad=True)

        loss = self.network(data, target)
        loss.backward()

        for ii in range(self.nloss):
            self.train_loss[ii]['tot'].append(self.network.loss[ii].loss_tot.item() / self.mini_batch_size)
            self.train_loss[ii]['coord'].append(self.network.loss[ii].loss_coord.item() / self.mini_batch_size)
            self.train_loss[ii]['conf'].append(self.network.loss[ii].loss_conf.item() / self.mini_batch_size)
            if self.network.loss[ii].loss_cls is not None:
                self.train_loss[ii]['cls'].append(self.network.loss[ii].loss_cls.item() / self.mini_batch_size)
    
    
    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        all_tot = 0.0
        all_coord = 0.0
        all_conf = 0.0
        all_cls = 0.0
        for ii in range(self.nloss):
            tot = mean(self.train_loss[ii]['tot'])
            coord = mean(self.train_loss[ii]['coord'])
            conf = mean(self.train_loss[ii]['conf'])
            all_tot += tot
            all_coord += coord
            all_conf += conf
            if self.classes > 1:
                cls = mean(self.train_loss[ii]['cls'])
                all_cls += cls
                
            log.info(f'{self.batch} # {ii}: Loss:{round(tot, 5)} (Coord:{round(coord, 2)} Conf:{round(conf, 2)} Cls:{round(cls, 2)})')


        log.info(f'{self.batch} # All : Loss:{round(all_tot, 5)} (Coord:{round(all_coord, 2)} Conf:{round(all_conf, 2)} Cls:{round(all_cls, 2)})')

        self.train_loss = [{'tot': [], 'coord': [], 'conf': [], 'cls': []} for _ in range(self.nloss)]
        if self.batch % self.backup_rate == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'weights_{self.batch}.pt'))

        if self.batch % 100 == 0:
            self.network.save_weights(os.path.join(self.backup_dir, f'backup_loss:'+str(round(all_tot, 3))+'.pt'))

        if self.batch % self.resize_rate == 0:
            if self.batch + 200 >= self.max_batches:
                finish_flag = True
            else:
                finish_flag = False
            self.dataloader.change_input_dim(finish=finish_flag)


    def quit(self):
        if self.sigint:
            self.network.save_weights(os.path.join(self.backup_dir, f'resume_backup.pt'))
            return True
        elif self.batch >= self.max_batches:
            self.network.save_weights(os.path.join(self.backup_dir, f'final.dw'))
            return True
        else:
            return False
