import torch
from models.base_model import BaselineModel

class BaselineExperiment: # See point 1. of the project
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = BaselineModel() # baseline的网络
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()

    def save_checkpoint(self, path, epoch, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['epoch'] = epoch
        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return epoch, iteration, best_accuracy, total_train_loss

    def train_iteration(self, data):
        # data: tuple(图片的tensor,类别label)
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device) # y dtype:int64
        logits = self.model(x) # 调用forward()向前传播 训练模型   训练完一次（base_model.py）
        loss = self.criterion(logits, y) # 经过交叉熵的到loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # loss 是一个tensor变量，含有单个元素的tensor.item()返回对应的值，若tensor有多个元素，则调用.item()报错
        return loss.item()

    def validate(self, loader):
        self.model.eval() #设置为evaluation 模式
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad(): # 禁用梯度计算，即使torch.tensor(xxx,requires_grad = True) 使用.requires_grad()也会返回False
            for x, y in loader: # type(x) tensor
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss += self.criterion(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss