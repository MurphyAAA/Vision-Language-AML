import torch
from models.base_model import DomainDisentangleModel
from my_loss import DomainDisentangleLoss
import torch.nn.functional as F

class DomainDisentangleExperiment: # See point 2. of the project

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        # Loss functions
        # self.criterion = DomainDisentangleLoss()
        self.nll_loss = torch.nn.NLLLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.rec_loss = torch.nn.MSELoss()
        self.alpha1 = 1.2
        self.alpha2 = 0.5
        self.w1 = 2 #主要训练category分类器 所以他的权重高一点，其他权重低一点
        self.w2 = 1
        self.w3 = 1 # 2,1,1
        # Setup optimization procedure
        # self.optimizer = torch.optim.Adam(list(self.model.parameters())+[self.alpha1,self.alpha2], lr=opt['lr'])
        # +[self.alpha1, self.alpha2]
        self.optimizer1 = torch.optim.Adam(list(self.model.reconstructor.parameters()) + list(self.model.category_encoder.parameters()) + list(self.model.domain_encoder.parameters()) ,lr=opt['lr'] )
        self.optimizer2 = torch.optim.Adam(list(self.model.category_classifier.parameters()) + list(self.model.domain_classifier.parameters())+ list(self.model.feature_extractor.parameters()),lr=opt['lr'])
        # print("model parameters: ",self.model.parameters())
        # print("criterion parameters: ",self.criterion.parameters())
    def entropy_loss(self, f):
        return -torch.sum(-F.softmax(f,1)*F.log_softmax(f,1),1).mean()
    def save_checkpoint(self, path, epoch, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['epoch'] = epoch
        checkpoint['iteration'] = iteration  # 当前第几个iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer1'] = self.optimizer1.state_dict()
        checkpoint['optimizer2'] = self.optimizer2.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        epoch = checkpoint['epoch']
        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer1.load_state_dict(checkpoint['optimizer1'])
        self.optimizer2.load_state_dict(checkpoint['optimizer2'])

        return epoch, iteration, best_accuracy, total_train_loss

    def freezeLayer(self, layerName, setState):
        for param in layerName.parameters():
            param.requires_grad = not setState
    def unfreezeAll(self):
        for param in self.model.parameters():
            param.requires_grad = True
    def print_calculated_paramters_name(self): # backward() 后调用，输出所有被计算梯度的参数的名字
        print("calculated parameters: [ ")
        for name,p in self.model.named_parameters():
            if p.grad != None:
                print(name)
        print(" ]")
    def print_parameters_can_be_calculate(self): # 在backward() 前调用，输出可以被计算梯度的参数的名字
        print("parameters can be compute: [ ")
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(name)
        print(" ]")
    def train_iteration(self, data_source, data_target):
        # [x]: source/target的图
        # [y]:  category label(狗，猫...)
        # [yd]: domain label (cartoon, photo...)

        x_s, y_s, yd_s = data_source
        x_t, _, yd_t = data_target

        x_s = x_s.to(self.device)
        y_s = y_s.to(self.device)
        yd_s = yd_s.to(self.device)

        x_t = x_t.to(self.device) # [32,3,224,224] 32是一个batch中图片数量
        yd_t = yd_t.to(self.device)

        # feature_extractor, domain_encoder, category_encoder, domain_classifier, category_classifier, reconstructor
        fG1, fG_hat1, Cfcs1, DCfcs1, DCfds1, Cfds1 = self.model(x_s)
        fG2, fG_hat2, _, DCfcs2, DCfds2, Cfds2 = self.model(x_t)

        # train reconstructor + category_encoder + domain_encoder
        self.freezeLayer(self.model.category_classifier, True)
        self.freezeLayer(self.model.domain_classifier, True)

        l_class_ent_1 = self.entropy_loss(DCfcs1) # train category_encoder 1 remove domain information from category encoder
        l_class_ent_2 = self.entropy_loss(DCfcs2) # train category_encoder 2
        l_domain_ent_1 = self.entropy_loss(Cfds1) # train domain_encoder 1 remove category information from domain encoder
        l_domain_ent_2 = self.entropy_loss(Cfds2) # train domain_encoder 2
        # l_domain_ent_1 = self.cross_entropy(Cfds1,y_s)
        # print((-l_class_ent_1 - l_class_ent_2).item(), (-l_domain_ent_1 - l_domain_ent_2).item())
        l_class_ent = -l_class_ent_1 - l_class_ent_2
        l_domain_ent = -l_domain_ent_1 - l_domain_ent_2

        l_rec_1 = self.rec_loss(fG1, fG_hat1) # train reconstructor 1
        l_rec_2 = self.rec_loss(fG2, fG_hat2) # train reconstructor 2
        L_rec = l_rec_1 + l_rec_2
        L1 = self.w3 * L_rec + \
             self.w1 * self.alpha1 * (l_class_ent_1 + l_class_ent_2) + \
             self.w2 * self.alpha2 * (l_domain_ent_1 + l_domain_ent_2 )
        self.optimizer1.zero_grad()
        L1.backward(retain_graph=True) # 计算了category_encoder + domain_encoder + reconstructor的梯度
        # self.optimizer.param_groups[0]['params'] = [p for p in self.model.parameters() if p.requires_grad]

        self.freezeLayer(self.model.category_classifier, False)
        self.freezeLayer(self.model.domain_classifier, False)
        self.freezeLayer(self.model.reconstructor, True)
        # train [domain_classifier]
        l_domain_1 = self.cross_entropy(DCfds1, yd_s)
        l_domain_2 = self.cross_entropy(DCfds2, yd_t)
        l_domain = l_domain_1 + l_domain_2
        # train [category_classifier]
        l_class = self.cross_entropy(Cfcs1, y_s)
        L2 = self.w1 * l_class + self.w2 * l_domain
        self.optimizer2.zero_grad() # 清空 category_classifier + domain_classifier
        # category_encoder+category_classifier虽然没有计算梯度，但上一次保留了计算图，所以结果还在，这里清空只是变成0，并不是None，所以虽然requires_grad设成false 还是可能更新梯度，要在step前面从optimizer中踢出
        L2.backward()  #feature_extractor +  domain_encoder_category_encoder+reconstructor  （用到的某个值被前面step()更新了 会有inplace operation错误)
        # self.optimizer.param_groups[0]['params'] = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer1.step()  # 更新了:  feature_extractor + category_encoder + domain_encoder + reconstructor + alpha1 + alpha2
        self.optimizer2.step()  # 更新了： category_classifier + domain_classifier
        self.unfreezeAll()
        # loss = self.w1 * L_class + self.w2 * L_domain + self.w3 * L_rec
        loss = L1+L2
        # print(self.alpha1.item(),self.alpha2.item())
        return loss.item(),l_class_ent.item(), l_domain_ent.item(), l_class.item(), l_domain.item(), L_rec.item()

    def validate(self, loader):
        self.model.eval()  # 设置为evaluation 模式
        accuracy = 0
        count = 0
        loss = 0
        dom_accuracy = 0
        with torch.no_grad():  # 禁用梯度计算，即使torch.tensor(xxx,requires_grad = True) 使用.requires_grad()也会返回False
            for x, y, yd in loader:  # type(x) tensor

                y = y.to(self.device)
                x = x.to(self.device)
                yd = yd.to(self.device)

                fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds = self.model(x)

                loss += self.cross_entropy(Cfcs, y)

                pred = torch.argmax(Cfcs, dim=-1)
                dom_pred = torch.argmax(DCfds, dim=-1)
                accuracy += (pred == y).sum().item()
                dom_accuracy += (dom_pred == yd).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        mean_dom_loss = dom_accuracy / count
        self.model.train()
        return mean_accuracy, mean_loss, mean_dom_loss