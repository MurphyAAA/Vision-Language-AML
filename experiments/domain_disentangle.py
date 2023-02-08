import torch
from models.base_model import DomainDisentangleModel
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
        self.w1 = 2
        self.w2 = 1
        self.w3 = 1 # 2,1,1
        # Setup optimization procedure
        self.optimizer1 = torch.optim.Adam(list(self.model.reconstructor.parameters()) + list(self.model.category_encoder.parameters()) + list(self.model.domain_encoder.parameters()) ,lr=opt['lr'] )
        self.optimizer2 = torch.optim.Adam(list(self.model.category_classifier.parameters()) + list(self.model.domain_classifier.parameters())+ list(self.model.feature_extractor.parameters()),lr=opt['lr'])
        self.optimizer3 = torch.optim.Adam( list(self.model.category_encoder.parameters()) + list(self.model.domain_encoder.parameters()) ,lr=opt['lr'] )
    def entropy_loss(self, f):
        res = -(torch.sum(-F.softmax(f,1)*F.log_softmax(f,1),1).mean())
        return res
    def save_checkpoint(self, path, epoch, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['epoch'] = epoch
        checkpoint['iteration'] = iteration
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
    def train_iteration(self, data_source, data_target, iteration):
        # [x]: source/target image
        # [y]:  category label(dog, elephant...)
        # [yd]: domain label (cartoon, photo...)

        x_s, y_s, yd_s = data_source
        x_t, _, yd_t = data_target

        x_s = x_s.to(self.device)
        y_s = y_s.to(self.device)
        yd_s = yd_s.to(self.device)

        x_t = x_t.to(self.device) # [32,3,224,224]
        yd_t = yd_t.to(self.device)

        # feature_extractor, domain_encoder, category_encoder, domain_classifier, category_classifier, reconstructor
        fG1, fG_hat1, Cfcs1, DCfcs1, DCfds1, Cfds1 = self.model(x_s)
        fG2, fG_hat2, _, DCfcs2, DCfds2, Cfds2 = self.model(x_t)
        # train [reconstructor + category_encoder + domain_encoder]
        self.freezeLayer(self.model.category_classifier, True)
        self.freezeLayer(self.model.domain_classifier, True)

        l_class_ent_1 = self.entropy_loss(DCfcs1) # train category_encoder 1 remove domain information from category encoder
        l_class_ent_2 = self.entropy_loss(DCfcs2) # train category_encoder 2
        l_domain_ent_1 = self.entropy_loss(Cfds1) # train domain_encoder 1 remove category information from domain encoder
        l_domain_ent_2 = self.entropy_loss(Cfds2) # train domain_encoder 2
        l_class_ent = -l_class_ent_1 - l_class_ent_2
        l_domain_ent = -l_domain_ent_1 - l_domain_ent_2

        l_rec_1 = self.rec_loss(fG1, fG_hat1) # train reconstructor 1
        l_rec_2 = self.rec_loss(fG2, fG_hat2) # train reconstructor 2
        L_rec = l_rec_1 + l_rec_2
        L1 = self.w3 * L_rec + \
             self.w1 * self.alpha1 * (l_class_ent_1 + l_class_ent_2) + \
             self.w2 * self.alpha2 * (l_domain_ent_1 + l_domain_ent_2)
        self.optimizer1.zero_grad()
        self.optimizer3.zero_grad()
        L1.backward(retain_graph=True) # category_encoder + domain_encoder + reconstructor

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
        self.optimizer2.zero_grad() # clear category_classifier + domain_classifier
        L2.backward()
        if iteration % 40 == 0 :
            self.optimizer1.step()  # [category_encoder + domain_encoder + reconstructor]
            self.optimizer2.step()  # [feature extractor + category_classifier + domain_classifier]
            self.optimizer3.step()
        else:
            self.optimizer3.step()
        self.unfreezeAll()
        loss = L1+L2
        return loss.item(),l_class_ent.item(), l_domain_ent.item(), l_class.item(), l_domain.item(), L_rec.item()

    def validate(self, loader):
        self.model.eval()  # evaluation mode
        accuracy = 0
        count = 0
        loss = 0
        dom_accuracy = 0
        with torch.no_grad():
            for x, y, yd in loader:  # type(x): tensor

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
        mean_dom_accuracy = dom_accuracy / count
        self.model.train()
        return mean_accuracy, mean_loss, mean_dom_accuracy