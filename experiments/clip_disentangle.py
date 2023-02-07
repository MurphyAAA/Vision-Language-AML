import torch
import clip
from models.base_model import CLIPDisentangleModel
import torch.nn.functional as F

import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

class CLIPDisentangleExperiment:  # See point 4. of the project

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # load CLIP and freeze it. vision Transformer base
        if opt['train_clip'] =='True':
            self.clip_model, self.preprocess = clip.load('ViT-B/32',device='cpu',jit=False)  #Must set jit=False for training
        else:
            self.clip_model, self.preprocess = clip.load('ViT-B/32',device='cpu') # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model = self.clip_model.to(self.device)
        self.freeze_clip()
        # image = preprocess().unsqueeze(0).to(self.device) # 将PIL.image变成可以作为模型输入的tensor
        # text = clip.tokenize(["xxx","sss","eee"]).to(self.device)
        # img_feature = self.clip_model.encode_image(image)
        # text_feature = self.clip_model.encode_text(text)
        # logits_per_image,logits_per_text = self.clip_model(image,text)  # 传进model中也会将image和text传入encoder中编码成feature，即上面两步，不过这里就是验证，所以直接传的图片和text，由内部调用encoder
        # logits_per_image 第i行是第i个图，每一列是属于这个对应词汇类别的概率 [i] 0.3 0.4 0.1 0.2  所以属于第2类 (要手动对logits_per_image求个softmax)
        self.model = CLIPDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Loss functions
        self.clip_loss = torch.nn.MSELoss()
        # self.nll_loss = torch.nn.NLLLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.rec_loss = torch.nn.MSELoss()
        # hyper parameters
        self.alpha1 = 1.2
        self.alpha2 = 0.5
        self.w = [2, 1, 1, 0.5]
        # Setup optimization procedure
        params1 = list(self.model.reconstructor.parameters()) + list(self.model.category_encoder.parameters()) + list(self.model.domain_encoder.parameters())
        self.optimizer1 = torch.optim.Adam(params1 , lr=opt['lr'])

        params2 = list(self.model.domain_classifier.parameters()) + list(self.model.category_classifier.parameters())+ list(self.model.feature_extractor.parameters())
        self.optimizer2 = torch.optim.Adam(params2, lr=opt['lr'])

        self.clip_optimizer = torch.optim.Adam(self.clip_model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)#Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    def convert_models_to_fp32(self):
        for p in self.clip_model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()
    def freeze_clip(self):
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
    def unfreeze_clip(self):
        self.clip_model.train()
        for param in self.clip_model.parameters():
            param.requires_grad = True

    def entropy_loss(self, f):  # 应该返回一个标量 最后是求和的
        # mlogf = torch.mean(f, dim=0)
        # res = -torch.mean(mlogf)
        return -torch.sum(-F.softmax(f, 1) * F.log_softmax(f, 1), 1).mean()

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

    def save_clip_checkpoint(self, path, clip_iteration, total_clip_loss):
        checkpoint = {}
        # checkpoint['epoch'] = epoch
        checkpoint['clip_iteration'] = clip_iteration  # 当前第几个iteration
        checkpoint['total_clip_loss'] = total_clip_loss
        checkpoint['clip_model'] = self.clip_model.state_dict()
        checkpoint['clip_optimizer'] = self.clip_optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_clip_checkpoint(self, path):
        checkpoint = torch.load(path)
        clip_iteration = checkpoint['clip_iteration']
        total_clip_loss = checkpoint['total_clip_loss']
        # checkpoint['model_state_dict']["input_resolution"] = self.clip_model.input_resolution  # default is 224
        # checkpoint['model_state_dict']["context_length"] = self.clip_model.context_length  # default is 77
        # checkpoint['model_state_dict']["context_length"] = 300
        # checkpoint['model_state_dict']["vocab_size"] = self.clip_model.vocab_size
        self.clip_model.load_state_dict(checkpoint['clip_model'])
        self.clip_optimizer.load_state_dict(checkpoint['clip_optimizer'])

        return clip_iteration, total_clip_loss


    def freezeLayer(self, layerName, setState):
        for param in layerName.parameters():
            param.requires_grad = not setState

    def unfreezeAll(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def print_calculated_paramters_name(self):  # backward() 后调用，输出所有被计算梯度的参数的名字
        print("calculated parameters: [ ")
        for name, p in self.model.named_parameters():
            if p.grad != None:
                print(name)
        print(" ]")

    def print_parameters_can_be_calculate(self):  # 在backward() 前调用，输出可以被计算梯度的参数的名字
        print("parameters can be compute: [ ")
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(name)
        print(" ]")

    def clip_train_iteration(self,databatch):
        self.clip_optimizer.zero_grad()
        x, desc = databatch
        x = x.to(self.device)
        tokenized_desc = clip.tokenize(desc,truncate=True).to(self.device)

        logits_per_image, logits_per_text = self.clip_model(x, tokenized_desc)

        ground_truth = torch.arange(len(x), dtype=torch.long, device=self.device)

        total_loss = (self.cross_entropy(logits_per_image, ground_truth) + self.cross_entropy(logits_per_text, ground_truth)) / 2
        # print("clip loss: ",total_loss)
        total_loss.backward()
        if self.device == "cpu":
            self.clip_optimizer.step()
        else:
            self.convert_models_to_fp32()
            self.clip_optimizer.step()
            clip.model.convert_weights(self.clip_model) # mixed precision training. Convert applicable model parameters to fp16
        return total_loss.item()
    def train_iteration(self, data_source, data_target):

        x_s, y_s, yd_s, desc_s = data_source  # desc可能是-1
        x_t, _, yd_t, desc_t = data_target

        i1 = [i for i in range(len(desc_s)) if desc_s[i] != '-1'] # desc_s中值不是-1的索引
        i2 = [i for i in range(len(desc_t)) if desc_t[i] != '-1'] # desc_t中值不是-1的索引

        x_s = x_s.to(self.device)
        y_s = y_s.to(self.device)
        yd_s = yd_s.to(self.device)
        textToken_s = clip.tokenize(desc_s, truncate=True).to(self.device)
        textToken_s = textToken_s[i1]  # description有效的行

        x_t = x_t.to(self.device)  # [32,3,224,224] 32是一个batch中图片数量
        yd_t = yd_t.to(self.device)
        textToken_t = clip.tokenize(desc_t, truncate=True).to(self.device)
        textToken_t = textToken_t[i2]

        # print(len(y_s), len(desc_s), len(desc_t))
        # feature_extractor, domain_encoder, category_encoder, domain_classifier, category_classifier, reconstructor
        fG1, fG_hat1, Cfcs1, DCfcs1, DCfds1, Cfds1, fds1 = self.model(x_s)
        # print(fds1.size()) 32 * 512
        fG2, fG_hat2, _, DCfcs2, DCfds2, Cfds2, fds2 = self.model(x_t)

        text_feature_s = self.clip_model.encode_text(textToken_s)
        text_feature_t = self.clip_model.encode_text(textToken_t)
        # print(text_feature_s.size())
        # print(text_feature_t.size())

        # train reconstructor + category_encoder + domain_encoder
        self.freezeLayer(self.model.category_classifier, True)
        self.freezeLayer(self.model.domain_classifier, True)

        l_class_ent_1 = self.entropy_loss(DCfcs1)  # train category_encoder 1 remove domain info from category encoder
        l_class_ent_2 = self.entropy_loss(DCfcs2)  # train category_encoder 2
        l_domain_ent_1 = self.entropy_loss(Cfds1)  # train domain_encoder 1
        l_domain_ent_2 = self.entropy_loss(Cfds2)  # train domain_encoder 2
        l_class_ent = (l_class_ent_1 + l_class_ent_2)
        l_domain_ent = (l_domain_ent_1 + l_domain_ent_2)

        # fds1 = fds1[i1]
        if len(i1) >0:
            l_clip1 = self.clip_loss(text_feature_s, fds1[i1])
        else:
            l_clip1 = 0
        if len(i2) > 0:
            l_clip2 = self.clip_loss(text_feature_t, fds2[i2])
        else:
            l_clip2 = 0
        L_clip = l_clip1 + l_clip2
        # print("CLIP loss", L_clip.item())
        l_rec_1 = self.rec_loss(fG1, fG_hat1)  # train reconstructor 1
        l_rec_2 = self.rec_loss(fG2, fG_hat2)  # train reconstructor 2
        L_rec = l_rec_1 + l_rec_2
        # print(L_clip)
        L1 = self.w[2] * L_rec + self.w[3] * L_clip + \
             self.w[0] * self.alpha1 * (l_class_ent) + \
             self.w[1] * self.alpha2 * (l_domain_ent)
        self.optimizer1.zero_grad()
        L1.backward(retain_graph=True)  #  category_encoder + domain_encoder + reconstructor的梯度

        # train [domain_classifier]
        self.freezeLayer(self.model.category_classifier, False)
        self.freezeLayer(self.model.domain_classifier, False)
        self.freezeLayer(self.model.reconstructor, True)

        # train [domain_classifier]
        l_domain_1 = self.cross_entropy(DCfds1, yd_s)
        l_domain_2 = self.cross_entropy(DCfds2, yd_t)
        l_domain = l_domain_1 + l_domain_2
        # train [category_classifier]
        l_class = self.cross_entropy(Cfcs1, y_s)
        L2 = self.w[0] * l_class + self.w[1] * l_domain
        self.optimizer2.zero_grad()  # clean following grad: category_classifier + domain_classifier
        # category_encoder+category_classifier虽然没有计算梯度，但上一次保留了计算图，所以结果还在，这里清空只是变成0，并不是None，所以虽然requires_grad设成false 还是可能更新梯度，要在step前面从optimizer中踢出
        L2.backward()  # feature_extractor + domain_encoder + category_encoder + reconstructor  （用到的某个值被前面step()更新了 会有inplace operation错误)

        self.optimizer1.step()  # update following layers:  feature_extractor + category_encoder + domain_encoder + reconstructor + alpha1 + alpha2
        self.optimizer2.step()  # update following layers： category_classifier + domain_classifier
        self.unfreezeAll()
        loss = L1 + L2
        return loss.item(),l_class.item(), l_class_ent.item(), l_domain.item(), l_domain_ent.item(), L_rec.item(), L_clip.item()

    def validate(self, loader):
        self.clip_model.eval()
        self.model.eval()  # 设置为evaluation 模式
        accuracy = 0
        count = 0
        loss = 0
        dom_acc = 0

        with torch.no_grad():  # 禁用梯度计算，即使torch.tensor(xxx,requires_grad = True) 使用.requires_grad()也会返回False
            # flag=True
            for x, y, yd, _ in loader:  # type(x) tensor x,y,yd,description
                # train_transform = transforms.Compose([
                #     transforms.Resize(256),
                #     transforms.RandAugment(3, 15),
                #     transforms.CenterCrop(224),
                #     transforms.ToTensor(),
                #     transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # ])
                # print(x.shape)

                y = y.to(self.device)
                x = x.to(self.device)
                yd = yd.to(self.device)

                _, _, Cfcs, _, DCfds, _,_ = self.model(x)

                # if flag:
                #     print(F.softmax(Cfcs[0]))
                #     img = transform_invert(x[0].cpu(), train_transform)
                #     img.show()
                #     flag = False
                loss += self.cross_entropy(Cfcs, y)
                dom_pred = torch.argmax(DCfds, dim=-1)
                pred = torch.argmax(Cfcs, dim=-1)
                accuracy += (pred == y).sum().item()
                dom_acc += (dom_pred == yd).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        mean_dom_acc = dom_acc / count
        self.model.train()
        return mean_accuracy, mean_loss,mean_dom_acc


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_
