import torch
import clip


# TODO
class CLIPDisentangleExperiment:  # See point 4. of the project

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # setup train
        self.clip_model, _ = clip.load('ViT-B/32',
                                       device='cpu')  # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model = self.clip_model.to(self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.model = CLIPDisentangleExperiment()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Loss functions
        self.clip_loss = torch.nn.MSELoss()
        self.nll_loss = torch.nn.NLLLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.rec_loss = torch.nn.MSELoss()
        self.alpha1 = torch.nn.Parameter(torch.tensor(0.1, device='cuda'), requires_grad=True)
        self.alpha2 = torch.nn.Parameter(torch.tensor(0.1, device='cuda'), requires_grad=True)
        self.w1 = 1  # 主要训练category分类器 所以他的权重高一点，其他权重低一点
        self.w2 = 1
        self.w3 = 1
        # Setup optimization procedure
        params1 = list(self.model.domain_encoder.parameters()) + list(self.model.category_encoder.parameters()) + list(
            self.model.reconstructor.parameters())
        self.opt1 = torch.optim.Adam(params1 + [self.alpha1, self.alpha2], lr=opt['lr'])

        params2 = list(self.model.domain_classifier.parameters()) + list(self.model.category_classifier.parameters())
        self.opt2 = torch.optim.Adam(params2, lr=opt['lr'])

    def entropy_loss(self, f):  # 应该返回一个标量 最后是求和的

        # f = torch.clamp_min_(f,0.0001)
        # print(f,)
        logf = torch.log(f)
        mlogf = logf.mean(dim=0)
        return -mlogf.sum()

    def save_checkpoint(self, path, epoch, iteration, best_accuracy, total_train_loss):
        checkpoint = {}
        checkpoint['epoch'] = epoch
        checkpoint['iteration'] = iteration  # 当前第几个iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = [self.opt1.state_dict(), self.opt2.state_dict()]
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

    def train_iteration(self, data_source, data_target):

        x_s, y_s, yd_s = data_source
        x_t, _, yd_t = data_target
        x_s = x_s.to(self.device)
        y_s = y_s.to(self.device)
        yd_s = yd_s.to(self.device)
        x_t = x_t.to(self.device)  # [32,3,224,224] 32是一个batch中图片数量
        yd_t = yd_t.to(self.device)

        # x = torch.cat((x_s,x_t),0) # [64,3,224,224]
        # yd = torch.cat((yd_s,yd_t),0) # [64] 前32个是source domain label， 后32个是target

        # -------train source domain 部分
        fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds = self.model(x_s)
        # loss = self.criterion(fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds, y_s, yd)  # 这里要重新写！！！ 不能 直接模型处理完结果传到损失函数里，损失函数可能用的总体模型中不同阶段的输出，而不是最终整体的输出！！！！
        l_class = self.cross_entropy(Cfcs, y_s)
        l_class_ent_1 = self.entropy_loss(DCfcs)  # 没变化

        l_domain_1 = self.cross_entropy(DCfds, yd_s)
        l_domain_ent_1 = self.entropy_loss(Cfds)

        l_rec_1 = self.rec_loss(fG, fG_hat)

        l_clip_1 = self.clip_loss(fds, clip_out)
        description = 'a picture of a small dog playing with a ball'
        tokenized_text = clip.tokenize(description).to(self.device)
        text_features = self.clip_model.encode_text(tokenized_text)

        # -------train target domain 部分
        fG, fG_hat, _, _, DCfds, Cfds = self.model(x_t)

        # l_class_ent_2 = self.entropy_loss(DCfcs) #注释掉这个
        L_class = l_class + self.alpha1 * (l_class_ent_1)

        l_domain_2 = self.cross_entropy(DCfds, yd_t)
        l_domain = l_domain_1 + l_domain_2

        l_domain_ent_2 = self.entropy_loss(Cfds)  # 会变成0
        L_domain = l_domain + self.alpha2 * (l_domain_ent_1 + l_domain_ent_2)

        l_rec_2 = self.rec_loss(fG, fG_hat)
        L_rec = l_rec_1 + l_rec_2

        loss = self.w1 * L_class + self.w2 * L_domain + self.w3 * L_rec

        self.opt1.zero_grad()
        # dom_enc + cat_enc + rec
        loss1 = self.w1 * self.alpha1 * (l_class_ent_1) + self.w2 * self.alpha2 * (
                    l_domain_ent_1 + l_domain_ent_2) + self.w3 * L_rec
        loss1.backward(retain_graph=True)

        self.opt2.zero_grad()
        # dom_clr + cat_clr 
        loss2 = self.w1 * l_class + self.w2 * l_domain
        loss2.backward()

        self.opt1.step()
        self.opt2.step()
        # print(self.model.category_classifier[0].weight.grad)
        # for name,param in self.model.named_parameters():
        #     print(name)
        loss = loss1 + loss2
        return loss.item()

    def validate(self, loader):
        self.clip_model.eval()
        self.model.eval()  # 设置为evaluation 模式
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():  # 禁用梯度计算，即使torch.tensor(xxx,requires_grad = True) 使用.requires_grad()也会返回False
            for x, y, yd in loader:  # type(x) tensor

                y = y.to(self.device)
                x = x.to(self.device)
                yd = yd.to(self.device)

                fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds = self.model(x)
                # opt = parse_arguments()
                # Cfcs = Cfcs[0:opt['batch_size'], :]
                # loss += self.criterion(fG, fG_hat, Cfcs, DCfcs, DCfds, Cfds, y,yd)  # 这里要重新写！！！ 不能 直接模型处理完结果传到损失函数里，损失函数可能用的总体模型中不同阶段的输出，而不是最终整体的输出！！！！

                loss += self.cross_entropy(Cfcs, y)
                # L_domain = self.nll_loss(torch.log(DCfds), yd)  # + self.alpha2*self.entropy_loss(Cfds)
                # L_rec = self.rec_loss(fG, fG_hat)
                # loss += self.w1 * L_class + self.w2 * L_domain + self.w3 * L_rec

                pred = torch.argmax(Cfcs, dim=-1)
                # print(Cfcs.shape)
                accuracy += (pred == y).sum().item()
                count += x.size(0)
                # print(count)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss