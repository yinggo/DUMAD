# The CODE is implemented for unir, which is updated based on "UNIT" (NIPS 2016)
# author: Wenchao. Du

from networks import  MsImageDis, Dis_content, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg19, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
from GaussianSmoothLayer import GaussionSmoothLayer, GradientLoss
import os
from skimage.measure import  compare_ssim
import numpy as np

class UNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(UNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.gpuid = hyperparameters['gpuID']
        # @ add backgound discriminator for each domain
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.gen_a.apply(weights_init(hyperparameters['init']))
        self.gen_b.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # # Load VGG model if needed for test
        # if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
        #     self.vgg = load_vgg19()
        #     if torch.cuda.is_available():
        #         self.vgg.cuda(self.gpuid)
        #     self.vgg.eval()
        #     for param in self.vgg.parameters():
        #         param.requires_grad = False
        self.vgg = load_vgg19()
        if torch.cuda.is_available():
            self.vgg.cuda(self.gpuid)
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def recon_criterion(self, input, target):
        loss1 = torch.mean(torch.abs(input - target))
        loss2 = self.compute_vgg_loss(self.vgg, input, target)
        # return loss1+0.65*loss2
        return loss1+0.5*loss2
        # return loss1
        # return self.compute_vgg_loss(self.vgg, input, target)

    def ssim_criterion(self, input, target):
        loss_ssim = 0
        for index in range(input.shape[0]):
            input_tem  =  input[index,:,:,:].squeeze().data.cpu().numpy()
            target_tem  =  target[index,:,:,:].squeeze().data.cpu().numpy()
            loss_ssim += 1 - compare_ssim(input_tem,target_tem)

        return loss_ssim/input.shape[0]

    def forward(self, x_a, x_b):
        self.eval()
        h_a = self.gen_a.encode_cont(x_a)
        # h_a_sty = self.gen_a.encode_sty(x_a)
        # h_b = self.gen_b.encode_cont(x_b)
        
        x_ab = self.gen_b.decode_cont(h_a)
        # h_c = torch.cat((h_b, h_a_sty), 1)
        # x_ba = self.gen_a.decode_recs(h_c)
        # self.train()
        return x_ab #, x_ba

    # def gen_update(self, x_a, x_b, hyperparameters):
    # def gen_update(self, x_a, x_b, hyperparameters, loss_gen_adv_a, loss_gen_adv_b, loss_gen_recon_x_a,
    #                    loss_gen_recon_kl_a, loss_gen_recon_kl_b, loss_gen_recon_kl_sty, loss_gen_recon_kl_cyc_aba,
    #                    loss_gen_recon_kl_cyc_bab, loss_gen_recon_kl_cyc_sty, loss_gen_recon_x_b, loss_gen_cyc_x_a,
    #                    loss_gen_cyc_x_b, loss_gen_vgg_a, loss_gen_vgg_b, loss_bgm, loss_ContentD, loss_gen_total,
    #                    my_loss_bgm):
    def gen_update(self, x_a, x_b, hyperparameters, loss_gen_adv_a, loss_gen_adv_b, loss_gen_recon_x_a,loss_gen_recon_x_b, loss_gen_cyc_x_a,
                   loss_gen_cyc_x_b,loss_gen_total,my_sum_loss):

        self.gen_opt.zero_grad()
        # encode
        h_a = self.gen_a.encode_cont(x_a)
        h_b = self.gen_b.encode_cont(x_b)
        h_a_sty = self.gen_a.encode_sty(x_a)

        # decode (within domain)
        # h_a_cont = torch.cat((h_a, h_a_sty), 1)
        # noise_a = torch.randn(h_a_cont.size()).cuda(h_a_cont.data.get_device())
        # x_a_recon = self.gen_a.decode_recs(h_a_cont + noise_a)
        # noise_b = torch.randn(h_b.size()).cuda(h_b.data.get_device())
        # x_b_recon = self.gen_b.decode_cont(h_b + noise_b)

        h_a_cont = torch.cat((h_a, h_a_sty), 1)
        x_a_recon = self.gen_a.decode_recs(h_a_cont)
        x_b_recon = self.gen_b.decode_cont(h_b)

        # decode (cross domain)
        h_ba_cont = torch.cat((h_b, h_a_sty), 1)
        # x_ba = self.gen_a.decode_recs(h_ba_cont + noise_a)
        # x_ab = self.gen_b.decode_cont(h_a + noise_b)

        x_ba = self.gen_a.decode_recs(h_ba_cont)
        x_ab = self.gen_b.decode_cont(h_a)

        # encode again
        h_b_recon = self.gen_a.encode_cont(x_ba)
        h_b_sty_recon = self.gen_a.encode_sty(x_ba)

        h_a_recon = self.gen_b.encode_cont(x_ab)

        # decode again (if needed)
        h_a_cat_recs = torch.cat((h_a_recon, h_b_sty_recon), 1)

        # x_aba = (self.gen_a.decode_recs(h_a_cat_recs + noise_a)  ) if hyperparameters['recon_x_cyc_w'] > 0 else None
        # x_bab = (self.gen_b.decode_cont(h_b_recon + noise_b)  ) if hyperparameters['recon_x_cyc_w'] > 0 else None

        x_aba = (self.gen_a.decode_recs(h_a_cat_recs)) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = (self.gen_b.decode_cont(h_b_recon)) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)

        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a) if x_aba is not None else 0
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b) if x_aba is not None else 0

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # domain-invariant perceptual loss
        # self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        # self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        # add background guide loss
        # self.loss_bgm = 0
        #

        # self.my_entropy_loss = 0
        # # x_ab   x_bab
        #
        # for i in range(x_ab.shape[0]):
        #     input_tem = x_ab[i, :, :, :].squeeze()
        #     tem = torch.ones_like(input_tem)
        #     ind = torch.where(input_tem > 0, input_tem, tem)
        #     img_log = torch.log(ind)
        #     out = ind * img_log
        #     self.my_entropy_loss = self.my_entropy_loss - out.sum()/(128*128)
        #
        # for i in range(x_bab.shape[0]):
        #     input_tem = x_bab[i, :, :, :].squeeze()
        #     tem = torch.ones_like(input_tem)
        #     ind = torch.where(input_tem > 0, input_tem, tem)
        #     img_log = torch.log(ind)
        #     out = ind * img_log
        #     self.my_entropy_loss = self.my_entropy_loss - out.sum()/(128*128)
        #
        # self.my_entropy_loss = self.my_entropy_loss/ (x_ab.shape[0] * 2.0)


        self.my_sum_loss = 0

        for index in range(x_a.shape[0]):
            input_tem = x_a[index, :, :, :].squeeze()
            target_tem = x_ab[index, :, :, :].squeeze()
            sum_a_ori = torch.sum(input_tem, dim=0)
            sum_a_now = torch.sum(target_tem, dim=0)
            max_a_ori = torch.max(sum_a_ori)
            max_a_now = torch.max(sum_a_now)
            sum_a_ori = sum_a_ori / max_a_ori
            sum_a_now = sum_a_now / max_a_now
            self.my_sum_loss = self.my_sum_loss + torch.sum(abs(sum_a_ori - sum_a_now))

        for index in range(x_b.shape[0]):
            input_tem = x_b[index, :, :, :].squeeze()
            target_tem = x_ba[index, :, :, :].squeeze()
            sum_a_ori = torch.sum(input_tem, dim=0)
            sum_a_now = torch.sum(target_tem, dim=0)
            max_a_ori = torch.max(sum_a_ori)
            max_a_now = torch.max(sum_a_now)
            sum_a_ori = sum_a_ori / max_a_ori
            sum_a_now = sum_a_now / max_a_now
            self.my_sum_loss = self.my_sum_loss + torch.sum(abs(sum_a_ori - sum_a_now))

        self.my_sum_loss = self.my_sum_loss / (x_b.shape[0] * 2.0)


        # total loss
        # self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
        #                       hyperparameters['gan_w'] * self.loss_gen_adv_b + \
        #                       hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
        #                       hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
        #                       hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
        #                       hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
        #                       hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_sty + \
        #                       hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
        #                       hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
        #                       hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
        #                       hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
        #                       hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_sty + \
        #                       hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
        #                       hyperparameters['vgg_w'] * self.loss_gen_vgg_b + \
        #                       hyperparameters['BGM'] * self.loss_bgm + \
        #                       hyperparameters['gan_w'] * self.loss_ContentD + \
        #                       0* self.my_loss_bgm


        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              0.5*self.my_sum_loss  # 0.65*self.my_sum_loss
        # + self.my_entropy_loss.

        loss_gen_adv_a.append(hyperparameters['gan_w'] * self.loss_gen_adv_a.item())
        loss_gen_adv_b.append(hyperparameters['gan_w'] * self.loss_gen_adv_b.item())
        loss_gen_recon_x_a.append(hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a.item())
        loss_gen_recon_x_b.append(hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b.item())
        loss_gen_cyc_x_a.append(hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a.item())
        loss_gen_cyc_x_b.append(hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b.item())
        # loss_gen_vgg_a.append(hyperparameters['vgg_w'] * self.loss_gen_vgg_a.item())
        # loss_gen_vgg_b.append(hyperparameters['vgg_w'] * self.loss_gen_vgg_b.item())
        loss_gen_total.append(self.loss_gen_total.item())
        # my_sum_loss.append(0.65 * self.my_sum_loss.item())
        my_sum_loss.append(0.5 * self.my_sum_loss.item())
        # my_entropy_loss.append(  self.my_entropy_loss.item())

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):  #torch.Size([2, 1, 64, 64])  torch.Size([2, 1, 64, 64])
        img_vgg = img.repeat(1, 3, 1, 1)
        target_vgg = target.repeat(1, 3, 1, 1)
        # img_vgg = vgg_preprocess(img)
        # target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)  #torch.Size([2, 512, 8, 8])
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):
        if x_a is None or x_b is None:
            return None
        self.eval()
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a = self.gen_a.encode_cont(x_a[i].unsqueeze(0))
            h_a_sty = self.gen_a.encode_sty(x_a[i].unsqueeze(0))
            h_b = self.gen_b.encode_cont(x_b[i].unsqueeze(0))

            h_ba_cont = torch.cat((h_b, h_a_sty), 1)

            h_aa_cont = torch.cat((h_a, h_a_sty), 1)

            x_a_recon.append(self.gen_a.decode_recs(h_aa_cont) )
            x_b_recon.append(self.gen_b.decode_cont(h_b) )

            x_ba.append(self.gen_a.decode_recs(h_ba_cont))
            x_ab.append(self.gen_b.decode_cont(h_a))
            
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a = self.gen_a.encode_cont(x_a)
        h_a_sty = self.gen_a.encode_sty(x_a)
        h_b = self.gen_b.encode_cont(x_b)

        # decode (cross domain)
        # h_cat = torch.cat((h_b, h_a_sty), 1)
        # noise_b = torch.randn(h_cat.size()).cuda(h_cat.data.get_device())
        # x_ba = self.gen_a.decode_recs(h_cat + noise_b)
        # noise_a = torch.randn(h_a.size()).cuda(h_a.data.get_device())
        # x_ab = self.gen_b.decode_cont(h_a + noise_a)

        h_cat = torch.cat((h_b, h_a_sty), 1)
        x_ba = self.gen_a.decode_recs(h_cat)
        x_ab = self.gen_b.decode_cont(h_a)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

        self.loss_dis_total = hyperparameters['gan_w'] * (self.loss_dis_a + self.loss_dis_b )
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis_00188000")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b']) 
        
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)

        #  opt state
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)