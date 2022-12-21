import numpy as np
import matplotlib.pyplot as plt
import torch as torch

# x = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/data/motionChannel1.npy')
# x = x[0:2, :, :]
# x = x - 20
# x = np.expand_dims(x,1)
# x_ab = torch.from_numpy(x)

x_ab = torch.rand((2,3)) - 0.4
tem = torch.ones_like(x_ab)
ind = torch.where(x_ab > 0,x_ab,tem)

img_log = torch.log(ind)
out = ind * img_log
E = out.sum() / (256 * 256)

# loss_ContentD = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR/output_tem2/loss/loss_ContentD.npy')
loss_gen_adv_a = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR_Mat2/output_vgg_without_l1/loss/loss_gen_adv_a.npy')
loss_gen_adv_b = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR_Mat2/output_vgg_without_l1/loss/loss_gen_adv_b.npy')
loss_gen_cyc_x_a = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR_Mat2/output_vgg_without_l1/loss/loss_gen_cyc_x_a.npy')
loss_gen_cyc_x_b = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR_Mat2/output_vgg_without_l1/loss/loss_gen_cyc_x_b.npy')
loss_gen_recon_x_a = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR_Mat2/output_vgg_without_l1/loss/loss_gen_recon_x_a.npy')
loss_gen_recon_x_b = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR_Mat2/output_vgg_without_l1/loss/loss_gen_recon_x_b.npy')
loss_gen_total = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR_Mat2/output_vgg_without_l1/loss/loss_gen_total.npy')
# loss_gen_vgg_a = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR/output_tem2/loss/loss_gen_vgg_a.npy')
# loss_gen_vgg_b = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR/output_tem2/loss/loss_gen_vgg_b.npy')
loss_tem1 = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR_Mat2/output_vgg_without_l1/loss/my_sum_loss.npy')
# loss_tem2 = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR/output_tem/loss/loss_tem2.npy')
# loss_tem3 = np.load('/home/user/data4/chongxin/LIR-for-Unsupervised-IR/output_tem/loss/loss_tem3.npy')


# plt.plot(loss_bgm)
# plt.title("Loss BC")
# plt.show()
#
# plt.plot(loss_ContentD)
# plt.title("Loss adv R")
# plt.show()

plt.plot(loss_gen_adv_a)
plt.title("Loss adv x")
plt.show()
plt.plot(loss_gen_adv_b)
plt.title("Loss adv y")
plt.show()

plt.plot(loss_gen_cyc_x_a)
plt.title("Loss CC x")
plt.show()
plt.plot(loss_gen_cyc_x_b)
plt.title("Loss CC y")
plt.show()

plt.plot(loss_gen_recon_x_a)
plt.title("Loss Rec x")
plt.show()
plt.plot(loss_gen_recon_x_b)
plt.title("Loss Rec y")
plt.show()

# plt.plot(loss_gen_vgg_a+loss_gen_vgg_b)
# plt.title("Loss SC/VGG")
# plt.show()

plt.plot(loss_tem1)
plt.title("Loss Encoder:Noise Content")
plt.show()

# plt.plot(loss_tem2)
# plt.title("Loss Noise")
# plt.show()
#
# plt.plot(loss_tem3)
# plt.title("Loss Rec: a aba")
# plt.show()

plt.plot(loss_gen_total)
plt.title("Loss Total")
plt.show()