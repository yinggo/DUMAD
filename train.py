"""
# training LIR for UNIR
# The codes is implemented by "UNIT", double encoder branches and self-supervised contraints are added for training.
# Author: Wenchao. Du
# Time: 2019. 08
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer, data_prefetcher,get_dataloaders
import argparse
from torch.autograd import Variable
from trainer import UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
# from torch.utils.tensorboard import SummaryWriter
import shutil
import scipy.io as sio
import random

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/unit_noise2clear-bn.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='output_sum_ablation0.5vgg', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

def main():

    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config['max_iter']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path

    # Setup model and data loader
    trainer = UNIT_Trainer(config)
    if torch.cuda.is_available():
        trainer.cuda(config['gpuID'])

    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_dataloaders(config)

    # train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    
    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    loss_path = opts.output_path + "/loss/"
    if not os.path.exists(loss_path):
        print("Creating directory: {}".format(loss_path))
        os.makedirs(loss_path)
    print('start training !!')
    # Start training
    iterations =  0
    
    TraindataA = data_prefetcher(train_loader_a,config)
    TraindataB = data_prefetcher(train_loader_b,config)
    testdataA = data_prefetcher(test_loader_a,config)
    testdataB = data_prefetcher(test_loader_b,config)

    loss_gen_adv_a = []
    loss_gen_adv_b = []
    loss_gen_recon_x_a = []
    loss_gen_recon_x_b = []
    loss_gen_cyc_x_a = []
    loss_gen_cyc_x_b = []
    # loss_gen_vgg_a = []
    # loss_gen_vgg_b = []
    loss_gen_total = []
    my_sum_loss = []
    # my_entropy_loss = []

    while True:
        dataA = TraindataA.next()  #torch.Size([2, 1, 64, 64]) torch.float64
        dataB = TraindataB.next()  #torch.Size([2, 1, 64, 64]) torch.float64
        if dataA is None or dataB is None:
            TraindataA = data_prefetcher(train_loader_a,config)
            TraindataB = data_prefetcher(train_loader_b,config)
            dataA = TraindataA.next()
            dataB = TraindataB.next()
        with Timer("Elapsed time in update: %f"):
            # Main training code
            for _ in range(3):
                trainer.dis_update(dataA, dataB, config)
            # trainer.gen_update(dataA, dataB, config)
            # trainer.gen_update(dataA, dataB, config, loss_gen_adv_a, loss_gen_adv_b, loss_gen_recon_x_a,loss_gen_recon_kl_a, loss_gen_recon_kl_b, loss_gen_recon_kl_sty,
            #                    loss_gen_recon_kl_cyc_aba,loss_gen_recon_kl_cyc_bab, loss_gen_recon_kl_cyc_sty, loss_gen_recon_x_b,
            #                    loss_gen_cyc_x_a,loss_gen_cyc_x_b, loss_gen_vgg_a, loss_gen_vgg_b, loss_bgm, loss_ContentD,
            #                    loss_gen_total,my_loss_bgm)
            trainer.gen_update(dataA, dataB, config, loss_gen_adv_a, loss_gen_adv_b, loss_gen_recon_x_a,loss_gen_recon_x_b,
                               loss_gen_cyc_x_a, loss_gen_cyc_x_b, loss_gen_total,my_sum_loss)
            # torch.cuda.synchronize()
        trainer.update_learning_rate()


        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
        if (iterations + 1) % config['image_save_iter'] == 0:
            testa = testdataA.next()
            testb = testdataB.next()
            if dataA is None or dataB is None or dataA.size(0) != display_size or dataB.size(0) != display_size:
                testdataA = data_prefetcher(test_loader_a,config)
                testdataB = data_prefetcher(test_loader_b,config)
                testa = testdataA.next()
                testb = testdataB.next()
            with torch.no_grad():
                test_image_outputs = trainer.sample(testa, testb)
                train_image_outputs = trainer.sample(dataA, dataB)
            if test_image_outputs is not None and train_image_outputs is not None:
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
            else:
                a = 1

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(dataA, dataB)
            if image_outputs is not None:
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            np.array(loss_gen_adv_a)
            np.array(loss_gen_adv_b)
            np.array(loss_gen_recon_x_a)
            np.array(loss_gen_recon_x_b)
            np.array(loss_gen_cyc_x_a)
            np.array(loss_gen_cyc_x_b)
            # np.array(loss_gen_vgg_a)
            # np.array(loss_gen_vgg_b)
            np.array(loss_gen_total)
            np.array(my_sum_loss)
            # np.array(my_entropy_loss)

            np.save(loss_path+"loss_gen_adv_a.npy",loss_gen_adv_a)
            np.save(loss_path+"loss_gen_adv_b.npy",loss_gen_adv_b)
            np.save(loss_path+"loss_gen_recon_x_a.npy",loss_gen_recon_x_a)
            np.save(loss_path+"loss_gen_recon_x_b.npy",loss_gen_recon_x_b)
            np.save(loss_path+"loss_gen_cyc_x_a.npy",loss_gen_cyc_x_a)
            np.save(loss_path+"loss_gen_cyc_x_b.npy",loss_gen_cyc_x_b)
            # np.save(loss_path+"loss_gen_vgg_a.npy",loss_gen_vgg_a)
            # np.save(loss_path+"loss_gen_vgg_b.npy",loss_gen_vgg_b)
            np.save(loss_path+"loss_gen_total.npy",loss_gen_total)
            np.save(loss_path+"my_sum_loss.npy",my_sum_loss)
            # np.save(loss_path+"my_entropy_loss.npy",my_entropy_loss)

            sys.exit('Finish training')
        

if __name__ == "__main__":
    main()    