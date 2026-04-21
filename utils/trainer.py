import logging
import math
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils.evaluation import Evaluator, segmentation_cross_entropy, noise_mse, write_images_to_tensorboard,noise_bce
from utils.utils import diffuse, get_patch_indices, dynamic_range

# from save_middle_imgs import save_as_image
class TrainerConfig:
    """
    Config settings (hyperparameters) for training.
    """
    # optimization parameters
    max_epochs = 100
    batch_size = 6  #这儿的没用
    learning_rate = 1e-5
    momentum = None
    weight_decay = 0.001 
    grad_norm_clip = 0.95

    # learning rate decay params
    lr_decay = True
    lr_decay_gamma = 0.98

    # network
    network = 'unet'

    # diffusion other settings
    train_on_n_scales = None
    not_recursive = False

    # checkpoint settings
    checkpoint_dir = 'output/checkpoints/'
    log_dir = 'output/logs/'
    load_checkpoint = None
    checkpoint = None
    weights_only = False

    # data inria或者是whu
    dataset_selection = 'whu'

    # other
    eval_every = 2
    save_every = 1 #每跑2次epoch就保存一次
    seed = 0
    n_workers = 8 #默认是8

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def save_config_file(self, filename):
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        logging.info("Saving TrainerConfig file: {}".format(filename))
        with open(filename, 'w') as f:
            for k,v in vars(self).items():
                f.write("{}={}\n".format(k,v))

class Trainer:

    def __init__(self, model, network_config, config, train_data_loader, validation_data_loader=None):
        self.model = model
        self.network_config = network_config
        self.config = config
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.device = config.device

    def create_run_name(self):
        """Creates a unique run name based on current time and network"""
        self.run_name = '{}_{}'.format(time.strftime("%Y%m%d-%H%M"), self.config.network)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, id=None):
        """Saves a model checkpoint"""
        if id is None:
            id = "e{}".format(epoch)
        path = os.path.normpath(self.config.checkpoint_dir + "{}/{}_{}.pt".format(self.run_name, self.run_name, id)) # path/time_network/time_network_epoch.pt
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        logging.info("Saving checkpoint: {}".format(path))
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, path)

    def get_optimizer(self):
        """Defines the optimizer"""

        # optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.999), weight_decay=self.config.weight_decay)
        if (self.config.checkpoint is not None) and (self.config.weights_only is False):
            optimizer.load_state_dict(self.config.checkpoint['optimizer_state_dict'])
        return optimizer
    def get_optimizer_learning_changes(self,model):
        """编码器学习率小，解码器学习率大"""
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if "backbone" in name:  # 编码器参数
                encoder_params.append(param)
            else:  # 解码器参数
                decoder_params.append(param)
        optimizer = optim.AdamW([
            {"params": encoder_params, "lr": self.config.learning_rate},
            {"params": decoder_params, "lr": self.config.learning_rate}
        ], betas=(0.9, 0.999), weight_decay=self.config.weight_decay)
        if (self.config.checkpoint is not None) and (self.config.weights_only is False):
            optimizer.load_state_dict(self.config.checkpoint['optimizer_state_dict'])
        return optimizer
    def get_scheduler(self, optimizer):
        """Defines the learning rate scheduler"""
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay_gamma)
        if (self.config.checkpoint is not None) and (self.config.weights_only is False):
            scheduler.load_state_dict(self.config.checkpoint['scheduler_state_dict'])
        return scheduler
    def train(self):
        """Trains the model"""
        self.create_run_name()
        model = self.model
        network_config = self.network_config
        config = self.config
        optimizer = self.get_optimizer()
        scaler = GradScaler()
        scheduler = self.get_scheduler(optimizer)
        writer = SummaryWriter(log_dir=(config.log_dir + self.run_name))
        # #我写的
        evaluator = Evaluator(model, network_config, self.device, dataset_selection=config.dataset_selection,
                              validation_data_loader=self.validation_data_loader, writer=writer)

        config.save_config_file(
            os.path.normpath(config.checkpoint_dir + "{}/{}_config.txt".format(self.run_name, self.run_name)))
        # 启用 anomaly detection 来定位 inplace 操作问题
        torch.autograd.set_detect_anomaly(True)
        def run_epoch():
            # 将模型设置为训练模式
            model.train()
            # 我写的
            epoch_loss = 0.0
            pbar_epoch = tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader),
                              desc='Epoch {}/{}'.format(epoch + 1, config.max_epochs), leave=False,
                              bar_format='{l_bar}{bar:50}{r_bar}')
            for it, samples in pbar_epoch:
                # Unpack the samples
                images, seg_gt = samples
                images = images.detach().cuda(non_blocking=True)
                seg_gt = seg_gt.detach().cuda(non_blocking=True)
                seg_gt_one_hot = F.one_hot(seg_gt, num_classes=network_config.n_classes + 1).permute(0, 3, 1, 2)[:, :-1,
                                 :, :].float()  # make one hot (if remove void class [:,:-1,:,:])

                # train
                optimizer.zero_grad()
                losses = {}
                predict_img = model(images)
                # 对 predict_img 的六个输出都计算 BCE Loss
                # if isinstance(predict_img, list):
                #     bce_loss_total = 0.0
                #     for idx, pred in enumerate(predict_img):
                #         bce_loss = noise_mse(pred, seg_gt_one_hot)
                #         bce_loss_total += bce_loss
                #         losses[f'bce_output_{idx}'] = bce_loss
                #     total_loss = bce_loss_total
                # else:
                #用BCE Loss
                # bce_loss = noise_mse(predict_img, seg_gt_one_hot)
                # losses['noise_mse'] = bce_loss
                # total_loss = bce_loss
                #用SCE  Loss
                seg_cross_entropy_loss = segmentation_cross_entropy(predict_img, seg_gt_one_hot.argmax(dim=1))
                losses['seg_cross_entropy'] = seg_cross_entropy_loss
                total_loss = seg_cross_entropy_loss

                # Backward pass
                total_loss.backward()

                # Update the parameters
                optimizer.step()

                # Write to tensorboard
                it_total = it + epoch * len(self.train_data_loader)
                if it_total % 10 == 0 and it_total > 0:
                    for loss_name, loss in losses.items():
                        writer.add_scalar('train/{}'.format(loss_name), loss, it_total)

                # Write images to tensorboard,我暂时屏蔽掉，这儿原来是200
                if it % 200 == 0:
                    write_images_to_tensorboard(writer, it_total, image=images[0], seg_predicted=predict_img[0],
                                            seg_gt=seg_gt[0], datasplit='train', dataset_name=config.dataset_selection)
            scheduler.step()

        with logging_redirect_tqdm():
            pbar_total = tqdm(range(config.max_epochs), desc='Total', bar_format='{l_bar}{bar:50}{r_bar}')
            for epoch in pbar_total:
                # Run an epoch
                run_epoch()

                # Save checkpoint
                # if (epoch+1) % config.save_every != 0: #现在不用2个epoch存一次
                self.save_checkpoint(model, optimizer, scheduler, epoch + 1)

                # Evaluate
                if self.validation_data_loader is not None:
                    # if (epoch+1) % config.eval_every != 0: #现在不用2个epoch存一次
                    evaluator.validate(epoch + 1)

            writer.flush()
            writer.close()
