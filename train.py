import torch
import torchvision
import yaml
from torch import nn
from torch.utils.data import DataLoader

from torchmetrics import KLDivergence
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.segmentation import DiceScore

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from datas.dataset import CSI2Mask_Dataset
from models.VAE import VAE, CLIPVAE, Decoder

from models.FullModel import CSIEncoder
from Loss import dice_loss as DICE
from Loss import kl_divergence_loss as KLD
from utils import random_src_mask, ContrastiveLoss

torch.set_float32_matmul_precision('medium')

def make_img_grid_for_clip(org_rgb, mask, vae_out, out):
    mask = mask.repeat(1, 3, 1, 1)
    vae_out = vae_out.repeat(1, 3, 1, 1)
    out = out.repeat(1, 3, 1, 1)
    images_to_log = torch.cat([
        org_rgb[:6], mask[:6], vae_out[:6], out[:6],
        org_rgb[6:12], mask[6:12], vae_out[6:12], out[6:12],
    ], dim=0)
    img_grid = torchvision.utils.make_grid(images_to_log, nrow=6)
    return img_grid

def make_img_grid(mask, vae_out, out):
    images_to_log = torch.cat([
        mask[:6], vae_out[:6], out[:6],
        mask[6:12], vae_out[6:12], out[6:12],
        mask[12:18], vae_out[12:18], out[12:18]
    ], dim=0)
    img_grid = torchvision.utils.make_grid(images_to_log, nrow=6)
    return img_grid

class CSIEncoderLightning(LightningModule):
    def __init__(self, config, modelconfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Create CSI Encoder
        if config['compile']:
            self.encoder = torch.compile(CSIEncoder(modelconfig))
            self.decoder = torch.compile(Decoder())
        else:
            self.encoder = CSIEncoder(modelconfig)
            self.decoder = Decoder()

        # Create pre-trained VAE
        if config['teacher_arch'] == 'CLIPVAE':
            self.teacher_arch = 'CLIPVAE'
            self.vae = CLIPVAE(activation='leakyrelu', clip_pretrain_model=config['clip_pretrain_model'])
        elif config['teacher_arch'] == 'VAE':
            self.teacher_arch = 'VAE'
            self.vae = VAE(activation='leakyrelu')
        
        # Load decoder and teacher model & freeze parameters
        if config['load_decoder']:    
            checkpoint = torch.load(config['decoder_path'])
            decoder_state_dict = {k.replace('net.','').replace('_orig_mod.','').replace('decoder.',''): v for k, v in checkpoint["state_dict"].items() if 'decoder' in k}
            self.decoder.load_state_dict(decoder_state_dict)
            for param in self.decoder.parameters():
                param.requires_grad = False
            checkpoint = None
            decoder_state_dict = None
        if config['teacher_arch']:
            checkpoint = torch.load(config['teacher_path'])
            vae_state_dict = {k.replace('net.','').replace('_orig_mod.',''): v for k, v in checkpoint["state_dict"].items()}
            self.vae.load_state_dict(vae_state_dict, strict=False)
            for param in self.vae.parameters():
                param.requires_grad = False
            checkpoint = None
            vae_state_dict = None
        
        # Loss functions and metrics
        self.mse = nn.MSELoss()
        self.BCE = nn.BCEWithLogitsLoss()
        self.ssim = SSIM(data_range=(0., 1.))
        self.DICE = DiceScore(num_classes=1, average="micro")
        self.KL = KLDivergence()
        self.CL = ContrastiveLoss()
        self.IoU = BinaryJaccardIndex(threshold=0.5)

        # Loss weights
        self.W_KD = 1
        self.W_MSE = 20 # 40
        self.W_BCE = 5 # 10
        self.W_DICE = 1.5 # 3
        self.W_KL = 50 # 100
        self.W_CL = 0.5 # 0.5

        self.W_SSIM = 0.5 # 1

    def forward(self, amp, pha, mask):
        # create random mask for self-attention
        src_mask = random_src_mask([pha.shape[1], pha.shape[1]], 0.1).to(self.device)

        # CSI Encoder
        if self.encoder.ERC.return_channel_stream:
            csi_z, mu, logvar, amp_channel, pha_channel = self.encoder(amp, pha, src_mask)
        else:
            csi_z, mu, logvar = self.encoder(amp, pha, src_mask)

        # Pre-trained VAE
        vae_z, vae_out, _, _ = self.vae(mask)

        # Decoder
        out = self.decoder(csi_z)

        if self.encoder.ERC.return_channel_stream:
            return out, vae_out, mu, logvar, amp_channel, pha_channel, csi_z, vae_z
        else:
            return out, vae_out, mu, logvar, csi_z, vae_z
    
    def training_step(self, batch, batch_idx):
        [amp, pha, mask, clip_rgb, org_rgb], [another_amp, another_pha], label = batch

        if self.teacher_arch == 'CLIPVAE':
            out, vae_out, mu, logvar, amp_channel, pha_channel, csi_z, vae_z = self.forward(amp, pha, clip_rgb)
            # for contrastive learning
            _, _, _, _, another_amp_channel, another_pha_channel, _, _= self.forward(another_amp, another_pha, clip_rgb)
        elif self.teacher_arch == 'VAE':
            out, vae_out, mu, logvar, amp_channel, pha_channel, csi_z, vae_z = self.forward(amp, pha, mask)
            # for contrastive learning
            _, _, _, _, another_amp_channel, another_pha_channel, _, _= self.forward(another_amp, another_pha, mask)

        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        # dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        dice_loss = DICE(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * self.W_DICE
        kl_loss = torch.clip(KLD(mu, logvar) * self.W_KL, 0, 100)
        ssim_loss = (-self.ssim(torch.sigmoid(out).float(), mask.float())) * self.W_SSIM
        cl_loss = self.CL([amp_channel, pha_channel], [another_amp_channel, another_pha_channel], label) * self.W_CL
        KD_loss = self.mse(csi_z, vae_z) * self.W_KD

        IoU = self.IoU(torch.sigmoid(out).float(), mask.long())

        total_loss = mse_loss + bce_loss + dice_loss + kl_loss + cl_loss + ssim_loss + KD_loss

        self.log_dict({
            'train/total_loss': total_loss,
            'train/mse': mse_loss/self.W_MSE,
            'train/bce': bce_loss/self.W_BCE,
            'train/dice': dice_loss/self.W_DICE,
            'train/kl': kl_loss/self.W_KL,
            'train/ssim': -ssim_loss/self.W_SSIM,
            'train/cl': cl_loss/self.W_CL,
            'train/KD': KD_loss/self.W_KD,
            'train/IoU': IoU,
            'train/lr': self.optimizers().param_groups[0]['lr'],
        }, on_step=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 1000 == 0:
            if self.teacher_arch == 'CLIPVAE':
                img_grid = make_img_grid_for_clip(org_rgb, mask, vae_out, out)
            elif self.teacher_arch == 'VAE':
                img_grid = make_img_grid(mask, vae_out, out)
            self.logger.experiment.add_images('train/images', img_grid, self.global_step, dataformats="CHW")

        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        amp, pha, mask, clip_rgb, org_rgb = batch

        if self.teacher_arch == 'CLIPVAE':
            out, vae_out, mu, logvar, amp_channel, pha_channel, csi_z, vae_z = self.forward(amp, pha, clip_rgb)
        elif self.teacher_arch == 'VAE':
            out, vae_out, mu, logvar, amp_channel, pha_channel, csi_z, vae_z = self.forward(amp, pha, mask)

        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        dice_loss = DICE(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * self.W_DICE
        ssim_loss = (-self.ssim(torch.sigmoid(out).float(), mask.float())) * self.W_SSIM
        KD_loss = self.mse(csi_z, vae_z) * self.W_KD
        IoU = self.IoU(torch.sigmoid(out).float(), mask.long())
        
        # total_loss = bce_loss + dice_loss + ssim_loss
        total_loss = mse_loss + bce_loss + dice_loss + ssim_loss

        if dataloader_idx == 0:
            prefix = 'seen'
        elif dataloader_idx == 1:
            prefix = 'unseen_1'
        else:
            prefix = 'unseen_2'

        self.log_dict({
            f'val/{prefix}/total_loss': total_loss,
            f'val/{prefix}/mse': mse_loss/self.W_MSE,
            f'val/{prefix}/bce': bce_loss/self.W_BCE,
            f'val/{prefix}/dice': dice_loss/self.W_DICE,
            f'val/{prefix}/ssim': -ssim_loss/self.W_SSIM,
            f'val/{prefix}/kl': KD_loss/self.W_KL,
            f'val/{prefix}/IoU': IoU,
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        # for checkpoint
        if dataloader_idx == 1:
            self.log(f'val_loss', total_loss,
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        # log image
        if batch_idx == 0:
            if self.teacher_arch == 'CLIPVAE':
                img_grid = make_img_grid_for_clip(org_rgb, mask, vae_out, out)
            elif self.teacher_arch == 'VAE':
                img_grid = make_img_grid(mask, vae_out, out)
            self.logger.experiment.add_images(f'val/{prefix}/images', img_grid, self.global_step, dataformats="CHW")
    
    def test_step(self, batch, batch_idx):
        amp, pha, mask, clip_rgb, org_rgb = batch
        if self.teacher_arch == 'CLIPVAE':
            out, vae_out, mu, logvar, amp_channel, pha_channel, csi_z, vae_z = self.forward(amp, pha, clip_rgb)
        elif self.teacher_arch == 'VAE':
            out, vae_out, mu, logvar, amp_channel, pha_channel, csi_z, vae_z = self.forward(amp, pha, mask)

        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        # dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        dice_loss = DICE(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * self.W_DICE
        ssim_loss = (-self.ssim(torch.sigmoid(out).float(), mask.float())) * self.W_SSIM
        IoU = self.IoU(torch.sigmoid(out).float(), mask.long())
        
        total_loss = bce_loss + dice_loss + ssim_loss

        self.log_dict({
            'val/total_loss': total_loss,
            'val/mse': mse_loss/self.W_MSE,
            'val/bce': bce_loss/self.W_BCE,
            'val/dice': dice_loss/self.W_DICE,
            'val/ssim': -ssim_loss/self.W_SSIM,
            'val/IoU': IoU,
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # log image
        if batch_idx == 0:
            if self.teacher_arch == 'CLIPVAE':
                img_grid = make_img_grid_for_clip(org_rgb, mask, vae_out, out)
            elif self.teacher_arch == 'VAE':
                img_grid = make_img_grid(mask, vae_out, out)
            self.logger.experiment.add_images('test/images', img_grid, self.global_step, dataformats="CHW")


    def configure_optimizers(self):
        opt_name = self.config['optimizer']
        lr = self.config['lr']
        params = self.parameters()

        if opt_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        elif opt_name == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        else:  # Default to SGD
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-5)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1500)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7, 10, 13, 16], gamma=0.7)
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "train/total_loss",
        }]


def main(config):
    """
    Main function for training CSI2Mask model with PyTorch Lightning.
    :param config: dict, configuration for training and file paths
    """

    # Load model-specific configuration from YAML
    with open(config['model_config_path'], 'r') as f:
        modelconfig = yaml.load(f, Loader=yaml.CLoader)

    # Initialize LightningModule
    model = CSIEncoderLightning(config=config, modelconfig=modelconfig)

    # Setup dataset and dataloaders
    # =============================train=============================
    train_dataset = CSI2Mask_Dataset(
        json_path=config['train&val_json_path'],
        data_root=config['data_root'],
        mode='train',
        clip_pretrain_model=config['clip_pretrain_model']
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=True
    )
    # =============================val=============================
    val_dataset = CSI2Mask_Dataset(
        json_path=config['train&val_json_path'],
        data_root=config['data_root'],
        mode='val',
        clip_pretrain_model=config['clip_pretrain_model']
    )
    val_loader_00 = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    val_dataset = CSI2Mask_Dataset(
        json_path=config['val_json_path'],
        data_root=config['data_root'],
        mode='val',
        clip_pretrain_model=config['clip_pretrain_model']
    )
    val_loader_01 = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    val_dataset = CSI2Mask_Dataset(
        json_path=config['test_json_path'],
        data_root=config['data_root'],
        mode='test',
        clip_pretrain_model=config['clip_pretrain_model']
    )
    val_loader_02 = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    val_loaders = [val_loader_00, val_loader_01, val_loader_02]
    # =============================test=============================
    test_dataset = CSI2Mask_Dataset(
        json_path=config['test_json_path'],
        data_root=config['data_root'],
        mode='test',
        clip_pretrain_model=config['clip_pretrain_model']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )

    # Setup Tensorboard logger
    logger = TensorBoardLogger("lightning_logs", name="WiFi2Seg")
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val/total_loss',
            patience=20,
            mode='min'
        )
    ]

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['dirpath'],
        save_top_k=5,
        monitor='val_loss',
        mode="min",
        filename='WiFi2Seg-{epoch:02d}-{step}-{val_loss:.2f}'
    )

    # Initialize PyTorch Lightning Trainer
    if config['mode'] == 'train':
        trainer = Trainer(
            max_epochs=config['epochs'],
            logger=logger,
            strategy="ddp_find_unused_parameters_true",
            devices=[0, 1],
            callbacks=[checkpoint_callback],
            precision='16-mixed',
            log_every_n_steps=100,
            num_sanity_val_steps=2,
        )
        trainer.fit(model, train_loader, val_loaders, ckpt_path=config['ckpt_path'])
    elif config['mode'] == 'test':
        trainer = Trainer(logger=logger, devices=[0], num_nodes=1)
        trainer.test(model, val_loader_00, ckpt_path=config['ckpt_path'])

if __name__ == '__main__':
    config = {
        'lr': 1e-3,
        'batch_size': 48,
        'num_workers': 24,
        'epochs': 30,
        'optimizer': 'SGD',
        'mode': 'train',
        'compile': False,
        'model_config_path': './model_config.yaml',
        'data_root': '/root/CSI/CSI_dataset_NYCU/', # /root/SSD/PiWiFi/NYCU/
        'train&val_json_path': '/root/terry/CoWIP/datas/NYCU/train&val.json',
        'val_json_path': '/root/terry/CoWIP/datas/NYCU/val.json',
        'test_json_path': '/root/terry/CoWIP/datas/NYCU/test.json',
        # resume setting
        'dirpath': None,
        'ckpt_path': None,
        # decoder setting
        'load_decoder': True,
        'decoder_path': '/root/terry/CoWIP/lightning_logs/VAE/VAE_1x1_new/checkpoints/VAE-epoch=174-step=125125-val_loss=3.61.ckpt',
        # teacher setting
        'teacher_arch': 'VAE',
        'teacher_path': '/root/terry/CoWIP/lightning_logs/VAE/VAE_1x1_new/checkpoints/VAE-epoch=174-step=125125-val_loss=3.61.ckpt',
        # pre-trained model setting
        'clip_pretrain_model': 'RN50',
        # model list: ['RN50','RN101','RN50x4','RN50x16','RN50x64',
        #              'ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
    }

    main(config=config)

    

    
