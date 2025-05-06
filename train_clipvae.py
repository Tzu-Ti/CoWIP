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

from datas.dataset import Mask_CLIP_Dataset
from models.VAE import CLIPVAE
from Loss import dice_loss as DICE
from Loss import kl_divergence_loss as KLD

torch.set_float32_matmul_precision('medium')

def make_img_grid(org_rgb, mask, out):
    mask = mask.repeat(1, 3, 1, 1)
    out = out.repeat(1, 3, 1, 1)
    images_to_log = torch.cat([
        org_rgb[:6], mask[:6], out[:6],
        org_rgb[6:12], mask[6:12], out[6:12],
    ], dim=0)
    img_grid = torchvision.utils.make_grid(images_to_log, nrow=6)
    return img_grid

class VAELightning(LightningModule):
    def __init__(self, config, modelconfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Create VAE model
        if config['compile']:
            self.net = torch.compile(CLIPVAE(activation='leakyrelu', clip_pretrain_model=config['clip_pretrain_model']))
        else:
            self.net = CLIPVAE(activation='leakyrelu', clip_pretrain_model=config['clip_pretrain_model'])
        print(f"[INFO] Load pre-trained CLIP Encoder from {self.config['clip_pretrain_model']}")

        # I just want to calculate the size of different components in the model.
        self.encoder = self.net.CLIP_model.visual
        self.adapter = self.net.adapter
        self.decoder = self.net.decoder

        # Load pre-trained CLIP model & freeze parameters
        for name, param in self.net.CLIP_model.named_parameters():
            param.requires_grad = False
        print('[INFO] Pre-trained CLIP Vision encoder loaded successfully!')

        # Loss functions and metrics
        self.mse = nn.MSELoss(reduction='mean')
        self.BCE = nn.BCEWithLogitsLoss(reduction='mean')
        self.ssim = SSIM(data_range=(0., 1.))
        self.DICE = DiceScore(num_classes=1, average="micro") # Not good
        self.KL = KLDivergence()

        # Loss weights
        self.W_MSE = 100
        self.W_BCE = 100
        self.W_KL = 100
        self.W_SSIM = 1
        self.W_DICE = 10

    def forward(self, mask):
        z, out, mu, logvar = self.net(mask)
        return out, mu, logvar
    
    def training_step(self, batch, batch_idx):
        mask, clip_rgb, org_rgb = batch
        out, mu, logvar = self.forward(clip_rgb)

        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        # dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        dice_loss = DICE(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * self.W_DICE
        kl_loss = torch.clip(KLD(mu, logvar) * self.W_KL, 0, 100)
        ssim_loss = (-self.ssim(torch.sigmoid(out).float(), mask.float())) * self.W_SSIM
        
        total_loss = mse_loss + bce_loss + dice_loss + kl_loss + ssim_loss

        self.log_dict({
            'train/total_loss': total_loss,
            'train/mse': mse_loss/self.W_MSE,
            'train/bce': bce_loss/self.W_BCE,
            'train/dice': dice_loss/self.W_DICE,
            'train/kl': kl_loss/self.W_KL,
            'train/ssim': -ssim_loss/self.W_SSIM,
            'train/lr': self.optimizers().param_groups[0]['lr'],
        }, on_step=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 1000 == 0:
            img_grid = make_img_grid(org_rgb, mask, out)
            self.logger.experiment.add_images('train/images', img_grid, self.global_step, dataformats="CHW")

        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        mask, clip_rgb, org_rgb = batch
        out, mu, logvar = self.forward(clip_rgb)

        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        # dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        dice_loss = DICE(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * self.W_DICE
        ssim_loss = -self.ssim(torch.sigmoid(out).float(), mask.float()) * self.W_SSIM

        total_loss = mse_loss + bce_loss + dice_loss + ssim_loss

        if dataloader_idx == 0:
            prefix = 'seen'
        else:
            prefix = 'unseen'

        self.log_dict({
            f'val/{prefix}/total_loss': total_loss,
            f'val/{prefix}/mse': mse_loss/self.W_MSE,
            f'val/{prefix}/bce': bce_loss/self.W_BCE,
            f'val/{prefix}/dice': dice_loss/self.W_DICE,
            f'val/{prefix}/ssim': -ssim_loss/self.W_SSIM,
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        # for checkpoint
        if dataloader_idx == 1:
            self.log(f'val_loss', total_loss,
            on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, add_dataloader_idx=False)

        # log image
        if batch_idx == 0:
            img_grid = make_img_grid(org_rgb, mask, out)
            self.logger.experiment.add_images(f'val/{prefix}/images', img_grid, self.global_step, dataformats="CHW")
    
    def test_step(self, batch, batch_idx):
        mask, clip_rgb, org_rgb = batch
        out, mu, logvar = self.forward(clip_rgb)

        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        # dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        dice_loss = DICE(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * self.W_DICE
        ssim_loss = -self.ssim(torch.sigmoid(out).float(), mask.float()) * self.W_SSIM

        total_loss = mse_loss + bce_loss + dice_loss + ssim_loss

        self.log_dict({
            'val/total_loss': total_loss,
            'val/mse': mse_loss/self.W_MSE,
            'val/bce': bce_loss/self.W_BCE,
            'val/dice': dice_loss/self.W_DICE,
            'val/ssim': -ssim_loss/self.W_SSIM,
        }, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # log image
        if batch_idx == 0:
            img_grid = make_img_grid(org_rgb, mask, out)
            self.logger.experiment.add_images('test/images', img_grid, self.global_step, dataformats="CHW")

    def configure_optimizers(self):
        opt_name = self.config['optimizer']
        lr = self.config['lr']
        params = self.net.parameters()

        if opt_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        elif opt_name == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        else:  # Default to SGD
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-5)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=300)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 250], gamma=0.7)
        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "train/total_loss",
        }]


def main(config):
    """
    Main function for training CLIPVAE model with PyTorch Lightning.
    :param config: dict, configuration for training and file paths
    """

    # Load model-specific configuration from YAML
    with open(config['model_config_path'], 'r') as f:
        modelconfig = yaml.load(f, Loader=yaml.CLoader)

    # Initialize LightningModule
    model = VAELightning(config=config, modelconfig=modelconfig)

    # Setup dataset and dataloaders
    # =============================train=============================
    train_dataset = Mask_CLIP_Dataset(
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
    val_dataset = Mask_CLIP_Dataset(
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
    val_dataset = Mask_CLIP_Dataset(
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
    val_loaders = [val_loader_00, val_loader_01]
    # =============================test=============================
    test_dataset = Mask_CLIP_Dataset(
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
    logger = TensorBoardLogger("lightning_logs", name="CLIPVAE")

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
        filename='CLIPVAE-{epoch:02d}-{step}-{total_loss:.2f}'
    )

    # Initialize PyTorch Lightning Trainer
    if config['mode'] == 'train':
        trainer = Trainer(
            max_epochs=config['epochs'],
            logger=logger,
            devices=[0, 1],
            callbacks=[checkpoint_callback],
            precision='16-mixed',
            log_every_n_steps=100,
            num_sanity_val_steps=2,
        )
        trainer.fit(model, train_loader, val_loaders, ckpt_path=config['ckpt_path'])
    elif config['mode'] == 'test':
        trainer = Trainer(logger=logger)
        trainer.test(model, test_loader, ckpt_path=config['ckpt_path'])

if __name__ == '__main__':
    config = {
        'lr': 1e-3,
        'batch_size': 96,
        'num_workers': 24,
        'epochs': 200,
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
        # pre-trained model setting
        'clip_pretrain_model': 'RN50',
        # model list: ['RN50','RN101','RN50x4','RN50x16','RN50x64',
        #              'ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
    }

    main(config=config)

