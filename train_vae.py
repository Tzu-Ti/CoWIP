from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.segmentation import DiceScore
from torchmetrics import KLDivergence

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from torch.utils.data import DataLoader
from datas.dataset import Mask_Dataset
from models.VAE import VAE, CLIPVAE
from Loss import dice_loss as DICE
from torch import nn
import torchvision
import torch
import yaml

torch.set_float32_matmul_precision('medium')

def make_img_grid(mask, out):
    images_to_log = torch.cat([
        mask[:6], out[:6],
        mask[6:12], out[6:12],
        mask[12:18], out[12:18]
    ], dim=0)
    img_grid = torchvision.utils.make_grid(images_to_log, nrow=6)
    return img_grid

class VAELightning(LightningModule):
    def __init__(self, config, modelconfig):
        super().__init__()
        self.config = config
        # save hyparparameters
        self.save_hyperparameters()
        # Create VAE model
        if config['teacher_model'] == 'VAE':
            self.net = torch.compile(VAE(activation='leakyrelu'))
        else:
            self.net = torch.compile(CLIPVAE(activation='leakyrelu', teacher_model=config['teacher_model']))
            print(f"[INFO] Load pre-trained CLIP Encoder from {self.config['teacher_model']}")
            for name, param in self.net.CLIP_model.named_parameters():
                param.requires_grad = False
                print(f"[MODEL] {name} is fixed")
            print('[INFO] Pre-trained CLIP Vision encoder loaded successfully!')

        # Loss functions and metrics
        self.total_loss = None
        self.mse = nn.MSELoss(reduction='mean')
        self.BCE = nn.BCEWithLogitsLoss(reduction='mean')
        self.ssim = SSIM(data_range=(0., 1.))
        self.DICE = DiceScore(num_classes=1, average="micro") # Not good
        self.KL = KLDivergence()

        # Loss weights
        self.W_MSE = 100
        self.W_BCE = 100
        self.W_KL = 1
        self.W_SSIM = 1
        self.W_DICE = 10

    def forward(self, mask):
        # VAE net
        out, mu, logvar = self.net(mask)
        return out, mu, logvar
    
    def training_step(self, batch, batch_idx):
        mask, clip_mask = batch

        # forward
        if self.config['teacher_model'] == 'VAE':
            out, mu, logvar = self.forward(mask)
        else:
            out, mu, logvar = self.forward(clip_mask)

        # loss
        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        # dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        dice_loss = DICE(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * self.W_DICE
        kl_loss = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())) * self.W_KL
        kl_loss = torch.clip(kl_loss, 0, 100)
        ssim_loss = -self.ssim(torch.sigmoid(out).float(), mask.float()) * self.W_SSIM
        
        # total_loss = mse + bce + kl + ssim + dice
        total_loss = mse_loss + bce_loss + dice_loss + kl_loss + ssim_loss

        # log
        self.log('train/total_loss', total_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/mse', mse_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/bce', bce_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/dice', dice_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/kl', kl_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/ssim', -ssim_loss, on_step=True, prog_bar=True, logger=True)
        # log learning rate
        self.log('train/lr', self.optimizers().state_dict()['param_groups'][0]['lr'], on_step=True, prog_bar=True, logger=True)
        # log image
        if batch_idx % 1000 == 0:
            img_grid = make_img_grid(mask, out)
            self.logger.experiment.add_images('train/images', img_grid, self.global_step, dataformats="CHW")

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        mask, clip_mask = batch

        # forward
        if self.config['teacher_model'] == 'VAE':
            out, mu, logvar = self.forward(mask)
        else:
            out, mu, logvar = self.forward(clip_mask)

        # loss
        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        # dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        dice_loss = DICE(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * self.W_DICE
        ssim_loss = -self.ssim(torch.sigmoid(out).float(), mask.float()) * self.W_SSIM

        # total_loss = mse + bce + kl + ssim + dice
        total_loss = mse_loss + bce_loss + dice_loss + ssim_loss

        # log
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/mse', mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/bce', bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/dice', dice_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/ssim', -ssim_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # log image
        if batch_idx == 0:
            img_grid = make_img_grid(mask, out)
            self.logger.experiment.add_images('val/images', img_grid, self.global_step, dataformats="CHW")
    
    def test_step(self, batch, batch_idx):
        mask, clip_mask = batch

        # forward
        if self.config['teacher_model'] == 'VAE':
            out, mu, logvar = self.forward(mask)
        else:
            out, mu, logvar = self.forward(clip_mask)

        # loss
        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        # dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        dice_loss = DICE(torch.sigmoid(out).squeeze(1), mask.squeeze(1)) * self.W_DICE
        ssim_loss = -self.ssim(torch.sigmoid(out).float(), mask.float()) * self.W_SSIM

        # total_loss = mse + bce + kl + ssim + dice
        total_loss = mse_loss + bce_loss + dice_loss + ssim_loss

        # log
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/mse', mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/bce', bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/dice', dice_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/ssim', -ssim_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # log image
        if batch_idx == 0:
            img_grid = make_img_grid(mask, out)
            self.logger.experiment.add_images('test/images', img_grid, self.global_step, dataformats="CHW")

    def configure_optimizers(self):
        # Optimizer
        if self.config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=1e-5)
        elif self.config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], betas=(0.9, 0.999), eps=1e-8)
        elif self.config['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'], betas=(0.9, 0.999), eps=1e-8)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config['lr'], momentum=0.9, weight_decay=1e-5)
        # Scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=300)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120, 250], gamma=0.7)
        scheduler_setup = {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": "train/total_loss",
        }
        return [optimizer], [scheduler_setup]


def main(config):
    """
    Main function for training
    :param config: dict, configuration for training and file paths
    """

    # Load config
    model_config_path = config['model_config_path']
    with open(model_config_path, 'r') as f:
        modelconfig = yaml.load(f, Loader=yaml.CLoader)

    # Create lighning model
    model = VAELightning(config=config, modelconfig=modelconfig)

    # Create Dataloader
    dataset = Mask_Dataset(json_path=config['train&val_json_path'], data_root=config['data_root'], mode='train', teacher_model=config['teacher_model'])
    train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True, pin_memory=True)

    dataset = Mask_Dataset(json_path=config['train&val_json_path'], data_root=config['data_root'], mode='val', teacher_model=config['teacher_model'])
    val_dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True, pin_memory=True)

    dataset = Mask_Dataset(json_path=config['test_json_path'], data_root=config['data_root'], mode='test', teacher_model=config['teacher_model'])
    test_dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False, pin_memory=True)

    # Create Tensorboard logger
    logger = TensorBoardLogger("lightning_logs", name="VAE")

    # Create Early Stopping
    early_stopping = EarlyStopping(monitor="val/total_loss", patience=20, mode="min")
    
    # Create Trainer
    if config['mode'] == 'train':
        trainer = Trainer(max_epochs=config['epochs'], logger=logger, precision="16-mixed")
        trainer.fit(model, train_dataloader, val_dataloader)
    elif config['mode'] == 'test':
        trainer = Trainer(logger=logger)
        trainer.test(model, test_dataloader, ckpt_path=config['ckpt_path'])

if __name__ == '__main__':
    config = {
        'lr': 1e-3,
        'batch_size': 256,
        'num_workers': 8,
        'epochs': 100,
        'optimizer': 'SGD',
        'model_config_path': './model_config.yaml',
        'data_root': '/root/SSD/PiWiFi/NYCU/', # /root/SSD/PiWiFi/NYCU/
        'train&val_json_path': '/root/terry/CoWIP/datas/NYCU/train&val.json',
        'val_json_path': '/root/terry/CoWIP/datas/NYCU/val.json',
        'test_json_path': '/root/terry/CoWIP/datas/NYCU/test.json',
        'mode': 'train',
        'ckpt_path': '/root/workspace/CoWIP/lightning_logs/WiFi2Seg/version_6/checkpoints/epoch=4-step=11705.ckpt',
        'teacher_model': 'VAE',
        # model list: ['VAE','RN50','RN101','RN50x4','RN50x16','RN50x64',
        #              'ViT-B/32','ViT-B/16','ViT-L/14','ViT-L/14@336px']
    }

    main(config=config)

    

    
