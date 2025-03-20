from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import KLDivergence

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from utils import random_src_mask, ContrastiveLoss
from datas.dataset import CSI2Mask_Dataset
from torch.utils.data import DataLoader
from models.FullModel import CSIEncoder
from models.VAE import Decoder
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

class EncoderLightning(LightningModule):
    def __init__(self, config, modelconfig):
        super().__init__()
        self.config = config
        # save hyparparameters
        self.save_hyperparameters()
        # Create CSI Encoder
        self.encoder = CSIEncoder(modelconfig)
        # Create VAE Decoder
        self.decoder = Decoder()
        # <--------------------------------------Load pre-trained VAE Decoder-------------------------------------->
        if self.config['decoder_weight'] is not None:
            print(f"[INFO] Load pre-trained VAE Decoder from {self.config['decoder_weight']}")
            decoder_weight = torch.load(self.config['decoder_weight'])
            for name, param in self.decoder.named_parameters():
                if name in decoder_weight:
                    param.data = decoder_weight[name]
                    param.requires_grad = False
                else:
                    print(f'[Warning] \"{name}\"<- parameter not in pre-trained weight!')
            print('[INFO] VAE Decoder loaded successfully!')
            decoder_weight = None
        # <-------------------------------------------------------------------------------------------------------->
        # Loss functions and metrics
        self.total_loss = None
        self.mse = nn.MSELoss()
        self.BCE = nn.BCEWithLogitsLoss()
        self.ssim = SSIM(data_range=(0., 1.))
        self.DICE = DiceScore(num_classes=1, average="micro")
        self.KL = KLDivergence()

        self.CL = ContrastiveLoss() # Contrastive loss
        self.IoU = BinaryJaccardIndex(threshold=0.5)

        # Loss weights
        self.W_MSE = 40
        self.W_BCE = 10
        self.W_KL = 100
        self.W_SSIM = 1
        self.W_DICE = 3
        self.W_CL = 0.5

    def forward(self, amp, pha):
        # create random mask for self-attention
        src_mask = random_src_mask([pha.shape[1], pha.shape[1]], 0.1).to(self.device)

        # Encoder
        if self.encoder.ERC.return_channel_stream:
            z, mu, logvar, amp_channel, pha_channel = self.encoder(amp, pha, src_mask)
        else:
            z, mu, logvar = self.encoder(amp, pha, src_mask)

        # Decoder
        out = self.decoder(z)

        if self.encoder.ERC.return_channel_stream:
            return out, mu, logvar, amp_channel, pha_channel
        else:
            return out, mu, logvar
    
    def training_step(self, batch, batch_idx):
        [amp, pha, mask], [another_amp, another_pha], label = batch

        # forward
        out, mu, logvar, amp_channel, pha_channel = self.forward(amp, pha)
        # for contrastive learning
        _, _, _, another_amp_channel, another_pha_channel = self.forward(another_amp, another_pha)

        # loss
        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        kl_loss = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())) * self.W_KL
        kl_loss = torch.clamp(kl_loss, 0, 100)
        ssim_loss = -self.ssim(torch.sigmoid(out).float(), mask.float()) * self.W_SSIM
        cl_loss = self.CL([amp_channel, pha_channel], [another_amp_channel, another_pha_channel], label) * self.W_CL
        
        # total_loss = mse + bce + kl + ssim + dice
        total_loss = mse_loss + bce_loss + dice_loss + kl_loss + ssim_loss + cl_loss

        # log
        self.log('train/total_loss', total_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/mse', mse_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/bce', bce_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/dice', dice_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/kl', kl_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/ssim', -ssim_loss, on_step=True, prog_bar=True, logger=True)
        self.log('train/cl', cl_loss, on_step=True, prog_bar=True, logger=True)
        # log learning rate
        self.log('train/lr', self.optimizers().state_dict()['param_groups'][0]['lr'], on_step=True, prog_bar=True, logger=True)
        # log image
        if batch_idx % 1000 == 0:
            img_grid = make_img_grid(mask, out)
            self.logger.experiment.add_images('train/images', img_grid, self.global_step, dataformats="CHW")

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        [amp, pha, mask], [another_amp, another_pha], label = batch

        # forward
        out, mu, logvar, amp_channel, pha_channel = self.forward(amp, pha)

        # loss
        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        ssim_loss = -self.ssim(torch.sigmoid(out).float(), mask.float()) * self.W_SSIM
        
        IoU = self.IoU(torch.sigmoid(out).float(), mask.long())
        
        # total_loss = mse + bce + dice + ssim
        total_loss = mse_loss + bce_loss + dice_loss + ssim_loss

        # log
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/mse', mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/bce', bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/dice', dice_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/ssim', -ssim_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('val/IoU', IoU, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # log image
        if batch_idx == 0:
            img_grid = make_img_grid(mask, out)
            self.logger.experiment.add_images('val/images', img_grid, self.global_step, dataformats="CHW")
    
    def test_step(self, batch, batch_idx):
        amp, pha, mask = batch

        # forward
        out, mu, logvar, amp_channel, pha_channel = self.forward(amp, pha)

        # loss
        mse_loss = self.mse(torch.sigmoid(out), mask) * self.W_MSE
        bce_loss = self.BCE(out, mask) * self.W_BCE
        dice_loss = (1 - self.DICE(torch.sigmoid(out), mask)) * self.W_DICE
        ssim_loss = -self.ssim(torch.sigmoid(out).float(), mask.float()) * self.W_SSIM
        
        IoU = self.IoU(torch.sigmoid(out).float(), mask.long())
        
        # total_loss = mse + bce + dice + ssim
        total_loss = mse_loss + bce_loss + dice_loss + ssim_loss

        # log
        self.log('test/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test/mse', mse_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test/bce', bce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test/dice', dice_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('test/ssim', -ssim_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('test/IoU', IoU, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # log image
        if batch_idx == 0:
            img_grid = make_img_grid(mask, out)
            self.logger.experiment.add_images('test/images', img_grid, self.global_step, dataformats="CHW")


    def configure_optimizers(self):
        # Optimizer
        if self.config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config['lr'])
        elif self.config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], betas=(0.9, 0.999), eps=1e-8)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config['lr'])
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100)
        scheduler_setup = {
            "scheduler": scheduler,
            "interval": "step",
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
    model = EncoderLightning(config=config, modelconfig=modelconfig)

    # Create Dataloader
    dataset = CSI2Mask_Dataset(json_path=config['train&val_json_path'], data_root=config['data_root'], mode='train')
    train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)

    dataset = CSI2Mask_Dataset(json_path=config['train&val_json_path'], data_root=config['data_root'], mode='val')
    val_dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)

    dataset = CSI2Mask_Dataset(json_path=config['test_json_path'], data_root=config['data_root'], mode='test')
    test_dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)

    # Create Tensorboard logger
    logger = TensorBoardLogger("lightning_logs", name="WiFi2Seg")
    

    # Create Trainer
    if config['mode'] == 'train':
        trainer = Trainer(max_epochs=config['epochs'],
                        strategy="ddp_find_unused_parameters_true",
                        logger=logger)
        trainer.fit(model, train_dataloader, val_dataloader)
    elif config['mode'] == 'test':
        trainer = Trainer(strategy="ddp_find_unused_parameters_true", logger=logger)
        trainer.test(model, test_dataloader, ckpt_path=config['ckpt_path'])

if __name__ == '__main__':
    config = {
        'lr': 1e-4,
        'batch_size': 16,
        'num_workers': 8,
        'epochs': 20,
        'optimizer': 'SGD',
        'model_config_path': './model_config.yaml',
        'data_root': '/root/SSD/PiWiFi/NYCU/', # /root/SSD/PiWiFi/NYCU/
        'train&val_json_path': '/root/terry/CoWIP/datas/NYCU/train&val.json',
        'val_json_path': '/root/terry/CoWIP/datas/NYCU/val.json',
        'test_json_path': '/root/terry/CoWIP/datas/NYCU/test.json',
        'mode': 'train',
        'ckpt_path': '/root/workspace/CoWIP/lightning_logs/WiFi2Seg/version_6/checkpoints/epoch=4-step=11705.ckpt',
        'decoder_weight': None
    }

    main(config=config)

    

    
