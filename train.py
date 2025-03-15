import yaml

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import KLDivergence
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.FullModel import CSIEncoder
from models.VAE import Decoder
from datas.dataset import CSI2Mask_Dataset
from utils import random_src_mask, ContrastiveLoss

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

        # Loss functions and metrics
        self.total_loss = None
        self.BCE = nn.BCEWithLogitsLoss()
        self.DICE = DiceScore(num_classes=1, average="micro")
        self.KL = KLDivergence()
        self.CL = ContrastiveLoss() # Contrastive loss

        self.IoU = BinaryJaccardIndex()

        # Loss weights


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
        bce = self.BCE(out, mask)
        dice = 1 - self.DICE(out, mask)
        # mu = torch.clamp(mu, -10, 10)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl = torch.clamp(kl, 0, 10)
        cl = self.CL([amp_channel, pha_channel], [another_amp_channel, another_pha_channel], label)
        
        # total_loss = bce + dice + cl
        total_loss = bce + dice + kl + cl

        # log
        self.log('train/total_loss', total_loss, prog_bar=True, logger=True)
        self.log('train/bce', bce, on_step=True, prog_bar=True, logger=True)
        self.log('train/dice', dice, on_step=True, prog_bar=True, logger=True)
        self.log('train/kl', kl, on_step=True, prog_bar=True, logger=True)
        self.log('train/cl', cl, on_step=True, prog_bar=True, logger=True)
        # log learning rate
        self.log('train/lr', self.optimizers().state_dict()['param_groups'][0]['lr'], on_step=True, prog_bar=True, logger=True)
        # log image
        if batch_idx % 1000 == 0:
            self.logger.experiment.add_images('train/output', out, self.global_step)
            self.logger.experiment.add_images('train/mask', mask, self.global_step)

        return total_loss
    
    def test_step(self, batch, batch_idx):
        amp, pha, mask = batch

        # forward
        out, mu, logvar, amp_channel, pha_channel = self.forward(amp, pha)

        # metrics, IoU, Dice
        iou = self.IoU(torch.sigmoid(out), mask)
        dice = self.DICE(out, mask)

        self.log('test/iou', iou, on_step=True, prog_bar=True, logger=True)
        self.log('test/dice', dice, on_step=True, prog_bar=True, logger=True)

        # log image
        if batch_idx % 500 == 0:
            self.logger.experiment.add_images('test/output', out, self.global_step)
            self.logger.experiment.add_images('test/mask', mask, self.global_step)


    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'], betas=(0.9, 0.999), eps=1e-8)
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
        'lr': 1e-5,
        'batch_size': 32,
        'num_workers': 4,
        'epochs': 400,
        'model_config_path': './model_config.yaml',
        'data_root': '/root/SSD/PiWiFi/NYCU/', # /root/SSD/PiWiFi/NYCU/
        'train&val_json_path': '/root/terry/CoWIP/datas/NYCU/train&val.json',
        'val_json_path': '/root/terry/CoWIP/datas/NYCU/val.json',
        'test_json_path': '/root/terry/CoWIP/datas/NYCU/test.json',
        'mode': 'train',
        'ckpt_path': '/root/workspace/CoWIP/lightning_logs/WiFi2Seg/version_6/checkpoints/epoch=4-step=11705.ckpt'
    }

    main(config=config)

    

    
