import yaml

from pytorch_lightning import LightningModule, Trainer
from torchmetrics.segmentation import DiceScore
from torchmetrics import KLDivergence
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.FullModel import CSIEncoder
from datas.dataset import CSI2Mask_Dataset
from utils import random_src_mask, ContrastiveLoss

class EncoderLightning(LightningModule):
    def __init__(self, config, modelconfig):
        super().__init__()
        self.config = config
        # Create CSI Encoder
        self.model = CSIEncoder(modelconfig)

        # Loss functions and metrics
        self.total_loss = None
        self.BCE = nn.BCEWithLogitsLoss()
        self.DICE = DiceScore(num_classes=1, average="micro")
        self.KL = KLDivergence()
        self.CL = ContrastiveLoss() # Contrastive loss

        # Loss weights

    def forward(self, amp, pha):
        # create random mask for self-attention
        src_mask = random_src_mask([pha.shape[1], pha.shape[1]], 0.1).to(self.device)

        if self.model.encoder.return_channel_stream:
            out, mu, logvar, amp_channel, pha_channel = self.model(amp, pha, src_mask)
            return out, mu, logvar, amp_channel, pha_channel
        else:
            out, mu, logvar = self.model(amp, pha, src_mask)
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
        kl = self.KL(mu, logvar)
        cl = self.CL([amp_channel, pha_channel], [another_amp_channel, another_pha_channel], label)

        total_loss = bce + dice + kl + cl

        return total_loss

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'])
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100)
        scheduler_setup = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "total_loss",
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
    dataset = CSI2Mask_Dataset(json_path=config['train&val_json_path'], mode='train')
    train_dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)

    dataset = CSI2Mask_Dataset(json_path=config['train&val_json_path'], mode='val')
    val_dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)

    dataset = CSI2Mask_Dataset(json_path=config['test_json_path'], mode='test')
    test_dataloader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)

    # Create Trainer
    trainer = Trainer(max_epochs=config['epochs'])
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    config = {
        'lr': 1e-4,
        'batch_size': 32,
        'num_workers': 4,
        'epochs': 400,
        'model_config_path': './model_config.yaml',
        'train&val_json_path': '/root/CSI/WiFi2Seg_pl/datas/train&val.json',
        'val_json_path': '/root/CSI/WiFi2Seg_pl/datas/val.json',
        'test_json_path': '/root/CSI/WiFi2Seg_pl/datas/test.json'
    }

    main(config=config)

    

    
