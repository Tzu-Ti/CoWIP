from matplotlib import pyplot as plt
from argparse import ArgumentParser
from matplotlib.axes import Axes
from torchvision import utils
import numpy as np
import torch
from torch import nn
import time
import os

def random_src_mask(shape: list, ratio: float = 0.):
    mask = torch.rand(shape)
    mask = mask < ratio
    return mask

class ContrastiveLoss():
    def __init__(self):
        self.COSINE = nn.CosineEmbeddingLoss(margin=0.5)

    def forward(self, anchor, feature, label):
        amp1, pha1 = anchor
        amp2, pha2 = feature

        B = amp1.shape[0]
        amp1 = amp1.view(B, -1)
        pha1 = pha1.view(B, -1)
        amp2 = amp2.view(B, -1)
        pha2 = pha2.view(B, -1)

        loss = (self.COSINE(amp1, amp2, label) + self.COSINE(pha1, pha2, label)) / 2
        return loss

def Activation(activation: str):
    activation = activation.lower()
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'leakyrelu':
        return torch.nn.LeakyReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'silu':
        return torch.nn.SiLU()
    elif activation == 'gelu':
        return torch.nn.GELU()
    else:
        return torch.nn.ReLU()
    

class Losses(object):
    def __init__(self, losses_name: list):
        '''
        Init name of all loss, and create 2 lists for loss:
            1. self_dict[name] for loss of each iteration.
            2. self_dict[f'{name}_epoch'] for loss of each epoch.
        '''
        self.losses_name = losses_name
        self._total_loss = 0
        self.epoch_loss: dict[str, list] = {}
        self.steps = 0
        for name in losses_name:
            setattr(self, name, [])
            self.epoch_loss[f'{name}_epoch'] = []

    def avg_loss(self):
        '''
        Calculate the avg loss of each epoch, and return the loss string.
        add the avg loss to the list {name}_avg.
        ex: {'loss_avg': 0.123, 'ce_loss_avg': 0.456, ...}
        return: ' loss:0.123 | ce_loss:0.456 | ...'
        '''
        output_str = ''
        for i, name in enumerate(self.losses_name):
            setattr(self, f'{name}_avg', np.mean(self.__dict__[name]))
            if i == 0:
                self._total_loss = self.__dict__[f'{name}_avg']
            output_str += f' {name}:{self.__dict__[f"{name}_avg"]:.3f} |'
        return output_str

    def save_loss(self):
        '''
        Save the avg loss of every epoch to the list {name}_epoch, and clear the list of {name}.
        ex: {'loss_epoch': [0.123, 0.456, ...], 'ce_loss_epoch': [0.789, 0.012, ...], ...}
        '''
        self_dict = self.__dict__
        for i, name in enumerate(self.losses_name):
            self.epoch_loss[f'{name}_epoch'].append(np.mean(self_dict[name]))
            if i == 0:
                self._total_loss = self.epoch_loss[f'{name}_epoch'][-1]
            self_dict[name] = []
        return self.epoch_loss

    def write_log_file(self):
        '''
        Write the avg loss of every epoch to the log dict.
        return: {'loss': 0.123, 'ce_loss': 0.456, ...}
        '''
        log_dict = {}
        for element in self.__dict__:
            if '_avg' in element:
                log_dict[element.replace('_avg', '')] = self.__dict__[element]
        return log_dict

    def push_loss(self, loss_list: list):
        '''
        Push the loss of each iteration to the list {name}.
        '''
        self.steps += 1
        for i, name in enumerate(self.losses_name):
            self.__dict__[name].append(loss_list[i])

class Training_Log():
    def __init__(self, model_name: str = 'My_Model',
                 save_loss_fig: bool = True, save_weight: bool = True, weight_start: int = 30, weight_mode: str = 'min',
                 step_mode: str = 'epoch'):
        self.arg_parse(model_name)
        self.save_init()
        self.save_loss_fig = save_loss_fig
        self.save_weight = save_weight
        self.weight_start = weight_start
        self._all_loss = {}
        self.step_mode = step_mode
        self.epochs = 0
        self.steps_stone = []
        self.warning = False

        if weight_mode == 'min':
            self.best = float('inf')
            self.judgement = lambda input, best: input < best
        elif weight_mode == 'max':
            self.best = float('-inf')
            self.judgement = lambda input, best: input > best

    def save_init(self,):
        if not os.path.exists(self.project_name):
            os.makedirs(self.project_name)
        WEIGHT_SAVE = f"{self.project_name}/ck"
        if not os.path.exists(WEIGHT_SAVE):
            os.makedirs(WEIGHT_SAVE)
        STEP_SAVE = f"{self.project_name}/out"
        if not os.path.exists(STEP_SAVE):
            os.makedirs(STEP_SAVE)
        self.STEP_SAVE = STEP_SAVE
        self.WEIGHT_SAVE = WEIGHT_SAVE

    def init_log_file(self, loss_type: str):
        with open(f'./{self.project_name}/{loss_type}_log.log', '+a') as loss_log:
            loss_log.write(f'Training setting: \n\t{{learning rate:{self.lr}, batch size:{self.batch}, optimizer:\'{self.optimizer}\', total epochs:{self.total_epochs}}}\n')
            loss_log.write(f'\t{{autocast:\'{self.auto_cast}\', compile:\'{self.compile}\', resume:\'{self.resume}\'}}\n')
            loss_log.write(f'\t{{gradient accumulation: {self.grad_accum}}}\n')

    def init_loss(self, **kwargs):
        """
        Args:
            kwargs: {'train_losses': ['loss1', 'loss2', ...], 'test_losses': ['loss1', 'loss2', ...], 'val_losses': ['loss1', 'loss2', ...], 'metrics': ['loss1', 'loss2', ...]}
        """
        support_type = ['train_losses', 'test_losses', 'val_losses', 'metrics']
        for loss_type, loss_name in kwargs.items():
            if loss_type not in support_type:
                raise ValueError(f'\033[0;33m\033[1mInvalid loss type: \"{loss_type}\", support type: {support_type}\033[0m')
            self._all_loss[loss_type] = {'names': loss_name, 'loss_boj': Losses(loss_name)}
            self.init_log_file(loss_type)
        if 'train_losses' in self._all_loss:
            self.train_loss: Losses = self._all_loss['train_losses']['loss_boj']
        if 'test_losses' in self._all_loss:
            self.test_loss: Losses = self._all_loss['test_losses']['loss_boj']
        if 'val_losses' in self._all_loss:
            self.val_loss: Losses = self._all_loss['val_losses']['loss_boj']
        if 'metrics' in self._all_loss:
            self.metrics: Losses = self._all_loss['metrics']['loss_boj']

    def arg_parse(self, model_name='My_Model'):
        parser = ArgumentParser()
        random_name = time.strftime(f"%Y%m%d_%H%M%S", time.localtime()) 
        parser.add_argument('-DW', '--decoder-weight', type=str, default=None, help='decoder weight path')
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        parser.add_argument('-B', '--batch', type=int, default=64, help='batch size')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('-E', '--epochs', type=int, default=600, help='epochs')
        parser.add_argument('-NK', "--num-workers", type=int, default=12, help='num_workers')
        parser.add_argument('-PN', '--project', type=str, default=random_name, help='project name')
        parser.add_argument('-C', '--compile', help='compile model', action='store_true')
        parser.add_argument('-A', '--auto-cast', help='using auto-cast', action='store_true')
        parser.add_argument('-O', '--optimizer', type=str, default='SGD', help='optimizer')
        parser.add_argument('-R', '--resume', type=str, help="reload model from given path", default=None)
        parser.add_argument('-GA', '--grad-accum', type=int, default=1, help='gradient accumulation steps, set to 1 to disable')
        parser.add_argument('-N', '--npy-num', help='set number of input numpy file number', type=int, default=3)
        args = parser.parse_args()
        self.decoder_weight = args.decoder_weight
        self.gpu = args.gpu
        self.batch = args.batch
        self.lr = args.lr
        self.total_epochs = args.epochs
        self.num_workers = args.num_workers
        self.project_name = model_name + '_' + args.project
        self.compile = args.compile
        self.auto_cast = args.auto_cast
        self.optimizer = args.optimizer
        self.resume = args.resume
        self.grad_accum = args.grad_accum
        self.npy_num = args.npy_num

    def loss_txt_log(self):
        self.steps_stone.append(self.train_loss.steps)
        for loss_type, loss_dict in self._all_loss.items():
            log_content = loss_dict['loss_boj'].write_log_file()
            with open(f'./{self.project_name}/{loss_type}_log.log', '+a') as loss_log:
                log_str = '{'
                for k, v in log_content.items():
                    log_str += f'\'{k}\': {v:2.10f},  '
                log_str += '}'
                step_line = f' steps {self.train_loss.steps:6d}' if self.step_mode == 'step' else ''
                log_line = f'epoch {self.epochs:4d}' + step_line + ':' + log_str + '\n'
                loss_log.write(log_line)

    def loss_fig_log(self):
        fig_height = 0.2 + 6 * len(self._all_loss)
        fig = plt.figure(figsize=(16, fig_height))
        self.lightGray = (0.9, 0.9, 0.9)
        fig.suptitle(f'Loss of Epoch: {self.epochs}',color=self.lightGray, fontsize=18, y=0.92)
        fig.patch.set_facecolor((0.12, 0.12, 0.12))

        plot_num = len(self._all_loss)*100 + 11 # 211, 311, 411, ...
        for loss_type, loss_dict in self._all_loss.items():
            ax = fig.add_subplot(plot_num)
            ax = self.ax_setting(ax, title=loss_type, x_label=self.step_mode, y_label='value')
            all_loss = loss_dict['loss_boj'].save_loss()
            for one_loss in loss_dict['names']:
                x = np.arange(len(all_loss[one_loss + '_epoch'])) if self.step_mode == 'epoch' else self.steps_stone
                ax.plot(
                    x, 
                    all_loss[one_loss + '_epoch'], 
                    label=one_loss)
            plt.grid(color=(0.25, 0.25, 0.28), linestyle="-", linewidth=1)
            plt.legend()
            plot_num += 1

        plt.savefig(f'{self.project_name}/loss_fig.png')
        plt.close()

    def ax_setting(self, ax: Axes, title: str = 'loss', x_label: str = 'epoch', y_label='loss'):
        ax.set_title(title, color=self.lightGray, fontsize=16)
        ax.set_facecolor((0.2, 0.2, 0.2))
        ax.set_xlabel(x_label, fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.xaxis.label.set_color(self.lightGray)
        ax.yaxis.label.set_color(self.lightGray)
        ax.tick_params(axis="x", colors=self.lightGray)
        ax.tick_params(axis="y", colors=self.lightGray)
        ax.spines[:].set_color(self.lightGray)
        return ax

    def step(self, epochs=None, net_weight=None, optimizer_state=None, **kwargs):
        """
        Args:
            net_weight (OrderedDict): The state_dict of the model.
            valid_value (Any): The value of the validation.
            img to save [train_img, test_img, val_img, result_img](torch.Tensor) shape: [batch, channel, height, width]
            kwargs: {'train_img': torch.Tensor, 'test_img': torch.Tensor, 'val_img': torch.Tensor, 'result_img': torch.Tensor}
        """
        if self.step_mode == 'step' and epochs is not None:
            self.epochs = epochs
        else:
            self.epochs += 1
        
        # Save loss to txt and fig
        self.loss_txt_log()
        if self.save_loss_fig:
            self.loss_fig_log()


        for key, value in kwargs.items():
            if key not in ['train_img', 'test_img', 'val_img', 'result_img']:
                print(f"\033[0;33m[WARNING] Invalid key: {key}, only support [train_img, test_img, val_img, result_img]\033[0m")
                continue
            if value is not None:
                save_name = f"{key}_{self.epochs:04d}e.png" if self.step_mode == 'epoch' else f"{key}_{self.train_loss.steps:06d}s.png"
                utils.save_image(value.float(), f"{self.STEP_SAVE}/{save_name}", pad_value=0.3)
                print(f"\033[0;32m[INFO]Image of {key} has been saved.\033[0m")

            elif value is None:
                print(f"\033[0;33m[WARNING]Receive 'None', no image saved for {key}.\033[0m")

        if self.save_weight and net_weight is not None:
            compiled = '_compiled' if self.compile else ''

            if self.judgement(self.test_loss._total_loss, self.best) and self.epochs >= self.weight_start:
                torch.save(net_weight, f"{self.WEIGHT_SAVE}/ck_{str(self.epochs).zfill(4)}{compiled}.pt")

                if optimizer_state is not None:
                    torch.save(optimizer_state, f"{self.WEIGHT_SAVE}/ck_{str(self.epochs).zfill(4)}{compiled}_optim.pt")

                self.best = self.test_loss._total_loss

            torch.save(net_weight, f"{self.WEIGHT_SAVE}/ck_last{compiled}.pt")
            torch.save(optimizer_state, f"{self.WEIGHT_SAVE}/ck_last{compiled}_optim.pt")
            
        elif self.save_weight and not self.warning:
            if net_weight is None :
                print("\033[0;33m[WARNING] No weight saved. Please check your input model weight!\033[0m")
            if optimizer_state is None:
                print("\033[0;33m[WARNING] No optimizer state_dict saved. Please check your input optimizer!\033[0m")
            self.warning = True
