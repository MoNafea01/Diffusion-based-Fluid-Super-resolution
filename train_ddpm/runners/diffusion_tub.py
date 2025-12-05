import os
import logging
import time

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import ConditionalModel
from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry

from tensorboardX import SummaryWriter

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datasets.utils import KMFlowTensorDataset

torch.manual_seed(0)
np.random.seed(0)


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class ConditionalDiffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        u, v = None, None

        # Load training and test datasets
        if os.path.exists(config.data.stat_path):
            print("Loading dataset statistics from {}".format(config.data.stat_path))
            train_data = KMFlowTensorDataset(config.data.data_dir, stat_path=config.data.stat_path)
            train_data_u = KMFlowTensorDataset(config.data.data_dir_u, stat_path=config.data.stat_path_u)
            train_data_v = KMFlowTensorDataset(config.data.data_dir_v, stat_path=config.data.stat_path_v)
            train_datas = (train_data, train_data_u, train_data_v)
        else:
            print("No dataset statistics found. Computing statistics...")
            train_data = KMFlowTensorDataset(config.data.data_dir)
            train_data_u = KMFlowTensorDataset(config.data.data_dir_u)
            train_data_v = KMFlowTensorDataset(config.data.data_dir_v)

            train_data.save_data_stats(config.data.stat_path)
            train_data_u.save_data_stats(config.data.stat_path_u)
            train_data_v.save_data_stats(config.data.stat_path_v)

            train_datas = (train_data, train_data_u, train_data_v)
        x_offset, x_scale = train_data.stat['mean'], train_data.stat['scale']
        train_loader = [torch.utils.data.DataLoader(train_data_i,
                                                   batch_size=config.training.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.data.num_workers) for train_data_i in train_datas]

        model = ConditionalModel(config)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(num_params)
        model = model.to(self.device)
        # model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            print(f"start_epoch: {states[2]}\t step: {states[3]}")
            model.load_state_dict(states[0])

            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        writer = SummaryWriter()
        num_iter = 0
        log_freq = 100
        print('Starting training...')
        for epoch in range(start_epoch, self.config.training.n_epochs):
            loader_x, loader_u, loader_v = train_loader 
            data_start = time.time()
            data_time = 0
            epoch_loss, base_loss, pde_loss = [], [], []
            for i, (x, u, v) in enumerate(zip(loader_x, loader_u, loader_v)):
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(self.device)  # size: [32, 3, 256, 256]
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                loss, b_loss, p_loss = loss_registry[config.model.type](model, x, t, e, b, 
                x_offset.item(), x_scale.item(), u, v, config.training.dx, config.training.dy, 
                config.training.dt, config.training.Re, config.training.w_base)
            
                epoch_loss.append(loss.item())
                base_loss.append(b_loss.item())
                pde_loss.append(p_loss.item())

                tb_logger.add_scalar("loss", loss, global_step=step)

                if num_iter % log_freq == 0:
                    logging.info(
                        f"step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}"
                    )
                #
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('data_time', data_time / (i + 1), step)

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()
                num_iter = num_iter + 1
            print("==========================================================")
            print(f"Epoch: {epoch}/{self.config.training.n_epochs}", 
            f"Loss: {np.mean(epoch_loss)}", 
            f"Base Loss: {np.mean(base_loss)}", 
            f"PDE Loss: {np.mean(pde_loss)}")
        print("Finished training")
        logging.info(
            f"step: {step}, loss: {loss.item()}, data time: {data_time / (i + 1)}"
        )

        torch.save(
            states,
            os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
        )
        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
        print("Model saved at: ", self.args.log_path + "ckpt_{}.pth".format(step))

        writer.export_scalars_to_json("./runs/all_scalars.json")
        writer.close()

    def sample(self):
        # do nothing
        # leave the sampling procedure to sdeit
        pass

    def sample_sequence(self, model):
        pass

    def sample_interpolation(self, model):
        pass

    def sample_image(self, x, model, last=True):
        pass

    def test(self):
        pass

