import abc
import os
import pickle
from argparse import Namespace
# import wandb
import os.path
# from criteria.localitly_regulizer import Space_Regulizer
import torch
from torchvision import transforms
# from lpips import LPIPS
# from projector import w_projector
from ada import global_config
from ada.models_utils import toogle_grad, load_old_G


class BaseCoach:
    def __init__(self, data_loader, use_wandb):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0

        self.e4e_image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # Initialize loss
        # self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

        # # Initialize checkpoint dir
        # self.checkpoint_dir = global_config.checkpoints_dir
        # os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):

        # Initialize networks
        self.G = load_old_G()
        toogle_grad(self.G, True)
        self.original_G = load_old_G()
        # self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{global_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if global_config.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not global_config.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        if global_config.first_inv_type == 'w+':
            w_potential_path = f'{w_path_dir}/{global_config.e4e_results_keyword}/{image_name}/0.pt'
        else:
            w_potential_path = f'{w_path_dir}/{global_config.pti_results_keyword}/{image_name}/0.pt'
        if not os.path.isfile(w_potential_path):
            return None
        w = torch.load(w_potential_path).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image, image_name):
        id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
        w = id_image
        # w = w_projector.project(self.G, id_image, device=torch.device(global_config.device), w_avg_samples=600,
        #                         num_steps=global_config.first_inv_steps, w_name=image_name,
        #                         use_wandb=self.use_wandb)

        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=global_config.pti_learning_rate)

        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
        loss = 0.0

        if global_config.pt_l2_lambda > 0:
            l2_loss_val = torch.nn.MSELoss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * global_config.pt_l2_lambda
        # if global_config.pt_lpips_lambda > 0:
        #     loss_lpips = self.lpips_loss(generated_images, real_images)
        #     loss_lpips = torch.squeeze(loss_lpips)
        #     if self.use_wandb:
        #         wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
        #     loss += loss_lpips * global_config.pt_lpips_lambda

        # if use_ball_holder and global_config.use_locality_regularization:
        #     ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
        #     loss += ball_holder_loss_val

        return loss, l2_loss_val  #, loss_lpips

    def forward(self, w):
        # generated_images = self.G.synthesis(w, noise_mode='const', force_fp32=True)
        generated_images, _ = self.G([w], input_is_latent=True)
        return generated_images