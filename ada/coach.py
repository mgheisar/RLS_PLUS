import os
import torch
from tqdm import tqdm
from ada import global_config
from ada.basecoach import BaseCoach
from ada.log_utils import log_images_from_w
from torch.functional import F
from bicubic import BicubicDownSample
import torchvision
import lpips
from PIL import Image

toPIL = torchvision.transforms.ToPILImage()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_image(tensor):
    return (
        Image.fromarray(
            tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(1, 2, 0)
            .to("cpu")
            .numpy()
        )
    )


class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self, lambda_percept=0, lambda_space=0.1, lambda_percept_space=1, lambda_tv=0, lambda_disc=0.01):
        self.image_counter = 0
        # w_path_dir = f'{global_config.embedding_base_dir}/{global_config.input_data_id}'
        # os.makedirs(w_path_dir, exist_ok=True)
        # os.makedirs(f'{w_path_dir}/{global_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        # for fname, image in tqdm(self.data_loader):
        # from model import Generator
        # g_ema = Generator(1024, 512, 8).to(device)
        # g_ema.load_state_dict(torch.load(global_config.stylegan_ckpt, map_location=device)["g_ema"], strict=False)
        # g_ema.eval()
        for image, im_hr, image_name in self.data_loader:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            self.restart_training()

            if self.image_counter >= global_config.max_images_to_invert:
                break

            # embedding_dir = f'{w_path_dir}/{global_config.pti_results_keyword}/{image_name}'
            # os.makedirs(embedding_dir, exist_ok=True)

            # w_pivot = None
            #
            # if global_config.use_last_w_pivots:
            #     w_pivot = self.load_inversions(w_path_dir, image_name)
            #
            # elif not global_config.use_last_w_pivots or w_pivot is None:
            #     w_pivot = self.calc_inversions(image, image_name)
            #
            # # w_pivot = w_pivot.detach().clone().to(device)
            # w_pivot = w_pivot.to(device)
            #
            # torch.save(w_pivot, f'{embedding_dir}/0.pt')
            w_pivot = torch.load(f'input/project/resSR/test/wnf_{global_config.factor}/wnf_{self.image_counter}').to(device)
            w_anchors = torch.load(f'input/project/resSR/test/wnf_{global_config.factor}/w_nf_{self.image_counter}_10')
            w_anchors = torch.stack(w_anchors).squeeze(1).to(device)
            log_images_counter = 0
            real_images_batch = image.to(device)
            Downsampler = BicubicDownSample(factor=global_config.factor)
            percept = lpips.PerceptualLoss(model="net-lin", net="alex")
            # lambda_percept = 1
            # lambda_space = 0.5  # 0.5
            # lambda_tv = 0  # 0.1
            # lambda_disc = 0  # 0.01
            fake_images_anchor = self.forward(w_anchors).clone().detach()
            pbar = tqdm(range(global_config.max_pti_steps))
            loss_space, loss_tv, discriminator_loss, loss_lpips = 0, 0, 0, 0
            for i in pbar:
                # generated_images, _ = g_ema([w_pivot], input_is_latent=True, randomize_noise=True)
                generated_images = self.forward(w_pivot)
                generated_images_anchor = self.forward(w_anchors)
                loss_space = F.mse_loss(generated_images_anchor, fake_images_anchor) + lambda_percept_space\
                             * percept(generated_images_anchor, fake_images_anchor).mean()
                # a = self.discriminator_forward(fake_images_anchor)
                # b = a .mean()
                # c = self.discriminator_forward(generated_images)
                discriminator_loss = -torch.log(F.softplus(self.discriminator_forward(generated_images) -
                                                torch.mean(self.discriminator_forward(fake_images_anchor)))).mean()
                generated_images = (generated_images + 1) / 2
                l1_loss = F.l1_loss(Downsampler(generated_images), real_images_batch)
                # loss_lpips = percept(Downsampler(generated_images), real_images_batch, normalize=True).mean()

                # loss_tv = F.l1_loss(generated_images[:, :, :, :-1], generated_images[:, :, :, 1:]) +\
                #           F.l1_loss(generated_images[:, :, :-1, :], generated_images[:, :, 1:, :])
                # discriminator_loss = -(self.discriminator_forward(generated_images) -
                #                        torch.mean(self.discriminator_forward(fake_images_anchor)))
                loss = l1_loss + lambda_percept * loss_lpips + lambda_space * loss_space + lambda_tv * loss_tv + \
                       lambda_disc * discriminator_loss
                self.optimizer.zero_grad()

                # if loss_lpips <= global_config.LPIPS_value_threshold:
                #     break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % global_config.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [image_name])
                pbar.set_description(f'L1 Loss: {l1_loss.item()}, LPIPS: {loss_lpips}, '
                                     f'disc_loss: {discriminator_loss}, loss_space: {loss_space},'
                                     f'loss_tv: {loss_tv}')
                global_config.training_step += 1
                log_images_counter += 1
                if i % 10 == 0:
                    pil_img = toPIL(generated_images[0].cpu().detach().clamp(0, 1))  # normalizing image (0, 1)
                    # pil_img = make_image(generated_images[0]) # normalizing image (-1, 1)
                    pil_img.save(f'input/project/resSR/test/traing/{image_name}_{i}_'
                                 f'perc{lambda_space}_disc{lambda_disc}.jpg')
            self.image_counter += 1

            # torch.save(self.G,
            #            f'{global_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')
