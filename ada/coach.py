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

    def train(self):

        # w_path_dir = f'{global_config.embedding_base_dir}/{global_config.input_data_id}'
        # os.makedirs(w_path_dir, exist_ok=True)
        # os.makedirs(f'{w_path_dir}/{global_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        # for fname, image in tqdm(self.data_loader):
        # from model import Generator
        # g_ema = Generator(1024, 512, 8).to(global_config.device)
        # map_location = lambda storage, loc: storage.cuda()
        # g_ema.load_state_dict(torch.load(global_config.stylegan_ckpt, map_location=map_location)["g_ema"], strict=False)
        # g_ema.eval()
        for image, im_hr, image_name in self.data_loader:
            # torch.cuda.empty_cache()
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
            # # w_pivot = w_pivot.detach().clone().to(global_config.device)
            # w_pivot = w_pivot.to(global_config.device)
            #
            # torch.save(w_pivot, f'{embedding_dir}/0.pt')
            w_pivot = torch.load(f'w_nf_{self.image_counter}').to(global_config.device)
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)
            Downsampler = BicubicDownSample(factor=global_config.factor, device=global_config.device)
            percept = lpips.PerceptualLoss(
                model="net-lin", net="vgg", gpu_ids=[int(global_config.cuda_visible_devices)]
            )
            pbar = tqdm(range(global_config.max_pti_steps))
            for i in pbar:
                # generated_images, _ = g_ema([w_pivot], input_is_latent=True, randomize_noise=True)
                generated_images = self.forward(w_pivot)
                generated_images = (generated_images + 1) / 2
                l1_loss = F.l1_loss(Downsampler(generated_images), real_images_batch)
                # loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                #                                               self.G, use_ball_holder, w_pivot)
                loss_lpips = percept(Downsampler(generated_images), real_images_batch).mean()
                loss = l1_loss + 0.5 * loss_lpips
                self.optimizer.zero_grad()

                # if loss_lpips <= global_config.LPIPS_value_threshold:
                #     break

                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % global_config.locality_regularization_interval == 0

                if self.use_wandb and log_images_counter % global_config.image_rec_result_log_snapshot == 0:
                    log_images_from_w([w_pivot], self.G, [image_name])
                pbar.set_description(f'L1 Loss: {l1_loss.item()}, LPIPS: {loss_lpips.item()}')
                global_config.training_step += 1
                log_images_counter += 1
                if i % 10 == 0:
                    pil_img = toPIL(generated_images[0].cpu().detach().clamp(0, 1))  # normalizing image (0, 1)
                    # pil_img = make_image(generated_images[0]) # normalizing image (-1, 1)
                    pil_img.save(f'input/project/train/{image_name}_{i}.png')
            self.image_counter += 1

            # torch.save(self.G,
            #            f'{global_config.checkpoints_dir}/model_{global_config.run_name}_{image_name}.pt')