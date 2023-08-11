import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from random import choice
from string import ascii_uppercase
from torchvision.transforms import transforms
from ada import global_config
# import wandb
import torchvision
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import glob
from ada.coach import SingleIDCoach

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Images(Dataset):
    def __init__(self, image_list, duplicates):
        # args.files = [sorted(glob.glob(f"input/project/inputt/*_{args.factor}x.jpg"))[args.img_idx]]
        self.image_list = image_list
        self.duplicates = duplicates  # Number of times to duplicate the image in the dataset to produce multiple HR
        self.transform = torchvision.transforms.Compose([transforms.ToTensor()])
        # self.transform = torchvision.transforms.Compose([transforms.ToTensor(),
        #                                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return self.duplicates * len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx // self.duplicates]
        image = self.transform(Image.open(img_path)).to(device)
        # image_hr = self.transform(Image.open(img_path.split('_')[0] + '_HR.jpg')).to(torch.device("cuda"))
        hr_path = 'input/project/resHR/00001_64x_HR.jpg'
        image_hr = self.transform(Image.open(hr_path)).to(device)
        if self.duplicates == 1:
            return image, image_hr, os.path.splitext(os.path.basename(img_path))[0]
        else:
            return image, image_hr, os.path.splitext(os.path.basename(img_path))[0] + f"_{(idx % self.duplicates) + 1}"


def run_PTI(run_name='', use_wandb=False, use_multi_id_training=False):
    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=global_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{global_config.embedding_base_dir}/{global_config.input_data_id}/{global_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    image_list = sorted(glob.glob(f"input/project/lrr/lr_{global_config.factor}/*_{global_config.factor}x.jpg"))
    dataset = Images(image_list, duplicates=1)
    dataloader = DataLoader(dataset, batch_size=global_config.batch_size)
    # dataset = ImagesDataset(global_config.input_data_path, transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))
    #
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # if use_multi_id_training:
    #     coach = MultiIDCoach(dataloader, use_wandb)
    # else:
    coach = SingleIDCoach(dataloader, use_wandb)
    space_vec = [0.5]
    disc_vec = [0]  # , 0.005]
    for i_space in space_vec:
        for i_disc in disc_vec:
            print(f"disc: {i_disc}, space: {i_space}")
            coach.train(lambda_disc=i_disc, lambda_space=i_space)


if __name__ == '__main__':
    run_PTI(run_name='', use_wandb=False, use_multi_id_training=False)
