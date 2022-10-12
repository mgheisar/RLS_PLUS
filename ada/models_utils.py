import pickle
import functools
import torch
from ada import global_config
from model import Generator


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type):
    new_G_path = f'{global_config.checkpoints_dir}/model_{run_id}_{type}.pt'
    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_old_G():
    # with open(global_config.stylegan2_ada_ffhq, 'rb') as f:
        # old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        # old_G = old_G.float()
    old_G = Generator(global_config.size, 512, 8).to(global_config.device)
    map_location = lambda storage, loc: storage
    old_G.load_state_dict(torch.load(global_config.stylegan_ckpt, map_location=map_location)["g_ema"], strict=False)
    old_G.eval()
    old_G = old_G.float()
    return old_G