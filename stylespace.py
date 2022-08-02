import os
import argparse
import numpy as np
from PIL import Image
from model import Generator
from tqdm import tqdm

import torch
from torch.nn import functional as F


def conv_warper(layer, input, style, noise):
    conv = layer.conv
    batch, in_channel, height, width = input.shape

    style = style.view(batch, 1, in_channel, 1, 1)  # reshape (e.g., 512 --> 1,512,1,1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    out = layer.noise(out, noise=noise)
    out = layer.activate(out)

    return out


def encoder(G, noise, input_is_latent=False):
    style_space = []
    inject_index = G.n_latent
    if not input_is_latent:
        styles = [noise]  # (1, 512)
        styles = [G.style(s) for s in styles]
        latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)  # (18, 512)
    else:
        styles = noise
        latent = styles

    style_space.append(G.conv1.conv.modulation(latent[:, 0]))  # ()
    noises_single = generator.make_noise()
    noise = []
    for nois in noises_single:
        noise.append(nois.repeat(1, 1, 1, 1).normal_())
    # noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    i = 1
    # EqualLinear layers to fit the channel dimension (e.g., 512 --> 64)
    for conv1, conv2 in zip(G.convs[::2], G.convs[1::2]):
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_space.append(conv2.conv.modulation(latent[:, i + 1]))
        i += 2
    return style_space, latent, noise


def decoder(G, style_space, latent, noise):
    out = G.input(latent)
    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip = G.to_rgb1(out, latent[:, 1])

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
            G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i + 1], noise=noise2)
        skip = to_rgb(out, latent[:, i + 2], skip)

        i += 2

    image = skip

    return image


def generate_img(generator, input, layer_no, channel_no, degree=30):
    style_space, latent, noise = encoder(generator, input)  # len(style_space) = 11
    style_space[layer_no][:, channel_no] += degree
    image = decoder(generator, style_space, latent, noise)
    return image


def save_fig(output, name, size=128):
    output = (output + 1) / 2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    output = output[0].detach().cpu().permute(1, 2, 0).numpy()
    output = (output * 255).astype(np.uint8)
    im = Image.fromarray(output).resize((size, size), Image.ANTIALIAS)
    im.save(name)


if __name__ == '__main__':
    torch.cuda.set_device(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument("--ckpt", type=str, default="checkpoint/BBBC021/150000.pt")
    parser.add_argument("--out_dir", type=str, default='stylespace')
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--save_all_attr", type=int, default=1)

    args = parser.parse_args()

    generator = Generator(size=128, style_dim=512, n_mlp=8).to(device)
    map_location = lambda storage, loc: storage.cuda()
    gan = torch.load(args.ckpt, map_location=map_location)
    generator.load_state_dict(gan["g_ema"], strict=False)
    generator.eval()
    del gan
    torch.cuda.empty_cache()

    s_channel = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256]

    os.makedirs(args.out_dir, exist_ok=True)

    # default image generation
    torch.manual_seed(args.seed)
    # input = torch.randn(1, 512).to(device)
    loss_str = "1*L1+1*Percept+0.1*Adv"
    latent_dir = "Domain_Projection_Prev/1000Steps_without_noise/demecolcine_10.0"
    dt_list = torch.load(latent_dir + loss_str + ".pt", map_location=map_location)
    dt_files = list(dt_list.keys())
    noises_single = generator.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(1, 1, 1, 1).normal_())
    i = 61
    input = dt_list[dt_files[i]]['latent'].unsqueeze(0)
    image, _ = generator([input], input_is_latent=True, noise=noises)
    save_fig(image, os.path.join(args.out_dir, f'{str(i).zfill(3)}_default.png'))
    if args.save_all_attr:
        # 1. SAVE_ALL ATTR MANIPUlATION RESULT: Let's find out
        # TAKES SOME TIME
        style_space, latent, noise = encoder(generator, input, input_is_latent=True)  # len(style_space) = 11
        for layer_no in range(generator.num_layers):
            os.makedirs(os.path.join(args.out_dir, str(layer_no)), exist_ok=True)
            for channel_no in tqdm(range(s_channel[layer_no])):
                degree = 50
                s_val = style_space[layer_no][:, channel_no].clone().detach()
                style_space[layer_no][:, channel_no] += degree
                image = decoder(generator, style_space, latent, noise)
                style_space[layer_no][:, channel_no] = s_val
                save_fig(image, os.path.join(args.out_dir, str(layer_no), f'{str(args.seed).zfill(6)}_{layer_no}_{channel_no}.png'))
    else:
        # 2. MANIPULATE SPECIFIC ATTRIBUTE
        # pose (?)
        for i in [-30, -10, 10, 30]:
            image = generate_img(generator, input, layer_no=3, channel_no=95, degree=i)
            save_fig(image, os.path.join(args.out_dir, f'{str(args.seed).zfill(6)}_pose_{i}.png'))
    print("generation complete...!")
