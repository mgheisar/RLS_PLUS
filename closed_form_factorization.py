import argparse
import torch

# Face: "checkpoint/stylegan2-ffhq-config-f.pt" --out "factor_face.pt"
# DMSO: "checkpoint/BlueBubble/150000.pt" --out "factor_BlueBubble.pt"
# "checkpoint/BlueBubbleDMSO/400000.pt" --out "factor_BlueBubbleDMSO.pt"
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )
    parser.add_argument("--ckpt", type=str, default="checkpoint/BBBC021/150000.pt", help="name of the model checkpoint")

    args = parser.parse_args()
    map_location = device
    ckpt = torch.load(args.ckpt, map_location=map_location)
    modulate = {
        k: v
        for k, v in ckpt["g_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)
