import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch_fidelity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use")
    # /projects/superres/Marzieh/ddrm/exp/image_samples/ffhq_1024_64x
    parser.add_argument("--input_dir", type=str, default='input/project/resSR/RLSPlus/Test_ablation_pco/logp01')
    parser.add_argument("--real_path", type=str, default='input/project/resHR')
    parser.add_argument("--kid_subset_size", type=int, default=1000)  # 1000
    parser.add_argument("--fid_saved_result_file", type=str, default="")
    parser.add_argument("--kid_saved_result_file", type=str, default="")
    return parser.parse_args()


args = parse_args()

dir_path = "input/project/resSR/RLSPlus/Test_ablation_pco"
dirs = os.listdir(dir_path)
logp = []
fidd, kidd = [], []
for dir in dirs:
    args.input_dir = os.path.join(dir_path, dir)
    print(dir)
    # Calc metrics
    # args.input_dir = "input/project/resSR/8x_base"
    # args.input_dir = "input/project/resSR/RLSPlus/train/64x"
    metric_scores_dict = torch_fidelity.calculate_metrics(
        input1=args.input_dir,
        input2=args.real_path,
        cuda=True,
        batch_size=args.batch_size,
        fid=True,
        kid=True,
        verbose=False,
        kid_subset_size=args.kid_subset_size,
    )

    fid_score = metric_scores_dict["frechet_inception_distance"]
    kid_mean = metric_scores_dict["kernel_inception_distance_mean"] * 1e3
    kid_std = metric_scores_dict["kernel_inception_distance_std"] * 1e3

    # Log on terminal
    print(f"FID = {fid_score}")
    print(f"KID (x 10^3) = {kid_mean} +- {kid_std}")
    print(metric_scores_dict)
    fidd.append(fid_score)
    kidd.append(kid_mean)
    logp.append('0.' + dir.split('logp')[1])

print(logp)
print(fidd)
print(kidd)
# plot the figure of FID and KID in each subplots
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(logp, fidd, 'r')
plt.xlabel(r'$\lambda_w$')
plt.ylabel('FID')
plt.subplot(1, 2, 2)
plt.plot(logp, kidd, 'g')
plt.xlabel(r'$\lambda_w$')
plt.ylabel('KID')
plt.savefig('input/project/resSR/RLSPlus/fidkid_Test_ablation_pco.png')




# # Save to file
# with open(args.fid_saved_result_file, "w") as f:
#     f.write(f"FID = {fid_score}")
#
# with open(args.kid_saved_result_file, "w") as f:
#     f.write(f"KID (x 10^3) = {kid_mean} +- {kid_std}")
