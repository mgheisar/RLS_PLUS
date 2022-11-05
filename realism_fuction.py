import torch
import numpy as np
import operator
import argparse
from tqdm import tqdm
from model import Generator
root = '/projects/superres/Marzieh/pytorch-flows/data/'
Winv_DIRECTORY = "w_nf"
W_DIRECTORY = "w_samples_train_face.pt"
gpu_num = 2
torch.cuda.set_device(gpu_num)
cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class realism(object):
    def __init__(self, args):
        # parameters
        self.args = args
        # self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.cpu = args.cpu
        self.k = 10

        # g_ema = Generator(128, 512, 8).cuda()
        # map_location = device
        # g_ema.load_state_dict(torch.load(args.ckpt, map_location=map_location)["g_ema"], strict=False)
        # g_ema.eval()
        # latent = torch.randn((10000, 512), dtype=torch.float32).cuda()
        # latent = g_ema.style(latent)
        # torch.save(latent, 'w_samples.pt')
        # exit(0)

    def run(self):

        # load data using vgg16
        # extractor = feature_extractor(self.args)
        # generated_features, real_features, generated_img_paths = extractor.extract()
        map_location = device
        w_samples = torch.load(root+W_DIRECTORY, map_location=map_location)
        w_invs = torch.load(Winv_DIRECTORY, map_location=map_location)
        w_invs = w_invs.squeeze(0)
        # equal number of samples
        w_samples = w_samples[:10000]
        # KNN_list_in_w = self.calculate_w_NNK(w_samples, self.k)
        KNN_list_in_w = torch.load("KNN_list_in_wk5_.pt", map_location=map_location)
        real_vals = []
        for i, w_inv in enumerate(tqdm(w_invs, ncols=80)):
            max_value = 0
            for w_sample, KNN_radius in KNN_list_in_w:
                d = torch.norm((w_sample - w_inv), 2)
                value = KNN_radius / d
                if max_value < value:
                    max_value = value

            # print images with specific names
            real_vals.append(max_value)
        print(f'\n realism score: {torch.mean(torch.stack(real_vals))}')

        return

    @staticmethod
    def calculate_w_NNK(w_samples, k):
        KNN_list_in_w = {}
        for w_sample in tqdm(w_samples, ncols=80):
            pairwise_distances = np.zeros(shape=(len(w_samples)))

            for i, w_sample_prime in enumerate(w_samples):
                d = torch.norm((w_sample - w_sample_prime), 2)
                pairwise_distances[i] = d

            v = np.partition(pairwise_distances, k)[k]
            KNN_list_in_w[w_sample] = v

        # remove half of larger values
        KNN_list_in_w = sorted(KNN_list_in_w.items(), key=operator.itemgetter(1))
        torch.save(KNN_list_in_w, "KNN_list_in_w.pt")
        # KNN_list_in_w = KNN_list_in_w[:int(data_num / 2)]

        return KNN_list_in_w


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="calcualte realism metric using W rather feature vector")
    # parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the model')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--ckpt', type=str, default="checkpoint/BBBC021/150000.pt")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_size', type=int, default=100)
    parser.add_argument('--Winv_dir', default=Winv_DIRECTORY)
    parser.add_argument('--W_dir', default=W_DIRECTORY)
    args = parser.parse_args()
    realism = realism(args)
    realism.run()
    print("finished")