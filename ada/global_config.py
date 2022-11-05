## Device
cuda_visible_devices = '3'
device = 'cuda:3'

## Logs
training_step = 1
image_rec_result_log_snapshot = 100
pivotal_training_steps = 10
model_snapshot_interval = 400
factor = 32
max_pti_steps = 41
max_images_to_invert = 8
pti_learning_rate = 0.001  # 3e-4
## Run name to be updated during PTI
run_name = ''
size = 1024
stylegan_ckpt = "checkpoint/stylegan2-ffhq-config-f_.pt"

# ## Pretrained models paths
# e4e = './pretrained_models/e4e_ffhq_encode.pt'
# stylegan2_ada_ffhq = '../pretrained_models/ffhq.pkl'
# style_clip_pretrained_mappers = ''
# ir_se50 = './pretrained_models/model_ir_se50.pth'
# dlib = './pretrained_models/align.dat'

## Dirs for output files
# checkpoints_dir = './checkpoints'
embedding_base_dir = './embeddings'
# styleclip_output_dir = './StyleCLIP_results'

## Input info
### Input dir, where the images reside
input_data_path = '/disk2/danielroich/Sandbox/Data/Images/barcelona/aligned/0'
### Inversion identifier, used to keeping track of the inversion results. Both the latent code and the generator
input_data_id = 'barcelona'

## Keywords
pti_results_keyword = 'PTI'
# e4e_results_keyword = 'e4e'
# sg2_results_keyword = 'SG2'
# sg2_plus_results_keyword = 'SG2_plus'
#
# ## Edit directions
# interfacegan_age = 'editings/interfacegan_directions/age.pt'
# interfacegan_smile = 'editings/interfacegan_directions/smile.pt'
# interfacegan_rotation = 'editings/interfacegan_directions/rotation.pt'
# ffhq_pca = 'editings/ganspace_pca/ffhq_pca.pt'

## Architechture
lpips_type = 'alex'
first_inv_type = 'w'
optim_type = 'adam'
batch_size = 1

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1

## Steps
LPIPS_value_threshold = 0.06
first_inv_steps = 450

## Optimization
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = False
