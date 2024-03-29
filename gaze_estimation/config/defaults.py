from .config_node import ConfigNode

config = ConfigNode()

# option: ETH-XGaze, MPIIFaceGaze
config.mode = 'ETHXGaze'

config.dataset = ConfigNode()
config.dataset.dataset_dir = 'datasets/MPIIFaceGaze.h5'
config.dataset.raf_dataset_path = r'D:\datasets\RafDB'
config.dataset.fer_dataset_path = r'D:\datasets\fer2013_superresolution'
config.dataset.image_size = 224
config.dataset.n_channels = 3
config.dataset.mean = [0.485, 0.456, 0.406]
config.dataset.std = [0.229, 0.224, 0.225]
config.dataset.transform = ConfigNode()
config.dataset.transform.train = ConfigNode()
config.dataset.transform.train.horizontal_flip = False
config.dataset.transform.train.resize = 224
config.dataset.transform.train.color_jitter = ConfigNode()
config.dataset.transform.train.color_jitter.brightness = [0.4, 1.5]
config.dataset.transform.train.color_jitter.contrast = [0.5, 1.5]
config.dataset.transform.train.color_jitter.saturation = [0.6, 1.3]
config.dataset.transform.train.color_jitter.hue = [0.0, 0.0]
config.dataset.transform.val = ConfigNode()
config.dataset.transform.val.resize = 224
config.dataset.transform.test = ConfigNode()
config.dataset.transform.test.resize = 224


# transform
config.transform = ConfigNode()
config.transform.mpiifacegaze_face_size = 224
config.transform.mpiifacegaze_gray = False

config.device = 'cuda'

config.model = ConfigNode()
config.model.name = 'lenet'
config.model.backbone = ConfigNode()
config.model.backbone.name = 'resnet_simple'
config.model.backbone.pretrained = 'resnet18'
config.model.backbone.resnet_block = 'basic'
config.model.backbone.resnet_layers = [2, 2, 2]

config.train = ConfigNode()
config.train.batch_size = 64
config.train.val_indices = [0]
config.train.emo_pretrained = False
config.train.gaze_pretrained = False
config.train.resume_path = None
config.train.wandb = True
# optimizer (options: sgd, adam, amsgrad)
config.train.optimizer = 'adam'
config.train.base_lr = 0.01
config.train.momentum = 0.9
config.train.nesterov = True
config.train.weight_decay = 1e-4
config.train.no_weight_decay_on_bn = False
# options: L1, L2, SmoothL1
config.train.loss = 'L2'
config.train.class_weights = [1.5, 2, 2, 0.5, 0.8, 1, 1]
config.train.label_smoothing = 0.2
config.train.seed = 0
config.train.val_first = True
config.train.val_period = 1

config.train.output_dir = 'experiments/gaze_emo/exp00'
config.train.log_period = 100
config.train.checkpoint_period = 2

# config.tensorboard = ConfigNode()
# config.tensorboard.train_images = False
# config.tensorboard.val_images = False
# config.tensorboard.model_params = False

# optimizer
config.optim = ConfigNode()
# Adam
config.optim.adam = ConfigNode()
config.optim.adam.betas = (0.9, 0.999)

# scheduler
config.scheduler = ConfigNode()
config.scheduler.epochs = 40
# scheduler (options: multistep, cosine)
config.scheduler.type = 'multistep'
# Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones
config.scheduler.milestones = [20, 30]
config.scheduler.lr_decay = 0.1 # gamma
config.scheduler.lr_min_factor = 0.001

# train data loader
config.train.train_dataloader = ConfigNode()
config.train.train_dataloader.num_workers = 2
config.train.train_dataloader.drop_last = True
config.train.train_dataloader.pin_memory = False
config.train.val_dataloader = ConfigNode()
config.train.val_dataloader.num_workers = 1
config.train.val_dataloader.pin_memory = False

# task weights for the total loss function
config.train.task_weights = ConfigNode()
config.train.task_weights.gaze = 1.0
config.train.task_weights.emotion = 1.0

# test config
config.test = ConfigNode()
config.test.test_id = 0
config.test.checkpoint = ''
config.test.output_dir = ''
config.test.batch_size = 256
# test data loader
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 2
config.test.dataloader.pin_memory = False

# cuDNN
config.cudnn = ConfigNode()
config.cudnn.benchmark = True
config.cudnn.deterministic = False


def get_default_config():
    return config.clone()
