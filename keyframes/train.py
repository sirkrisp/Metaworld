import sys
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import yaml
import torch
torch.set_float32_matmul_precision("medium")


# ================================
# 0) set up paths
# ================================

# do not change
ROOT_DIR = f"/home/user/Documents/projects/osil"
sys.path.append(ROOT_DIR)

# => CHANGE
import projects.metaworld.pl_modules as pl_modules
PROJECT_NAME = "metaworld"
SUB_PROJECT_NAME = "feature_matching"
CONFIG_IDX = "01"

# do not change
RUN_NAME = f"{SUB_PROJECT_NAME}_{CONFIG_IDX}"
SUB_PROJECT_DIR = f"{ROOT_DIR}/projects/{PROJECT_NAME}/{SUB_PROJECT_NAME}"
CONFIG_PATH = f"{SUB_PROJECT_DIR}/configs/config_{SUB_PROJECT_NAME}_{CONFIG_IDX}.yaml"
LOG_DIR = f"{SUB_PROJECT_DIR}/logs"

# ================================
# 1) load config
# ================================

with open(CONFIG_PATH, mode="rt", encoding="utf-8") as file:
    config = yaml.safe_load(file)
# validate config
assert (
    config["dataset_args"] is not None
    and config["model_args"] is not None
    and config["optimizer_args"] is not None
    and config["training"] is not None
)
print(config)


# ================================
# 2) intialize dataloader
# ================================

# img_manuals_dataset = img_manuals_dataset.ImgManualsDatasetWithImageTokens(data_folder=DATASET_DIR)
model_data = pl_modules.PLData(
    batch_size=config["training"]["batch_size"],
    num_workers=config["training"]["num_workers"], # TODO make this a param
    **config["dataset_args"],
)

# ================================
# 3) initialize model
# ================================
model = pl_modules.PLModel(config["model_args"], config["optimizer_args"])
# TODO loading from checkpoint does not work..
if config["training"]["init_from"] != "scratch":
    print("Loading model from checkpoint")
    model = model.load_from_checkpoint(config["training"]["init_from"], model_args=config["model_args"], optimizer_args=config["optimizer_args"])
# model.model = BetterTransformer.transform(model.model)

# ================================
# 4) intialize wandb logger
# ================================
wandb_logger = WandbLogger(name=RUN_NAME, project=PROJECT_NAME, config=config)

# ================================
# 5) initialize trainer
# ================================
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
trainer = pl.Trainer(
    default_root_dir=LOG_DIR,
    accelerator="gpu" if str(device).startswith("cuda") else "cpu",
    devices=1,
    max_epochs=config["training"]["max_epochs"],
    callbacks=[
        ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
        LearningRateMonitor("step"),
    ],
    logger=wandb_logger,
    precision=config["training"]["precision"],
    gradient_clip_val=config["training"]["gradient_clip_val"],
    accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
    log_every_n_steps=50 // config["training"]["accumulate_grad_batches"],
    enable_progress_bar=False
)
trainer.logger._log_graph = (
    True  # If True, we plot the computation graph in tensorboard
)
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

# ================================
# 6) start training
# ================================
# trainer.fit(model, train_loader, val_loader)

# TODO this does not work "Cannot set attribute"
# if "global_step" in config["training"]:
#     trainer.global_step = config["training"]["global_step"]

trainer.fit(
    model, 
    model_data,
    # ckpt_path="some/path/to/my_checkpoint.ckpt"
)


# ================================
# 7) evaluate model
# ================================
model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
val_result = trainer.test(model, model_data, verbose=False)
test_result = trainer.test(model, model_data, verbose=False)
result = {"test": test_result[0]["test_loss"], "val": val_result[0]["test_loss"]}
print(result)