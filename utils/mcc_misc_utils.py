import torch
from dataclasses import dataclass
import models.extern.mcc_model as mcc_model
from tqdm import tqdm
import numpy as np


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        print("Resume checkpoint %s" % args.resume)
        print(model_without_ddp.load_state_dict(checkpoint['model'], strict=False))
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                print(loss_scaler.load_state_dict(checkpoint['scaler']))
            print("With optim & sched!")
            print("start epoch:", args.start_epoch)

@dataclass
class MCCConfig:
    # Model
    checkpoint: str
    resume: str
    eval: bool = True

    input_size: int = 224
    device: str = "cuda"  # cuda or cpu
    drop_path: float = 0.1  # drop_path probability
    regress_color: bool = False  # If true, regress color with MSE. Otherwise, 256-way classification for each channel.
    shrink_threshold: float = 10  # Any points with distance beyond this value will be shrunk.

    # rgb_weight: float = 0.01  # A constant to weight the color prediction loss
    # occupancy_weight: float = 1.0  # A constant to weight the occupancy loss
    # n_queries: int = 550  # Number of queries used in decoder.
    # embed_dim: int = 1024
    # depth: int = 24
    # num_heads: int = 16
    # decoder_embed_dim: int = 512
    # decoder_depth: int = 8
    # decoder_num_heads: int = 16
    # mlp_ratio: float = 4.
    # img_size: int = 224
    # patch_size: int = 16
    # in_chans: int = 3
    # norm_layer: partial(nn.LayerNorm, eps=1e-6)


def load_model_from_ckpt(ckpt_path: str, input_size=224, device="cuda"):
    args = MCCConfig(input_size=input_size, checkpoint=ckpt_path, resume=ckpt_path, device=device)
    
    model = mcc_model.get_mcc_model(
        occupancy_weight=1.0,
        rgb_weight=0.01,
        args=args,
    ).cuda()  # TODO this should be device agnostic

    load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)

    model.to(device=device)

    return model


def predict_mcc(
    model: torch.nn.Module, 
    model_input: dict, 
    temperature=0.1,  # temperature for color prediction.
    max_n_query_points = 2000
):
    model.eval()

    seen_rgb = model_input["seen_rgb"]
    seen_xyz = model_input["seen_xyz"]
    valid_seen_xyz = model_input["is_valid"]
    unseen_xyz = model_input["unseen_xyz"]
    unseen_rgb = model_input["unseen_rgb"]
    labels = model_input["is_seen"]

    pred_occupy = []
    pred_colors = []

    model.cached_enc_feat = None
    num_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_query_points))
    for p_idx in tqdm(range(num_passes)):
        p_start = p_idx     * max_n_query_points
        p_end = (p_idx + 1) * max_n_query_points
        cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
        cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()
        cur_labels = labels[:, p_start:p_end].zero_()

        with torch.no_grad():
            _, pred = model(
                seen_images=seen_rgb,
                seen_xyz=seen_xyz,
                unseen_xyz=cur_unseen_xyz,
                unseen_rgb=cur_unseen_rgb,
                unseen_occupy=cur_labels,
                cache_enc=True,
                valid_seen_xyz=valid_seen_xyz,
            )
        pred_occupy.append(pred[..., 0].cpu())
        pred_colors.append(
            (
                torch.nn.Softmax(dim=2)(
                    pred[..., 1:].reshape((-1, 3, 256)) / temperature
                ) * torch.linspace(0, 1, 256, device=pred.device)
            ).sum(axis=2)
        )
    
    return torch.cat(pred_occupy, dim=1), torch.cat(pred_colors, dim=0)