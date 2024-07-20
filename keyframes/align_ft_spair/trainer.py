if __name__ == "__main__":
    import sys
    import os
    import pathlib

    sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), "../.."))

import pytorch_lightning as pl
import pytorch_lightning.cli as pl_cli
from pytorch_lightning.cli import LightningArgumentParser
from keyframes.align_ft_spair import pl_modules


"""
Start training: 
python keyframes/align_ft_spair/trainer.py fit --config ./keyframes/align_ft_spair/configs/first_try.yaml

Or 

nohup python trainer.py fit --config /home/user/Documents/projects/Metaworld/keyframes/align_ft/configs/first_try.yaml > train_three_towers.out &

Create Config File:
python keyframes/finetune_lightglue/trainer.py fit --print_config > keyframes/finetune_lightglue/config_sample.yaml
"""


class MyLightningCLI(pl_cli.LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # parser.link_arguments("data.batch_size", "model.batch_size")
        pass


cli = MyLightningCLI(
    pl_modules.PLModel,
    pl_modules.PLData,
    parser_kwargs={"parser_mode": "omegaconf"},  # allows for variable interpolation
    save_config_kwargs={"overwrite": True}
)
