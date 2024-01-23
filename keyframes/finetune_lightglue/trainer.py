if __name__ == "__main__":
    import sys
    import os
    import pathlib

    sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), "../.."))

import pytorch_lightning as pl
import pytorch_lightning.cli as pl_cli
from pytorch_lightning.cli import LightningArgumentParser
import keyframes.finetune_lightglue.pl_modules as pl_modules

"""
Start training: 
python keyframes/finetune_lightglue/trainer.py --config keyframes/finetune_lightglue/config_01.yaml fit

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
