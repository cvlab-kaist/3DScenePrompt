import os
PATH = '/mnt/cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from finetune.models.utils import get_model_cls
from finetune.schemas import Args


def main():
    args = Args.parse_args()
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
