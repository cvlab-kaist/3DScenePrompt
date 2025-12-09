from ..cogvideox_i2v.cond_lora_trainer import CogVideoXI2VCondLoraTrainer
from ..utils import register


class CogVideoX1_5I2VCondLoraTrainer(CogVideoXI2VCondLoraTrainer):
    pass


register("cogvideox1.5-i2v", "cond_lora", CogVideoX1_5I2VCondLoraTrainer)
