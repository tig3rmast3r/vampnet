import audiotools

audiotools.ml.BaseModel.INTERN += ["vampnet.modules.**"]
audiotools.ml.BaseModel.EXTERN += ["einops", "flash_attn.modules.mha", "loralib"]

from .transformer import VampNet