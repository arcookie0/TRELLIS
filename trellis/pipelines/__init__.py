from . import samplers
from .trellis_image_to_3d import TrellisImageTo3DPipeline
from .trellis_text_to_3d import TrellisTextTo3DPipeline


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder.

    Args:
        path: The path to the model. Must be a local path.
    """
    import os
    import json
    
    # 로컬 파일만 사용
    config_file = f"{path}/pipeline.json"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Pipeline config not found at {config_file}. Please ensure the model files are downloaded locally.")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)
