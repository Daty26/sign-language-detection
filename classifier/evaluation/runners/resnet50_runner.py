from pathlib import Path
from typing import Any, Dict, Optional

from classifier.evaluation.models.resnet50 import ResNet50Config, run_resnet50

def run(split_csv: Path, fold: Optional[int]) -> Dict[str, Any]:
    cfg = ResNet50Config(split_csv=split_csv, fold=fold)
    return run_resnet50(cfg)