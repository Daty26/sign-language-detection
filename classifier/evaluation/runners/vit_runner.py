from pathlib import Path
from typing import Any, Dict, Optional

from classifier.evaluation.models.vit import VitConfig, run_vit

def run(split_csv: Path, fold: Optional[int]) -> Dict[str, Any]:
    cfg = VitConfig(split_csv=split_csv, fold=fold)
    return run_vit(cfg)
