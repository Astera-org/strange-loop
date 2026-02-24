from dataclasses import dataclass
from typing import Literal, Dict, Any
from torch.optim.lr_scheduler import (
    LRScheduler, ReduceLROnPlateau
)
from enum import Enum
from omegaconf import OmegaConf, DictConfig

@dataclass
class OptimizerOpts:
    learning_rate: float
    kind: Literal["adam", "adamw", "sgd"]
    b1: float = None
    b2: float = None
    eps: float = None
    eps_root: float = None
    weight_decay: float = None
    nesterov: bool = None

    def to_json(self):
        d = asdict(self)
        return json.dumps(d)


class ScheduleType(Enum):
    CONSTANT = "constant"
    INV_SQ_ROOT = "inv_sq_root"
    LINEAR_DECAY = "linear_decay"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


@dataclass
class ScheduleOpts:
    ty: ScheduleType
    args: Dict[str, Any]

    def __post_init__(self):
        try:
            self.ty = ScheduleType(self.ty)
            if isinstance(self.args, DictConfig):
                self.args = OmegaConf.to_container(self.args)
        except ValueError as v:
            raise ValueError(
                f"Received invalid schedule_type `{self.ty.value}`.  "
                f"Valid ty's are {', '.join(m.value for m in ScheduleType)}") from v

    def to_json(self):
        d = asdict(self)
        d["ty"] = d["ty"].value
        return json.dumps(d)


def build_schedule(optimizer, opts: ScheduleOpts) -> LRScheduler:
    match opts.ty:
        case ScheduleType.REDUCE_ON_PLATEAU:
            return ReduceLROnPlateau(optimizer, **opts.args)
        case default:
            raise RuntimeError(f"Unrecognized schedule type: {opts.ty}")



