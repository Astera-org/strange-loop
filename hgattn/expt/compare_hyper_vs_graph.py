import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass

from hgattn.models.simple import SimpleCompModel, SimpleCompOpts
# from hgattn.data import genData4

@dataclass
class CompareOpts:
    arch: SimpleCompOpts
    num_epochs: int
    batch_size: int
    task: int



@hydra.main(config_path="./opts", config_name="compare_hyper_vs_graph", version_base="1.2")
def main(cfg: DictConfig):
    opts: CompareOpts = instantiate(cfg)



if __name__ == "__main__":
    main()
