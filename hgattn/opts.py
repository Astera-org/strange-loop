from dataclasses import dataclass

from .models.simple import SimpleCompOpts
from .models.generative import GenerativeModelOpts
from .layers.embed import TokEmbedOpts
from .expt.compare_hyper_vs_graph import CompareOpts
from .expt.tempo_invariant import TempoInvariantOpts
from .expt.sing_speed import SingSpeedOpts
from .data.melody import MelodyDataOpts
from .data.copy_offset import CopyOffsetOpts
from .optim import OptimizerOpts, ScheduleOpts
from .logger import StreamvisOpts, TextLoggerOpts
from .layers.attn import AttentionOpts
from .debug import DebugOpts



@dataclass
class TrainOpts:
	do_test_metrics: bool
	num_epochs: int
	batch_size: int
	train_dataset_size: int
	test_dataset_size: int
	start_ds_fraction: float
	epoch_ds_increment: float
	use_label_mask: bool # if True, train on a subset of labels defined by a mask

@dataclass
class RunOpts:
	arch: SimpleCompOpts|GenerativeModelOpts
	attn: AttentionOpts
	train_data: CopyOffsetOpts|MelodyDataOpts
	test_data: CopyOffsetOpts|MelodyDataOpts
	optim: OptimizerOpts
	sched: ScheduleOpts
	embed: TokEmbedOpts
	logger: StreamvisOpts|TextLoggerOpts
	train: TrainOpts
	debug: DebugOpts
	seed: int
	code_tweak: str

@dataclass
class TestDatasetOpts:
	data: CopyOffsetOpts
	dataset_size: int
	num_epochs: int
	batch_size: int
	seed: int

