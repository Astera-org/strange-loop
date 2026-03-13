from dataclasses import dataclass

from .models.simple import SimpleCompOpts
from .models.generative import GenerativeModelOpts
from .layers.embed import PosEmbedOpts, TokEmbedOpts
from .expt.compare_hyper_vs_graph import CompareOpts
from .expt.tempo_invariant import TempoInvariantOpts
from .expt.sing_speed import SingSpeedOpts
from .data.melody import MelodyDataOpts
from .data.copy_offset import CopyOffsetOpts
from .optim import OptimizerOpts, ScheduleOpts
from .logger import StreamvisOpts, TextLoggerOpts


@dataclass
class TrainOpts:
	do_compile: bool
	do_test_metrics: bool
	report_every: bool
	num_epochs: int
	batch_size: int
	train_dataset_size: int
	test_dataset_size: int
	start_ds_fraction: float
	epoch_ds_increment: float

@dataclass
class RunOpts:
	arch: SimpleCompOpts|GenerativeModelOpts
	data: CopyOffsetOpts|MelodyDataOpts
	optim: OptimizerOpts
	sched: ScheduleOpts
	logger: StreamvisOpts|TextLoggerOpts
	train: TrainOpts
	seed: int

@dataclass
class TestDatasetOpts:
	data: CopyOffsetOpts|MelodyDataOpts
	batch_size: int
	seed: int

