from typing import Any
from dataclasses import dataclass, field
from .models.simple import SimpleCompOpts
from .models.generative import GenerativeModelOpts
from .layers.embed import TokEmbedOpts
from .expt.compare_hyper_vs_graph import CompareOpts
from .data.copy_offset import CopyOffsetOpts
from .data.strided_count import StridedCountOpts
from .data.mod_addition import ModAdditionOpts
from .data.expression import InductiveOpts
from .optim import OptimizerOpts, ScheduleOpts
from .layers.attn import AttentionOpts
from .debug import DebugOpts
from .metrics import MetricOpts
from .logger import StreamvisOpts, TextLoggerOpts


@dataclass
class TrainOpts:
	do_test_metrics: bool
	do_mock_metrics: bool
	num_epochs: int
	batch_size: int
	max_sgd_steps: int
	train_dataset_size: int
	test_dataset_size: int
	start_ds_fraction: float
	epoch_ds_increment: float
	use_label_mask: bool # if True, train on a subset of labels defined by a mask

@dataclass
class RunOpts:
	arch: SimpleCompOpts|GenerativeModelOpts
	attn: AttentionOpts
	data: CopyOffsetOpts|StridedCountOpts|InductiveOpts
	optim: OptimizerOpts
	sched: ScheduleOpts
	embed: TokEmbedOpts
	logger: StreamvisOpts|TextLoggerOpts
	train: TrainOpts
	metric: MetricOpts
	debug: DebugOpts
	seed: int
	init_scale: float
	attrs: dict[str, Any] = field(default_factory=dict)

@dataclass
class TestDatasetOpts:
	data: CopyOffsetOpts|InductiveOpts
	is_train: bool
	dataset_size: int
	num_epochs: int
	batch_size: int
	do_mapreduce: bool
	seed: int

