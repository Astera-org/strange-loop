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
from .data.polyseries import PolySeriesOpts
from .optim import OptimizerOpts, ScheduleOpts
from .layers.attn import AttentionOpts
from .debug import DebugOpts
from .metrics import MetricOpts
from .logger import StreamvisOpts, TextLoggerOpts


@dataclass
class TrainOpts:
	do_test_metrics: bool
	test_metrics_every: int
	do_mock_metrics: bool
	num_epochs: int
	batch_size: int
	test_batch_size: int
	max_sgd_steps: int
	train_dataset_size: int
	test_dataset_size: int
	start_ds_fraction: float
	epoch_ds_increment: float


@dataclass
class RunOpts:
	arch: SimpleCompOpts|GenerativeModelOpts
	attn: AttentionOpts
	data: CopyOffsetOpts|StridedCountOpts|InductiveOpts|PolySeriesOpts
	optim: OptimizerOpts
	sched: ScheduleOpts
	embed: TokEmbedOpts
	logger: StreamvisOpts|TextLoggerOpts
	train: TrainOpts
	metric: MetricOpts
	debug: DebugOpts
	seed: int
	init_scale: float

@dataclass
class TestDatasetOpts:
	data: CopyOffsetOpts|InductiveOpts|PolySeriesOpts
	is_train: bool
	dataset_size: int
	num_epochs: int
	batch_size: int
	do_print_stats: bool
	do_mapreduce: bool
	do_print_raw: bool
	do_print: bool
	do_speedtest: bool
	do_validate: bool
	analyze_step: int|None
	data_seed: int
	iter_seed: int

