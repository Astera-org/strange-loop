import random
from typing import Any
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from .models import ModelOpts
from .models.simple import SimpleCompOpts
from .models.generative import GenerativeModelOpts
from .layers.token_embed import TokenEmbedOpts, StdTokenEmbedOpts, ValSharedTokenEmbedOpts
from .layers.input_pos_encoding import (
	InputPosEncodingOpts, NoEncodingOpts, LearnedEncodingOpts, SinusoidalEncodingOpts
)
from .expt.compare_hyper_vs_graph import CompareOpts
from .data.copy_offset import CopyOffsetOpts
from .data.strided_count import StridedCountOpts
from .data.mod_addition import ModAdditionOpts
from .data.expression import InductiveOpts
from .data.polyseries import PolySeriesOpts
from .data.gather_scatter import GatherScatterOpts
from .data import DatasetOpts
from .optim import OptimizerOpts, ScheduleOpts
from .layers.attn import AttentionOpts
from .debug import DebugOpts
from .metrics import MetricOpts
from .logger import LoggerOpts, StreamvisOpts, TextLoggerOpts

def register_resolvers() -> None:
	def random31():
		return random.randint(0, 2**31 - 1)
	OmegaConf.register_new_resolver("random_int", random31)

def register_configs() -> None:
	# 'name' names the 'schema' - this is using a pure Schema approach with a
	# 1-to-1 relation between schema and node.  All overrides are expressed
	# in .yaml files
	cs = ConfigStore.instance()
	cs.store(name="train_opts_schema", node=TrainOpts)
	cs.store(name="run_schema", node=RunOpts)
	cs.store(group="arch", name="generative_model_schema", node=GenerativeModelOpts)
	cs.store(group="arch", name="simple_comp_schema", node=SimpleCompOpts)
	cs.store(group="attn", name="attention_schema", node=AttentionOpts)
	cs.store(group="data", name="copy_offset_schema", node=CopyOffsetOpts)
	cs.store(group="data", name="gather_scatter_schema", node=GatherScatterOpts)
	cs.store(group="optim", name="optimizer_schema", node=OptimizerOpts)
	cs.store(group="sched", name="schedule_schema", node=ScheduleOpts)
	for cls in (StdTokenEmbedOpts, ValSharedTokenEmbedOpts):
		cs.store(group="tok_embed", name=f"{cls.ty}_schema", node=cls)
	for cls in (NoEncodingOpts, LearnedEncodingOpts, SinusoidalEncodingOpts):
		cs.store(group="input_pos_enc", name=f"{cls.ty}_schema", node=cls)
	cs.store(group="logger", name="streamvis_schema", node=StreamvisOpts)
	cs.store(group="logger", name="text_schema", node=TextLoggerOpts)
	cs.store(group="train", name="train_schema", node=TrainOpts)
	cs.store(group="metric", name="metric_schema", node=MetricOpts)
	cs.store(group="debug", name="debug_schema", node=DebugOpts)


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
	arch: ModelOpts 
	attn: AttentionOpts
	data: DatasetOpts 
	optim: OptimizerOpts
	sched: ScheduleOpts
	tok_embed: TokenEmbedOpts
	input_pos_enc: InputPosEncodingOpts
	logger: LoggerOpts
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
	num_trials: int
	do_print_stats: bool
	do_mapreduce: bool
	do_print_raw: bool
	do_print: bool
	do_speedtest: bool
	do_validate: bool
	analyze_step: int|None
	data_seed: int
	iter_seed: int

