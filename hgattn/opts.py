from dataclasses import dataclass

from .models.simple import SimpleCompOpts
from .expt.compare_hyper_vs_graph import CompareOpts
from .expt.tempo_invariant import TempoInvariantOpts
from .expt.sing_speed import SingSpeedOpts
from .data.melody import MelodyDataOpts
from .optim import OptimizerOpts, ScheduleOpts
from .logger import StreamvisOpts, TextLoggerOpts


@dataclass
class TrainOpts:
	do_compile: bool
	do_test_metrics: bool
	report_every: bool
	num_epochs: int

@dataclass
class RunOpts:
	arch: SimpleCompOpts
	data: DataOpts|MelodyDataOpts
	optim: OptimizerOpts
	sched: ScheduleOpts
	logger: StreamvisOpts|TextLoggerOpts
	train: TrainOpts

