from dataclasses import dataclass

@dataclass
class DebugOpts:
	do_compile: bool
	report_every: int 
	log_probe_every: int
