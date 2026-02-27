from dataclasses import dataclass

@dataclass
class StreamvisOpts:
    active: bool
    grpc_uri: str
    flush_every: float
    use_run_handle: str
    run_attrs: RunAttributes

