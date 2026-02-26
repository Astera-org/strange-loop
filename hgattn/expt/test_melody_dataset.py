import fire
from ..data import melody
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map
import torch

def save_dataset(abc_dir: str, json_file: str):
    fac = melody.MelodyFactory()
    fac.parse(abc_dir)
    fac.save(json_file)


def _load_dataset(
	json_file: str, 
	ctx_len: int,
	use_cls_token: bool,
	num_tempos: int,
	num_tempos_in_train: int,
	output_onehot: bool
):
    fac = melody.MelodyFactory()
    fac.load(json_file)
    train, test = fac.get_datasets(
        ctx_len=ctx_len, 
		use_cls_token=use_cls_token, 
		num_tempos=num_tempos,
		num_tempos_in_train=num_tempos_in_train,
		output_onehot=output_onehot, 
    )
    return train, test

def load_dataset(
	json_file: str, 
	ctx_len: int,
	use_cls_token: bool,
	num_tempos: int,
	num_tempos_in_train: int,
	output_onehot: bool
):
    train, test = _load_dataset(
			json_file, 
			ctx_len,
			use_cls_token,
			num_tempos,
			num_tempos_in_train,
			output_onehot
	)
    item = train[0]
    print(item)

def make_map(device):
    def _mapfn(el):
        if isinstance(el, torch.Tensor):
            return el.to(device)
        return el
    return _mapfn

def speed_test(
    json_file: str, 
    output_onehot: bool, 
    batch_size: int,
    device: str, # cuda or cpu
    report_every: int
):
    train, test = _load_dataset(json_file, output_onehot)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, pin_memory=True)

    device = torch.device(device)
    print(f"Using device: {device}")

    map_fn = make_map(device)

    for batch_idx, item in enumerate(train_loader):
        item = tree_map(map_fn, item)
        if batch_idx % report_every == 0:
            print(f"Loaded batch {batch_idx}")


if __name__ == "__main__":
    cmds = { 
        "load-dataset": load_dataset, 
        "save-dataset": save_dataset,
        "speed-test": speed_test
    }
    fire.Fire(cmds)
