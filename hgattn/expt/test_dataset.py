import sys
import hydra
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from .. import data
from ..data import iterator
from .. import utils
from .. import rand
from ..opts import TestDatasetOpts, register_configs, register_resolvers

@hydra.main(config_path="./opts", config_name="test_dataset", version_base="1.2")
def main(cfg: DictConfig):
	opts: TestDatasetOpts = OmegaConf.to_object(cfg)
	if opts.data_seed is None:
		opts.data_seed = rand.get_system_random()
	if opts.iter_seed is None:
		opts.iter_seed = rand.get_system_random()

	utils.quiet_loggers()	
	jnp.set_printoptions(threshold=sys.maxsize, floatmode="fixed", linewidth=215)

	ds = data.make_dataset(opts.data, opts.is_train, opts.data_seed)

	print(f"vocab_size: {ds.vocab_size}")
	# print(ds.run_attrs)

	it = iterator.ShuffleIterator(
		dataset=ds, 
		num_elements=opts.dataset_size, 
		batch_size=opts.batch_size, 
		seed=opts.iter_seed,
		new_epoch_cb=None,
		num_epochs=opts.num_epochs)

	"""
	jaxpr = jax.make_jaxpr(it._get_batch_at_step)(jax.random.key(ds.seed), jnp.int32(0))
	print(f"length of jaxpr.eqns: {len(jaxpr.eqns)}")

	key_B = jax.random.split(jax.random.key(ds.seed), num=256)
	lowered = type(ds)._gen_batch.lower(ds, key_B)
	txt = lowered.as_text()
	print(f"StableHLO text size: {len(txt) / 2**20:.2f} MiB")

	lines = txt.splitlines()
	const_bytes = sum(len(l) for l in lines if "stablehlo.constant" in l)
	print(f"{len(lines)} lines; constant ops: {const_bytes / 2**20:.2f} MiB of text; "
		  f"longest line: {max(map(len, lines)) / 1024:.1f} KiB")
	"""

	key = jax.random.key(opts.data_seed)

	if opts.do_print_stats:
		ds.print_template_stats(key, num_trials=opts.num_trials, batch_size=1024)

	if opts.do_stats:
		results = ds.stats(key, num_trials=opts.num_trials)
		for k, val in results.items():
			print(f"{k:>15}: {val:5.2f}")

	if opts.analyze_step is not None:
		item = it.get_batch_at_step(opts.analyze_step)
		passed, msg = ds.validate_item(item)
		if not passed:
			print(f"Item failed to validate:\n\n{msg}\n")
		ds.print_raw_item(item)

	if opts.do_mapreduce:
		def map_fn(item, *, bias):
			return item.obs_sym.sum() + bias
		def reduce_fn(accu, result):
			return accu + result
		mr = it.mapreduce(map_fn, reduce_fn, 0.0, {"bias": 3.0})
		print(f"mapreduce result:\n{mr}")

	if opts.do_speedtest:
		print(f"starting speedtest")
		for item in it:
			# item = item.to_torch()
			if it.step_idx % 1000 == 0:
				print(f"step: {it.step_idx}")
		print(f"finished speedtest")

	if opts.do_validate:
		print("Validating...")
		for step, item in enumerate(it):
			passed, msg = ds.validate_item(item)
			if not passed:
				print(f"Item at step {step} failed to validate:\n\n{msg}")
				import pdb
				pdb.set_trace()
			if step % 100 == 0:
				print(f"step: {step}")

	if opts.do_print_raw:
		for step, item in enumerate(it):
			if step % 1000 == 0:
				print(f"step: {step}")
			print(ds.print_raw_item(item))

if __name__ == "__main__":
	# jax.config.update("jax_explain_cache_misses", True)
	# jax.config.update("jax_log_compiles", True)
	register_configs()
	register_resolvers()
	main()

