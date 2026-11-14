import jax
import jax.numpy as jnp
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from ..tools import graph
from ..tools.exact_2dcover import ExactCover2D
from ..tools.exact_2dcover_ref import ExactCover2DRef
from ..opts import RunOpts, register_configs, register_resolvers
from .. import utils

@hydra.main(config_path="./opts", config_name="exact_cover", version_base="1.2")
def main(cfg: DictConfig):
	utils.quiet_loggers()	

	opts: ExactCoverOpts = OmegaConf.to_object(cfg)

	key = jax.random.key(opts.seed)
	rng = np.random.default_rng(opts.seed)

	if opts.use_factored:
		graph_RWC = graph.gen_factored_witness_graph(
				opts.R, opts.W, opts.C, opts.w_per_row, opts.w_per_col, rng)
	else:
		graph_RWC = graph.gen_witness_graph(opts.R, opts.W, opts.C, opts.w_per_link, rng)

	print(f"Graph Density: {(graph_RWC.sum() / graph_RWC.size):5.3f}")

	cover2Dref = ExactCover2DRef(
			graph_RWC, 
			opts.num_extra_rows, 
			opts.num_extra_witnesses,
			opts.max_backtrack_steps,
			opts.max_propagate_steps,
			opts.max_search_steps,
			rng,
			)

	cover2D = ExactCover2D(
			jnp.array(graph_RWC),
			opts.num_extra_rows, 
			opts.num_extra_witnesses,
			opts.max_backtrack_steps,
			opts.max_propagate_steps,
			opts.max_search_steps,
			)


	if opts.do_jax:
		if False:
			print("Jax implementation ==================")
			for i in range(opts.num_tests):
				sub_key = jax.random.fold_in(key, i)
				res = cover2D.search(sub_key)
				validated = cover2D.validate(res.asg)
				print(
					f"{i}: {validated.item()=}, {res.valid.item()=}, "
					f"asg={res.asg}, depth={res.depth} step={res.step}"
				)

		print("Compiled Vmapped Jax =================")
		batch_search = jax.jit(jax.vmap(cover2D.search))

		for i in range(opts.num_tests):
			sub_key = jax.random.fold_in(key, i)
			keys = jax.random.split(sub_key, opts.batch_size)
			res = batch_search(keys)
			print(f"step: {i}, total found: {res.valid.sum().item()}")



	else:
		print("Reference Implementation ===================")
		for i in range(opts.num_tests):
			cfg, res, depth, step = cover2Dref.loop_search()
			is_valid = cover2D.validate(res.asg)
			print(f"{i}: {cfg.status=}, {is_valid=}, {res.asg}, {depth=} {step=}")


if __name__ == "__main__":
	register_configs()
	register_resolvers()
	main()

