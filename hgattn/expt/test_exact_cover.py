import jax
import jax.numpy as jnp
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from ..tools import graph
from ..tools.exact_2dcover import ExactCover2D
from ..tools.exact_2dcover_ref import ExactCover2DRef, Status
from ..opts import RunOpts, register_configs, register_resolvers
from .. import utils

@hydra.main(config_path="./opts", config_name="exact_cover", version_base="1.2")
def main(cfg: DictConfig):
	utils.quiet_loggers()	

	opts: ExactCoverOpts = OmegaConf.to_object(cfg)

	key = jax.random.key(opts.seed)
	rng = np.random.default_rng(opts.seed)

	if opts.cell_prob:
		graph_RWC = graph.simple_graph(opts.R, opts.W, opts.C, opts.cell_prob, rng)
	else:
		if opts.use_factored:
			graph_RWC = graph.gen_factored_witness_graph(
				opts.R, opts.W, opts.C, opts.w_per_row, opts.w_per_col, rng)
		else:
			graph_RWC = graph.gen_witness_graph(opts.R, opts.W, opts.C, opts.w_per_link, rng)

	if opts.save_graph_path is not None:
		with open(opts.save_graph_path, "wb") as fh:
			np.save(fh, graph_RWC)
			print(f"Saved graph to {opts.save_graph_path}")

	cover2dref = ExactCover2DRef(
			graph_RWC, 
			opts.num_extra_rows, 
			opts.num_extra_witnesses,
			opts.max_backtrack_steps,
			opts.max_propagate_steps,
			opts.max_search_steps,
			rng,
			)

	cover2d = ExactCover2D(
			jnp.array(graph_RWC),
			opts.num_extra_rows, 
			opts.num_extra_witnesses,
			opts.max_backtrack_steps,
			opts.max_propagate_steps,
			opts.max_search_steps,
			)

	print(f"Graph Density: {cover2d.graph_density:5.3f}")

	if opts.do_jax:
		if False:
			print("Jax implementation ==================")
			for i in range(opts.num_tests):
				sub_key = jax.random.fold_in(key, i)
				res = cover2d.search(sub_key)
				validated = cover2d.validate(res.asg)
				print(
					f"{i}: {validated.item()=}, {res.valid.item()=}, "
					f"asg={res.asg}, depth={res.depth} step={res.step}"
				)

		print("Compiled Vmapped Jax =================")
		batch_search = jax.jit(jax.vmap(cover2d.search))
		batch_validate = jax.jit(jax.vmap(cover2d.validate))

		for i in range(opts.num_tests):
			sub_key = jax.random.fold_in(key, i)
			keys = jax.random.split(sub_key, opts.batch_size)
			res = batch_search(keys)
			validated = batch_validate(res.asg)
			in_R = np.array(res.asg.in_R)
			in_W = np.array(res.asg.in_W)
			for b in range(opts.batch_size):
				if res.valid[b] and validated[b]:
					r_inds = ",".join(np.flatnonzero(in_R[b]).astype(str))
					w_inds = ",".join(np.flatnonzero(in_W[b]).astype(str))
					print(
							f"{i} {b}: {res.valid[b]} {validated[b]}\n"
							f"{r_inds}\n{w_inds}\n"
							)
			# print(f"step: {i}, total found: {res.valid.sum().item()}")

	else:
		print("Reference Implementation ===================")
		for i in range(opts.num_tests):
			cfg, res, depth, step = cover2dref.loop_search()
			is_valid = cover2d.validate(res.asg)
			if cfg.status == Status.SOLVED:
				r_inds = ",".join(np.flatnonzero(res.asg.in_R).astype(str))
				w_inds = ",".join(np.flatnonzero(res.asg.in_W).astype(str))
				print(
						f"{i}: {cfg.status=}, {is_valid=}, {depth=}, {step=}\n"
						f"{r_inds}\n{w_inds}\n")


if __name__ == "__main__":
	register_configs()
	register_resolvers()
	main()

