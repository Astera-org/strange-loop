import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jaxtyping import Array, Float
import optax
import hydra
from omegaconf import DictConfig, OmegaConf
from ..opts import FitTrilinearOpts, register_configs, register_resolvers
from ..logger import make_logger
from .. import utils
from ..tools import graph
from .. import jfuncs

class TrilinearModel(eqx.Module):
	opts: FitTrilinearOpts = eqx.field(static=True)
	Q: Float[Array, "Q D"]
	W: Float[Array, "W D"]
	T: Float[Array, "T D"]

	def __init__(self, opts: FitTrilinearOpts):
		keys = jax.random.split(jax.random.key(opts.seed), num=3)
		def _init(key, shape):
			return jax.random.normal(key, shape) * opts.d_model ** (-1/6) 

		self.opts = opts
		self.Q = _init(keys[0], (opts.Q, opts.d_model))
		self.W = _init(keys[1], (opts.W, opts.d_model))
		self.T = _init(keys[2], (opts.T, opts.d_model))

	def norms2(self):
		def _mean_norm2(vec):
			return (vec ** 2).sum(axis=1).mean()
		return _mean_norm2(self.Q), _mean_norm2(self.W), _mean_norm2(self.T)

	def loss(
		self, 
		graph_QWT: Float[Array, "Q W T"]
	) -> Float[Array, ""]:
		"""
		"""
		nQ = self.Q.shape[0]
		if self.opts.cell_prob is not None:
			logits_QA = jnp.einsum('qd,td->qt', self.Q, self.T)
		else:
			logits_QWT = jnp.einsum('qd,wd,td->qwt', self.Q, self.W, self.T, optimize="optimal")
			logits_QA = logits_QWT.reshape(nQ, -1)
		graph_QA = graph_QWT.reshape(nQ, -1)
		fn = jfuncs.kl_divergence_barron
		kl_QA = jax.vmap(fn)(logits_QA, graph_QA)
		kl_Q = kl_QA.sum(axis=1)
		# jax.debug.print(
		# 	"min(kl_Q) = {}, logits_QA: {}\ngraph_QA: {}\n", 
		# 	kl_Q.min(), logits_QA, graph_QA
		# )
		return kl_Q.mean()


@hydra.main(config_path="./opts", config_name="fit_trilinear", version_base="1.2")
def main(cfg: DictConfig):
	utils.quiet_loggers()	
	opts: FitTrilinearOpts = OmegaConf.to_object(cfg)

	rng = np.random.default_rng(opts.seed)
	if opts.cell_prob:
		graph_TWQ = graph.simple_graph(opts.T, opts.W, opts.Q, opts.cell_prob, rng)
	else:
		if opts.use_factored:
			graph_TWQ = graph.gen_factored_witness_graph(
				opts.T, opts.W, opts.Q, opts.w_per_row, opts.w_per_col, rng)
		else:
			graph_TWQ = graph.gen_witness_graph(opts.T, opts.W, opts.Q, opts.w_per_link, rng)
	# logits_graph_QWT = jnp.where(graph_TWQ, 0.0, -jnp.inf).transpose(2,1,0)
	logits_graph_QWT = jnp.where(graph_TWQ, 0.0, -1000.0).transpose(2,1,0)

	model = TrilinearModel(opts)
	tx = optax.adamw(
			learning_rate=opts.optim.learning_rate,
			b1=opts.optim.b1,
			b2=opts.optim.b2,
			eps=opts.optim.eps,
			eps_root=opts.optim.eps_root,
			weight_decay=opts.optim.weight_decay,
			nesterov=opts.optim.nesterov,
			)
	opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))

	@eqx.filter_jit
	def make_step(model, opt_state, y):
		def loss_fn(model, y):
			return model.loss(y)
		loss, grads = eqx.filter_value_and_grad(loss_fn)(model, y)
		updates, opt_state = tx.update(grads, opt_state, model, value=loss)
		model = eqx.apply_updates(model, updates)
		return model, opt_state, loss

	logger = make_logger(opts.logger)
	if opts.logger.use_run_handle is not None:
		logger.set_run_handle(opts.logger.use_run_handle)

	logger.start()
	logger.set_run_attributes(
		data_ds_task="fit-trilinear",
		data_graph_factored=opts.use_factored,
		data_n_vocab_witness=opts.W,
		data_n_vocab_query=opts.Q,
		data_n_vocab_target=opts.T,
	)

	for step in range(opts.max_sgd_steps):
		model, opt_state, loss = make_step(model, opt_state, logits_graph_QWT)
		logger.write(sgd_step=step, xent=loss)
		if step % opts.report_every == 0:
			q2, w2, t2 = model.norms2() 
			print(
				f"step: {step}, "
				f"loss: {loss.item():5.4f}, "
				f"q2: {q2.item():5.4f}, "
				f"w2: {w2.item():5.4f}, "
				f"t2: {t2.item():5.4f}"
			)


	logger.stop()


if __name__ == "__main__":
	register_configs()
	register_resolvers()
	main()
