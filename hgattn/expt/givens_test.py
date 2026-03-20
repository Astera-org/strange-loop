import torch
import fire
from ..layers.givens_rotation import GivensRotation
from rotary_embedding_torch import RotaryEmbedding 

def main(seed=42):
	torch.random.manual_seed(seed)
	n_ctx, d_model, n_heads, d_head = 100, 50, 5, 10
	batch = 1
	rot = RotaryEmbedding(d_head)
	giv = GivensRotation(d_model, n_heads, d_head)
	
	# These are the parameter settings for which RoPE is a special case of Givens
	with torch.no_grad():
		giv.embed_weight.zero_()
		giv.embed_weight[:,-1] = 1.0
	
	x = torch.randn((batch, n_ctx, d_model))
	q = torch.randn((batch, n_heads, n_ctx, d_head))

	x[:,:,-1] = torch.arange(n_ctx)

	qr = rot.rotate_queries_or_keys(q)

	g_mat = giv.compute_givens(x)
	qg = giv.rotate(g_mat, q)

	if torch.allclose(qr, qg):
		print("Passed")
	else:
		sd = (qr - qg).std()
		print(f"sd of diff: {sd.item()}")

if __name__ == "__main__":
	fire.Fire(main)
