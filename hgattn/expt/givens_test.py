import torch
import fire
from ..layers.givens_rotation import GivensRotation
from rotary_embedding_torch import RotaryEmbedding 

def main():
	n_ctx, d_model, n_heads, d_head = 100, 50, 5, 10
	batch = 1
	rot = RotaryEmbedding(d_head)
	giv = GivensRotation(d_model, n_heads, d_head)
	
	with torch.no_grad():
		giv.embed_weight.zero_()
	
	x = torch.randn((batch, n_ctx, d_model))
	q = torch.randn((batch, n_heads, n_ctx, d_head))

	qr = rot.rotate_queries_or_keys(q)

	g_mat = giv.compute_givens(x)
	qg = giv.rotate(g_mat, q)

	if torch.allclose(qr, qg):
		print("Passed")
	else:
		max_diff2 = ((qr - qg) ** 2).max()
		print(f"max diff2: {max_diff2.item()}")

if __name__ == "__main__":
	fire.Fire(main)
