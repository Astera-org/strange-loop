import torch
from torch import nn

class GivensRotation(nn.Module):
	"""
	Constructs a pure rotation operator to be applied to queries and keys.
	The rotation uses a geometric series of angles 
	theta * alpha ^^ -i, for i in [1, S].  
	S is the number of 2D subspaces, equal to floor(d_model / 2).
	theta is computed from input.

	Importantly, the fixed position information is NOT stored in the input, but
	computed on the fly from the input index elements.
	"""
	def __init__(
		self,
		d_model: int,
		n_head: int,
		d_head: int,
	):
		super().__init__()
		if d_head % 2 != 0:
			raise RuntimeError(
					f"GivensRotation requires an even-sized d_head.  Received {d_head}")
		self.d_model = M = d_model
		self.n_head = H = n_head
		self.d_head = D = d_head
		self.num_spaces = int(D / 2)
		self.embed_weight = nn.Parameter(torch.randn((H, M)))
		self.pos_weight = nn.Parameter(torch.full((H,), 1.0))
		theta = torch.pow(10000.0, -torch.arange(0, d_head, 2) / d_head)
		self.register_buffer('theta_steps_S', theta)

	def _compute_givens(self, embed_weight_M, pos_weight, x_CM) -> torch.Tensor: 
		input_C = torch.einsum('cm, m -> c', x_CM, embed_weight_M)
		pos_C = pos_weight * torch.arange(input_C.shape[0]) + input_C 
		theta_CS = self.theta_steps_S[None,:] * pos_C[:,None]
		sin_theta_CS = torch.sin(theta_CS)
		cos_theta_CS = torch.cos(theta_CS)
		elems_CS4 = torch.stack(
				[cos_theta_CS, sin_theta_CS, -sin_theta_CS, cos_theta_CS],
				axis=2
				)
		givens_CS22 = elems_CS4.reshape(*elems_CS4.shape[:2], 2, 2)
		return givens_CS22

	def compute_givens(self, x_BCM) -> torch.Tensor:
		"""
		Compute the Givens matrix, represented as:

		float[batch, ctx, subspace, 2, 2]
		"""
		head_fn = torch.vmap(self._compute_givens, in_dims=(0, 0, None), out_dims=(0,))
		batch_fn = torch.vmap(head_fn, in_dims=(None, None, 0), out_dims=(0,))
		return batch_fn(self.embed_weight, self.pos_weight, x_BCM)

	def rotate(self, givens_BHCS22, proj_BHCD) -> torch.Tensor:
		"""
		Compute query or key rotation
		Input:
		  givens_BHCS22:  float[batch, head, ctx, rot-space, 2, 2]
		  proj_BHCD:      float[batch, head, ctx, head-embed]

		Output:
		  float[batch, head, ctx, head-embed]
		"""
		B, H, C, D = proj_BHCD.shape
		S = self.num_spaces
		proj_BHCS2 = proj_BHCD.reshape(B, H, C, S, 2)
		rot_BHCS2 = torch.einsum('bhcsij, bhcsi -> bhcsj', givens_BHCS22, proj_BHCS2)
		rot_BHCD = rot_BHCS2.reshape(B, H, C, D)
		return rot_BHCD

