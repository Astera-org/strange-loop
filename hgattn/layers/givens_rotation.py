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
		alpha: float
	):
	super().__init__()
	if d_head % 2 != 0:
		raise RuntimeError(
			f"GivensRotation requires an even-sized d_head.  Received {d_head}"
		)

	self.d_model = M = d_model
	self.n_head = H = n_head
	self.d_head = D = d_head
	self.num_spaces = int(D / 2)
	self.embed_weight = nn.Parameter(torch.randn((H, M)))
	self.pos_weight = nn.Parameter(torch.full((H,), 1.0))
	self._alpha = torch.tensor(alpha)
	self.register_buffer('alpha', self._alpha)

	def _compute_givens(self, embed_weight_M, pos_weight, x_CM) -> torch.Tensor: 
		alpha_S = torch.pow(self.alpha, -torch.arange(1, self.num_spaces+1))
		input_C = torch.einsum('cm, m -> c', x_CM, embed_weight_M)
		pos_C = pos_weight * torch.arange(x_CM.shape[1])
		theta_C = input_C + pos_C
		theta_CS = theta_C[:,None] * alpha_S[None,:]
		sin_theta_CS = torch.sin(theta_CS)
		cos_theta_CS = torch.cos(theta_CS)
		elems_CS4 = torch.stack([cos_theta_CS, sin_theta_CS, -sin_theta_CS, cos_theta_CS])
		givens_CS22 = elems_CS4.reshape(*elems_CS4.shape[:2], 2, 2)
		return givens_CS22

	def compute_givens(self, x_BCM) -> torch.Tensor:
		head_fn = torch.vmap(self._compute_givens, in_dims=(0, 0, None), out_dims=(0,))
		batch_fn = torch.vmap(head_fn, in_dims=(None, None, 0), out_dims=(0,))
		return batch_fn(self.embed_weight, self.pos_weight, x_BCM)

	def rotate(self, givens_BCS22, proj_BCH) -> torch.Tensor:
		"""
		Compute query or key rotation
		"""
		B, C, H = proj_BCH.shape
		S = self.num_spaces
		proj_BCS2 = proj_BCH.reshape(B, C, S, 2)
		rot_BCS2 = torch.einsum('bcsij, bcsi -> bcsj', givens_BCS22, proj_BCS2)
		rot_BCH = q_rot_BCS2.reshape(B, C, H)
		return rot_BCH


