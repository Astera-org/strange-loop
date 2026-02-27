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
		alpha: float
	):
	super().__init__()
	if d_model % 2 != 0:
		raise RuntimeError(
			f"GivensRotation requires an even-sized d_model.  Received {d_model}"
		)

	self.d_model = d_model
	self.num_spaces = int(d_model / 2)
	self.embed_weight = nn.Parameter(torch.randn((self.d_model,)))
	self.pos_weight = nn.Parameter(torch.tensor(1.0))
	self._alpha = torch.tensor(alpha)
	self.register_buffer('alpha', self._alpha)

	def _compute_givens(self, x_CM) -> torch.Tensor: 
		alpha_S = torch.pow(self.alpha, -torch.arange(1, self.num_spaces+1))
		input_BC = torch.einsum('cm, m -> c', x_CM, self.embed_weight)
		pos_C = self.pos_weight * torch.arange(x_CM.shape[1])
		theta_C = input_C + pos_C
		theta_CS = theta_C[:,None] * alpha_S[None,:]
		
		sin_theta_CS = torch.sin(theta_CS)
		cos_theta_CS = torch.cos(theta_CS)

		elems_CS4 = torch.stack([cos_theta_CS, sin_theta_CS, -sin_theta_CS, cos_theta_CS])
		givens_CS22 = elems_CS4.reshape(*elems_CS4.shape[:2], 2, 2)
		return givens_CS22

	def compute_givens(self, x_BCM) -> torch.Tensor:
		return torch.vmap(self._compute_givens)(x_BCM)

	def rotate(self, ) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Compute query and key rotations 
		"""
		q_BCS2 = q_BCD.reshape(B, C, S, 2)
		k_BCS2 = k_BCD.reshape(B, C, S, 2)

		q_rot_BCS2 = torch.einsum('bcsij, bcsi -> bcsj', givens_BCS22, q_BCS2)
		k_rot_BCS2 = torch.einsum('bcsij, bcsi -> bcsj', givens_BCS22, k_BCS2)

		q_rot_BCM = q_rot_BCS2.reshape(B, C, M)
		k_rot_BCM = k_rot_BCS2.reshape(B, C, M)

		return q_rot_BCM, k_rot_BCM



