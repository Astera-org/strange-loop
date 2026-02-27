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
	self.d_model = d_model
	self.num_spaces = int(d_model / 2)
	self.embed_weight = nn.Parameter(torch.randn((self.d_model,)))
	self.pos_weight = nn.Parameter(torch.tensor(1.0))
	self._alpha = torch.tensor(alpha)
	self.register_buffer('alpha', self._alpha)

	def _mult_

	def forward(self, x_BCM, q_BCD, k_BCD) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Compute query and key rotations 
		"""
		alpha_S = torch.pow(self.alpha, -torch.arange(1, self.num_spaces+1))
		input_BC = torch.einsum('bcm, m -> bc', x_BCM, self.embed_weight)
		pos_BC = self.pos_weight * torch.arange(x_BCM.shape[2])
		theta_BC = input_BC + pos_BC
		theta_BCS = theta_BC[:,:,None] * alpha_S[None,None,:]
		B, C, S = theta_BCS.shape
		
		sin_theta_BCS = torch.sin(theta_BCS)
		cos_theta_BCS = torch.cos(theta_BCS)

		elems_BCS4 = torch.stack([cos_theta_BCS, sin_theta_BCS, -sin_theta_BCS, cos_theta_BCS])
		givens_BCS22 = elems_BCS4.reshape(B, C, S, 2, 2)
		q_BCS2 = q_BCD.reshape(B, C, S, 2)
		k_BCS2 = k_BCD.reshape(B, C, S, 2)

		q_









