from torch import nn
from .simple import SimpleCompModel

class BertlikeModel(SimpleCompModel):
	"""
	A SimpleCompModel with a final classifier at token position 0.
	Token position 0 is expected to be a CLS token
	"""

    def __init__(self, 
                 num_tokens: int, model_dim: int, mlp_hidden_dim: int,
                 output_dim: int, num_heads: int, n_layers: int, attn_impl: str='',
                 n_recurse: int=1
                 ): 
		super().__init__(
                num_tokens, model_dim, mlp_hidden_dim, num_heads, n_layers,
                attn_impl, n_recurse
				)
		self.output_dim = output_dim
		self.classifier = nn.Linear(model_dim, output_dim)

	def forward(self, x, mask):
		proj = super().forward(x, mask)
		out = self.classifier(proj[:,0])
		return out



