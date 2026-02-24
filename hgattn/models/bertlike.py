from torch import nn
from .simple import SimpleCompModel

class BertlikeModel(SimpleCompModel):
    """
    A SimpleCompModel with a final classifier at token position 0.
    Token position 0 is expected to be a CLS token
    """

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, num_heads:int,
                 n_layers:int, attn_impl:str='', n_recurse:int=1): 
        super().__init__(
            input_dim, hidden_dim, num_heads, n_layers, attn_impl, n_recurse
        )
        self.output_dim = output_dim
        self.classifier = nn.Linear(self.input_dim, output_dim)


    def forward(self, x, b):
        proj = super().forward(x, b)
        out = self.classifier(proj[:,0])
        return out



