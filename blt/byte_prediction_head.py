

from torch import nn


class BytePredictionHead(nn.Module):

    def __init__(self, config):

        super().__init__()

        # project embedding
        # to byte vocabulary

        self.linear = nn.Linear(

            config.n_embd,

            256
        )

    def forward(self, x):

        """
        x = [B,T,256]
        """

        logits = self.linear(x)

        return logits
    

