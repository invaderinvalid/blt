import torch.nn.functional as F
import torch.nn as nn
import torch
class EntropyPredictor(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.linear = nn.Linear(
            dim,
            256
        )

    def forward(self, x):

        """
        x = [batch, seq, dim]
        """

        logits = self.linear(x)

        probs = F.softmax(
            logits,
            dim=-1
        )

        entropy = -(
            probs * torch.log(
                probs + 1e-8
            )
        ).sum(dim=-1)

        return entropy