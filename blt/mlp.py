from torch import nn

class MLP(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd
        )

        self.activation = nn.GELU()

        self.proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd
        )

    def forward(self, x):

        # expand dimension
        x = self.fc(x)

        # non linearity
        x = self.activation(x)

        # project back
        x = self.proj(x)

        return x