import torch
import torch.nn as nn
import torch.nn.functional as F


class Mixtape(nn.Module):
    def __init__(self, vocab_size, d_model, d1, d2, num_gates=4):
        super(Mixtape, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d1 = d1
        self.d2 = d2
        self.num_gates = num_gates

        # Parameters for computing pre-activation gate priors
        self.U = nn.Parameter(torch.randn(self.num_gates, self.d2, self.d1))
        self.v = nn.Parameter(torch.randn(self.vocab_size, self.d2))
        self.u = nn.Parameter(torch.randn(self.num_gates, self.d1))
        self.b = nn.Parameter(torch.randn(self.vocab_size, self.num_gates))

        # Parameters for context embeddings
        self.H = nn.Parameter(
            torch.randn(self.num_gates, self.d_model, self.d1)
        )

        # Token embeddings (not specified in the abstract, assuming needed)
        self.token_embeddings = nn.Parameter(
            torch.randn(self.vocab_size, self.d_model)
        )

    def forward(self, gc):
        batch_size, seq_length, _ = gc.shape

        # Compute context embeddings for each gate
        # Expanded gc to [batch_size, seq_length, 1, d1] for broadcasting
        hc = torch.tanh(
            torch.einsum("kij,btj->btki", self.H, gc)
        )  # (batch_size, seq_length, num_gates, d_model)

        # Compute pre-activation gate priors for each token and gate
        # Expanded gc for broadcasting with different parameters
        lc = (
            torch.einsum(
                "ij,btj->bti",
                self.v,
                torch.tanh(torch.einsum("kij,btj->btki", self.U, gc)),
            )
            + torch.einsum("ij,btj->bti", self.u, gc)
            + self.b[None, None, :, :]
        )  # (batch_size, seq_length, vocab_size, num_gates)

        # Sigmoid tree decomposition
        gamma = torch.sigmoid(
            lc[..., :-1]
        )  # (batch_size, seq_length, vocab_size, num_gates-1)
        pis = [None] * self.num_gates
        pis[0] = gamma[..., 0] * gamma[..., 1]
        pis[1] = gamma[..., 0] * (1 - gamma[..., 1])
        pis[2] = (1 - gamma[..., 0]) * gamma[..., 2]
        pis[3] = (1 - gamma[..., 0]) * (1 - gamma[..., 2])

        # Convert list to tensor
        pi = torch.stack(
            pis, dim=-1
        )  # (batch_size, seq_length, vocab_size, num_gates)
        print(pi.shape)

        # Compute the logit sum for each token using vector gating
        logits = torch.einsum(
            "btki,btik->bti",
            hc,
            torch.einsum("btik,bjk->btikj", pi, self.token_embeddings),
        )
        print(logits.shape)
        probs = F.softmax(
            logits, dim=-1
        )  # (batch_size, seq_length, vocab_size)

        return probs


# Example usage
d_model = 512
d1 = 256
d2 = 128
vocab_size = 10000
seq_length = 20

model = Mixtape(vocab_size=vocab_size, d_model=d_model, d1=d1, d2=d2)
gc = torch.randn(
    10, seq_length, d1
)  # Simulated last-layer hidden states for a batch of 10 with sequence length 20
print(gc.shape)
output = model(gc)
print(output)
