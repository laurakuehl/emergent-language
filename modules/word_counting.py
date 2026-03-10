import torch
import torch.nn as nn

class WordCountingModule(nn.Module):
    def __init__(self, config):
        super(WordCountingModule, self).__init__()
        self.oov_prob = float(config.oov_prob)
        # Track aggregate token usage over the run; one counter per vocab token.
        self.register_buffer("word_counts", torch.zeros(config.vocab_size))

    def reset(self):
        self.word_counts.zero_()

    def forward(self, utterances):
        # Keep denominator strictly positive to avoid inf/nan at beginning.
        normalizer = torch.clamp(self.oov_prob + self.word_counts.sum(), min=1e-6)
        cost = -(utterances / normalizer).sum()
        self.word_counts.add_(utterances.sum(dim=(0, 1)).detach())
        return cost
