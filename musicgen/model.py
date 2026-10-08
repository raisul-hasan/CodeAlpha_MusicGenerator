"""A multi-head LSTM learns pitch, rhythm, articulation, and dynamics."""
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .midi import CARDINALITIES


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 128
    layers: int = 2
    dropout: float = 0.2
    embedding_sizes: tuple[int, ...] = (48, 16, 16, 8)

    def to_dict(self):
        return asdict(self)


class MusicLSTM(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList(nn.Embedding(count, size)
                                        for count, size in zip(CARDINALITIES, config.embedding_sizes))
        self.lstm = nn.LSTM(sum(config.embedding_sizes), config.hidden_size, config.layers,
                            dropout=config.dropout if config.layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.heads = nn.ModuleList(nn.Linear(config.hidden_size, count) for count in CARDINALITIES)

    def forward(self, events: torch.Tensor, state=None):
        features = torch.cat([embedding(events[..., index])
                              for index, embedding in enumerate(self.embeddings)], dim=-1)
        output, state = self.lstm(features, state)
        output = self.dropout(output)
        return [head(output) for head in self.heads], state
