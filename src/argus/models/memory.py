"""Per-node temporal memory (GRU) with dropout and eviction.

See docs/05_ARCHITECTURE.md §4.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from argus.models.norm import make_norm


class NodeMemory(nn.Module):
    """A single GRUCell shared across all nodes, plus LayerNorm on input/hidden.

    State is stored in an external dict keyed by node id so the module is stateless
    across BPTT chunks and supports streaming eviction.
    """

    def __init__(self, d_h: int, dropout: float = 0.1, norm: str = "layernorm") -> None:
        super().__init__()
        self.d_h = d_h
        self.dropout = dropout
        self.gru = nn.GRUCell(d_h, d_h, bias=True)
        self.norm_in = make_norm(d_h, norm)
        self.norm_hid = make_norm(d_h, norm)
        self._drop = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        node_ids: torch.Tensor,
        inputs: torch.Tensor,
        state: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Update memory for a batch of nodes.

        Args:
            node_ids: [N] global node ids
            inputs: [N, d_h] encoder aggregates
            state: current memory dict
        Returns:
            updated_states: dict[int, Tensor] (mutated in place)
            outputs: [N, d_h] new hidden states for the batch
        """
        device = inputs.device
        ids_cpu = node_ids.detach().cpu().tolist()
        # Deliberately NOT detached: within a BPTT chunk, gradient flows from
        # later bins' losses back through this recurrence into the GRU's own
        # weights and earlier bins' encoders. Truncation happens at chunk
        # boundaries only — train/loop.py::run_epoch detaches every stored
        # state right after each chunk's single backward()+step() call
        # (docs/05_ARCHITECTURE.md §4, T_bptt = 8 bins).
        hid = torch.stack(
            [state.get(i, torch.zeros(self.d_h, device=device)) for i in ids_cpu]
        )
        if self._drop is not None and self.training:
            # Memory dropout: zero a node's memory with probability p_mem.
            mask = torch.bernoulli(torch.full((hid.shape[0], 1), 1.0 - self.dropout, device=device))
            hid = hid * mask
        hid = self.norm_hid(hid)
        inp = self.norm_in(inputs)
        new_hid = self.gru(inp, hid)
        for i, h in zip(ids_cpu, new_hid):
            state[i] = h
        return new_hid, state


class MemoryBank:
    """Simple external state container for node memories."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.state: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self.state.clear()

    def evict(self, active_ids: set[int], evict_seconds: float, bin_duration: float) -> None:
        """Placeholder eviction policy; in streaming code this is driven by timestamps.

        For training we detach the state at BPTT chunk boundaries instead.
        """
        pass
