"""BPTT chunk-order shuffling (docs/BUGS.md #49).

CICIDS2018's anchor bins are time-ordered and each attack campaign occupies a
contiguous block — the last 10% of the training timeline is 100% `bot`. Walking
that timeline start-to-finish means the final ~900 optimizer steps of every
epoch see a single class, and the model ends the epoch as a near-constant `bot`
predictor: validation accuracy was *exactly* 0.0000 across the 85% of val bins
where `bot` is rare.

These tests pin the fix: training visits BPTT chunks in random order (bins stay
ordered within a chunk, so the TE6 recurrence is untouched), while evaluation
stays strictly sequential.
"""

from __future__ import annotations

import torch

from argus.train.loop import run_epoch


class _StubModel(torch.nn.Module):
    """Records the order bins arrive in, and the labels of the last step."""

    f_v = 4

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        # run_epoch probes model.head for a prototype bank to renormalise.
        self.head = torch.nn.Module()
        self.seen: list[int] = []

    def forward(self, node_feat, *args, memory=None, node_ids=None, **kw):
        # node_feat[0, 0] carries the bin id so we can observe visit order.
        self.seen.append(int(node_feat[0, 0].item()))
        if memory is not None and node_ids is not None:
            for nid in node_ids.tolist():
                memory[nid] = torch.ones(2)
        return {"z": self.lin(node_feat)}


class _StubSource:
    """One bin per class-block: bins 0-9 are class 0, bins 10-19 are class 1 —
    the temporal/class segregation that broke real training."""

    def __init__(self, n_bins: int = 20) -> None:
        self.unique_bins = list(range(n_bins))
        self.reset_calls = 0

    def reset_epoch_state(self) -> None:
        self.reset_calls += 1

    def build_bin_batch(self, bin_id: int, f_v: int = 4) -> dict:
        label = 0 if bin_id < len(self.unique_bins) // 2 else 1
        return {
            "node_feat": torch.full((2, 4), float(bin_id)),
            "node_ids": torch.tensor([bin_id * 2, bin_id * 2 + 1]),
            "scale_short": (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4), torch.zeros(0)),
            "scale_mid": (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4), torch.zeros(0)),
            "scale_long": (torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4), torch.zeros(0)),
            "target_edge_index": torch.zeros(2, 1, dtype=torch.long),
            "target_edge_attr": torch.zeros(1, 4),
            "target_labels": torch.tensor([label]),
            "n_targets": 1,
        }


def _loss_fn(outputs, targets, model):
    return outputs["z"].sum() * 0.0 + outputs["z"].pow(2).mean(), 0


def _run(shuffle: bool, seed: int = 0):
    torch.manual_seed(seed)
    model = _StubModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    run_epoch(_StubSource(), model, opt, _loss_fn, torch.device("cpu"),
              train=True, bptt_chunk=4, log_every_bins=10**9,
              shuffle_chunks=shuffle)
    return model.seen


def test_sequential_when_shuffle_disabled():
    assert _run(shuffle=False) == list(range(20))


def test_every_bin_visited_exactly_once_when_shuffled():
    seen = _run(shuffle=True)
    assert sorted(seen) == list(range(20)), "shuffling must not drop or duplicate bins"


def test_temporal_order_preserved_within_chunk():
    """The BPTT window must still be a real time window — bins ascend inside
    each chunk of 4, even though the chunks themselves are reordered."""
    seen = _run(shuffle=True)
    for start in range(0, len(seen), 4):
        block = seen[start:start + 4]
        assert block == sorted(block), f"chunk {block} is not in temporal order"
        assert block[-1] - block[0] == len(block) - 1, f"chunk {block} is not contiguous"


def test_shuffling_breaks_up_the_single_class_tail():
    """The actual regression: with 2 class-blocks, the final optimizer steps of a
    sequential epoch see only the second class. Shuffled, the last chunk is not
    reliably the class-1 tail."""
    tails = {tuple(_run(shuffle=True, seed=s)[-4:]) for s in range(8)}
    sequential_tail = tuple(range(16, 20))
    assert tails != {sequential_tail}, "shuffling never changed the epoch's final chunk"


def test_eval_pass_is_never_shuffled():
    """Evaluation must stay sequential so memory accumulates in real time order,
    even with shuffle_chunks left at its default."""
    torch.manual_seed(0)
    model = _StubModel()
    run_epoch(_StubSource(), model, None, _loss_fn, torch.device("cpu"),
              train=False, bptt_chunk=4, log_every_bins=10**9)
    assert model.seen == list(range(20))


def test_memory_is_cleared_at_shuffled_chunk_boundaries():
    """A shuffled boundary jumps in time, so node state must not carry across it."""
    torch.manual_seed(0)
    model = _StubModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    mem: dict = {}
    run_epoch(_StubSource(), model, opt, _loss_fn, torch.device("cpu"),
              train=True, bptt_chunk=4, memory_state=mem,
              log_every_bins=10**9, shuffle_chunks=True)
    # Only the final chunk's 4 bins (2 nodes each) may remain.
    assert len(mem) <= 8, f"memory leaked across shuffled chunks: {len(mem)} entries"
