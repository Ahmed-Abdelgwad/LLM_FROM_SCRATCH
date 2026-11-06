"""Key/value cache for generation (D-030).

Without a cache, producing each new token feeds the whole text so far through the model again,
although all but the last position would give the numbers it gave the step before. Causal
attention makes that waste avoidable: the keys and values of a position depend only on that
position and the ones before it, so they never change once computed. The cache keeps them, one
(keys, values) pair per layer, so a step needs only the newest token: one query against the stored
keys, instead of a whole window against itself.

The cache has the shape of the model's window: room for `context_length` positions per layer,
filled from the left. It holds the positions 0 .. length-1 of one text; it cannot slide (when the
text outgrows the window, the model's view shifts and every position's numbers change, so the
caller starts again without a cache; `generate` does this and the result is the same as without
any cache at all).
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self, n_layers: int, context_length: int) -> None:
        self.n_layers = n_layers
        self.context_length = context_length
        self.length = 0  # positions stored; advanced by the model after all layers have run
        self._keys: list[torch.Tensor | None] = [None] * n_layers
        self._values: list[torch.Tensor | None] = [None] * n_layers

    def update(
        self, layer: int, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store the new positions' keys and values (batch, heads, new, head_dim) after the ones
        already there; return all of them, (batch, heads, length + new, head_dim)."""
        new = keys.size(2)
        if self.length + new > self.context_length:
            raise ValueError(
                f"{self.length} stored + {new} new positions exceed the window of "
                f"{self.context_length}"
            )
        stored_k, stored_v = self._keys[layer], self._values[layer]
        if stored_k is None or stored_v is None:
            shape = (keys.size(0), keys.size(1), self.context_length, keys.size(3))
            stored_k = keys.new_empty(shape)
            stored_v = values.new_empty(shape)
            self._keys[layer], self._values[layer] = stored_k, stored_v
        elif stored_k.size(0) != keys.size(0):
            raise ValueError("the batch size changed since the cache was filled")
        stored_k[:, :, self.length : self.length + new] = keys
        stored_v[:, :, self.length : self.length + new] = values
        end = self.length + new
        return stored_k[:, :, :end], stored_v[:, :, :end]

    def advance(self, new: int) -> None:
        self.length += new
