"""``yaml.safe_load`` without exponential merge-key (``<<``) expansion.

PyYAML's ``flatten_mapping`` copies every merged pair into the merging node, duplicates
included, so ``a{i}: {<<: [*a{i-1}, *a{i-1}]}`` doubles the pair list per level: a
few hundred bytes of YAML exhaust the process (CWE-776). Dropping the exact duplicate
keys after each flatten keeps the work linear and constructs the same mapping (a
later pair already overrides an earlier one in ``construct_mapping``).
"""

from __future__ import annotations

from typing import IO

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode

__all__ = ["safe_load"]


class _DedupMergeLoader(yaml.SafeLoader):
    def flatten_mapping(self, node: MappingNode) -> None:
        super().flatten_mapping(node)
        # Key -> (key, value) pair; re-assignment keeps the first position and the last
        # value, exactly what construct_mapping does with duplicate keys.
        pairs: dict[object, tuple[Node, Node]] = {}
        for index, (key_node, value_node) in enumerate(node.value):
            key: object = (
                (key_node.tag, key_node.value) if isinstance(key_node, ScalarNode) else index
            )
            pairs[key] = (key_node, value_node)
        node.value = list(pairs.values())


def safe_load(stream: str | bytes | IO[str] | IO[bytes]) -> object:
    r"""Parse one YAML document with the safe constructor and linear merge-key cost.

    Example:
        >>> safe_load("a: &a {x: 1}\nb: {<<: [*a, *a], y: 2}")
        {'a': {'x': 1}, 'b': {'x': 1, 'y': 2}}
    """
    data: object = yaml.load(stream, Loader=_DedupMergeLoader)
    return data
