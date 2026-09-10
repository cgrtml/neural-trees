"""
Hierarchical mixture of experts.

`HierarchicalMixtureOfExperts` needs PyTorch and is imported on first use.
`HardRoutedExperts`, the numpy export of a trained mixture, does not, so a
model exported elsewhere can be loaded and used without torch.
"""

from typing import TYPE_CHECKING, Any

from .hard_router import HardRoutedExperts

if TYPE_CHECKING:
    from .hierarchical_moe import HierarchicalMixtureOfExperts


def __getattr__(name: str) -> Any:
    if name == "HierarchicalMixtureOfExperts":
        from .hierarchical_moe import HierarchicalMixtureOfExperts

        globals()[name] = HierarchicalMixtureOfExperts
        return HierarchicalMixtureOfExperts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["HardRoutedExperts", "HierarchicalMixtureOfExperts"]
