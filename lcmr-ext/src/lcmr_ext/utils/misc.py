import torch
from lcmr.utils.guards import typechecked

@typechecked
def hardmax(logits: torch.Tensor, dim:int=-1) -> torch.Tensor:
    indices = torch.argmax(logits, dim=dim, keepdim=True)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(dim, indices, 1.0)
    return one_hot
