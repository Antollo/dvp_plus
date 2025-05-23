import torch
from lcmr.grammar.scene_data import SceneData
from lcmr.utils.elliptic_fourier_descriptors import reconstruct_contour
from lcmr.utils.guards import typechecked

from lcmr_ext.loss.base_loss import BaseLoss


@typechecked
class EfdRegularizerLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def value(
        self,
        y_true: SceneData,
        y_pred: SceneData,
    ):
        c = reconstruct_contour(y_pred.scene.layer.object.efd).flatten(0, 2)
        deltas = c[:, 1:] - c[:, :-1]
        distances = torch.linalg.vector_norm(deltas, dim=2)
        total_lengths = torch.sum(distances, dim=1).mean()
        return total_lengths
