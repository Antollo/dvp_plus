import lpips
from lcmr.grammar.scene_data import SceneData
from lcmr.utils.guards import typechecked

from lcmr_ext.loss.base_loss import BaseLoss


@typechecked
class LPIPSLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.model = lpips.LPIPS(net="vgg")

    def value(
        self,
        y_true: SceneData,
        y_pred: SceneData,
    ):
        return self.model(y_true.image[..., :3].permute(0, 3, 1, 2), y_pred.image[..., :3].permute(0, 3, 1, 2), normalize=True).mean()
