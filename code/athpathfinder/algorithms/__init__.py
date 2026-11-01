from .family import (
    JaccardFamily,
    PhiPrototypeFamily,
    PhiSpaceFamily,
    SupervisedFamily,
)
from .pathway_like import (
    GOOTPathwayLike,
    JaccardPathwayLike,
    PhiSpacePathwayLike,
    SupervisedPathwayLike,
)
from .two_stage import TwoStageAnnotator, compose_end_to_end, threshold_by_balanced_accuracy

__all__ = [
    "GOOTPathwayLike",
    "JaccardFamily",
    "JaccardPathwayLike",
    "PhiPrototypeFamily",
    "PhiSpaceFamily",
    "PhiSpacePathwayLike",
    "SupervisedFamily",
    "SupervisedPathwayLike",
    "TwoStageAnnotator",
    "compose_end_to_end",
    "threshold_by_balanced_accuracy",
]


def __getattr__(name: str):
    if name in {
        "deployed_stage1",
        "deployed_stage2",
        "family_models",
        "make_models",
        "make_stage1_candidates",
        "make_stage2_candidates",
    }:
        from . import hypothesis_classes
        return getattr(hypothesis_classes, name)
    raise AttributeError(name)
