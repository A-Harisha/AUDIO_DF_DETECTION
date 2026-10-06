"""Evidence faithfulness: does the explanation overlap the truly manipulated region?"""
from .faithfulness import segment_iou

__all__ = ["segment_iou"]
