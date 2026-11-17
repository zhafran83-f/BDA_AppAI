"""Register only the three application history tables."""

from .predictions import FashionPrediction, PurchasePrediction, ReviewCluster

__all__ = ["FashionPrediction", "PurchasePrediction", "ReviewCluster"]
