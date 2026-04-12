
from enum import Enum

class DatasetFeature(Enum):
    TIMESERIES = "timeseries"
    TIMESTAMP = "timestamp"
    TIMESLIDE = "timeslide"
    LABELS = "labels"
