class TrackerConfig:
    """Small holder for Ultralytics ByteTrack/BoT-SORT options."""

    def __init__(self, enabled=False, tracker="bytetrack.yaml"):
        self.enabled = enabled
        self.tracker = tracker
