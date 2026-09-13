from berrytrack.config import TrackerConfig
from berrytrack.types import Tracker


def create_tracker(config: TrackerConfig, fps: float = 30) -> Tracker:
    if config.algorithm == "berrytrack":
        from berrytrack.tracking.berry import BerryTracker

        return BerryTracker(config)
    from berrytrack.tracking.baselines import create_baseline

    return create_baseline(config.algorithm, config, fps=fps)
