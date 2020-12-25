from abc import ABC, abstractmethod
from models import SlowFastModel
from video import Video
from features import Features
import config


class ExtractionStrategy(ABC):

    @abstractmethod
    def extract(self, video_path: str) -> None:
        pass


class SlowFastStrategy(ExtractionStrategy):

    def __init__(self):
        self.model = SlowFastModel()

    def extract(self, video_path: str) -> int:
        video = Video(path=video_path)
        features = Features(path=video_path)

        fps, frames_count = video.metadata
        frames_generator = video.frame_generator()

        num_segments = int(frames_count // config.SEGMENT_LENGTH) - 1
        segment_extract_counter = 0
        while segment_extract_counter < num_segments:
            new_feat = self.model.get_features(
                frames_generator, config.SEGMENT_LENGTH)
            features.write(segment_extract_counter, new_feat)
            yield segment_extract_counter / num_segments
            segment_extract_counter += 1


class FeatureExtraction:
    """
    Feature extraction interface
    """

    def __init__(self):
        self._extraction_strategy = None

    @property
    def extraction_strategy(self) -> ExtractionStrategy:
        return self._extraction_strategy

    @extraction_strategy.setter
    def extraction_strategy(self, new_extractor: ExtractionStrategy) -> None:
        self._extraction_strategy = new_extractor

    def extract(self, video_path: str) -> None:
        """ Extracts the features from a media and saves them

        Args:
        video_path : path to the media that features are going to be extracted
        """
        return self.extraction_strategy.extract(video_path)
