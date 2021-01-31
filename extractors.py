from abc import ABC, abstractmethod
import numpy as np
from gluoncv.data.transforms import video as video_tranforms
from gluoncv.model_zoo import get_model
from mxnet import nd
import math
from enum import Enum

from video import Video


class ExtractionStrategy(ABC):

    model = NotImplemented  # model to extract features
    clip_len = NotImplemented  # input clip frames length

    @abstractmethod
    def _sample_frames(self):
        pass

    def _preprocess(self, frames: np.ndarray, clip_len: int) -> np.ndarray:
        transform_fn = video_tranforms.VideoGroupValTransform(
            size=self.FRAME_SIDE_SIZE,
            mean=self.IMAGENET_MEAN,
            std=self.IMAGENET_SD)
        clips_input = transform_fn(frames)
        clips_input = np.stack(clips_input, axis=0)
        clips_input = clips_input.reshape((-1,) + (clip_len, 3, 224, 224))
        clips_input = np.transpose(clips_input, (0, 2, 1, 3, 4))
        return clips_input

    def extract(self, video: Video) -> None:
        frames_id_list = self._sample_frames(len(video))

        for i, ith_frames_id_list in enumerate(frames_id_list):
            frames = self._preprocess(video(ith_frames_id_list),
                                      clip_len=self.clip_len)

            features = self.model(nd.array(frames)).asnumpy()

            video.features.write(i, features)


class SlowFastStrategy(ExtractionStrategy):

    FRAME_SIDE_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_SD = [0.229, 0.224, 0.225]
    MODEL_NAME = 'slowfast_8x8_resnet50_kinetics400'

    def __init__(self):
        self.model = get_model(self.MODEL_NAME, nclass=400,
                               pretrained=True, feat_ext=True)
        self.clip_len = 48

    def _sample_frames(self, video_len: int) -> list:
        n_batches = math.floor(video_len / 32)

        frames_id_list = [list(range((i-1)*32, i*32, 1)) +  # fast pathway frames
                          list(range((i-1)*32, i*32, 2))   # slow pathway frames
                          for i in range(1, n_batches + 1)]

        if video_len % 32 > 0:  # make sure that all frames are used
            rest_frame_id_list = list(range(video_len - 32, video_len, 1)) +\
                                 list(range(video_len - 32, video_len, 2))
            frames_id_list.append(rest_frame_id_list)

        return frames_id_list


class I3DStrategy(ExtractionStrategy):
    FRAME_SIDE_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_SD = [0.229, 0.224, 0.225]
    MODEL_NAME = 'i3d_nl10_resnet50_v1_kinetics400'

    def __init__(self):
        self.model = get_model(self.MODEL_NAME, nclass=400,
                               pretrained=True, feat_ext=True)
        self.clip_len = 32

    def _sample_frames(self, video_len: int) -> list:
        n_batches = math.floor(video_len / 32)

        frames_id_list = [list(range((i-1)*32, i*32, 1))
                          for i in range(1, n_batches + 1)]

        if video_len % 32 > 0:  # make sure that all frames are used
            rest_frame_id_list = list(range(video_len - 32, video_len, 1))
            frames_id_list.append(rest_frame_id_list)

        return frames_id_list


class ExtractorFactory(Enum):

    SLOWFAST = "Slowfast"
    I3D = "I3D"

    @staticmethod
    def values_list() -> list:
        return [ctype.value for ctype in ExtractorFactory]

    @staticmethod
    def get(cluster_type: str) -> ExtractionStrategy:
        return {
            ExtractorFactory.SLOWFAST.value: SlowFastStrategy,
            ExtractorFactory.I3D.value: I3DStrategy,
        }[cluster_type]
