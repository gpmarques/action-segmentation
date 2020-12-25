from abc import ABC, abstractmethod

import torch
import numpy as np
import config

from external.slowfast_feature_extractor.configs.custom_config import load_config
from external.slowfast_feature_extractor.models import build_model
from external.slowfast_feature_extractor.datasets import VideoSet
from external.SlowFast.slowfast.utils import checkpoint


class Model(ABC):

    @abstractmethod
    def get_features(self):
        pass


class SlowFastModel(Model):

    CFG_PATH = "./slowfast_feature_extractor/configs/SLOWFAST_8x8_R50.yaml"

    def __init__(self):
        self.__cfg = load_config(config.SLOWFAST_PATH)
        self.model = build_model(self.__cfg)
        checkpoint.load_test_checkpoint(self.__cfg, self.model)

    def get_features(self, frame_gen, nframes_per_clip):
        dataset = VideoSet(
            self.__cfg, vid_path=None,
            vid_id=None, n_frames=nframes_per_clip,
            frames_generator=frame_gen, read_vid_file=True)

        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.__cfg.TEST.BATCH_SIZE,
            shuffle=False,
            sampler=None,
            num_workers=self.__cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=self.__cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=False,
        )

        return self.perform_inference(test_loader)

    @torch.no_grad()
    def perform_inference(self, test_loader):
        """
        Perform multi-view testing that samples a segment of frames from a
        video and extract features from a pre-trained model.
        Args:
            test_loader (loader): video testing loader
        """
        # Enable eval mode.
        self.model.eval()

        feat_arr = None

        for inputs in test_loader:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Perform the forward pass.
            preds, feat = self.model(inputs)

            feat = feat.cpu().numpy()

            if feat_arr is None:
                feat_arr = feat
            else:
                feat_arr = np.concatenate((feat_arr, feat), axis=0)

        return feat_arr
