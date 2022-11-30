# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

# import soundfile as sf
import torch
import numpy as np
import scipy.signal as signal
# import torchaudio

from feature_utils import get_path_iterator, dump_feature
# from fairseq.data.audio.audio_utils import get_features_or_waveform

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_motion_feature")

class WaveletFeatureReader(object):
    def __init__(self, morlet_w, n_wavelets):
        self.morlet_w = morlet_w
        self.n_wavelets = n_wavelets

    def get_feats(self, path, ref_len=None): # ref len not used but included for compatibility
        x = np.load(path)
        x = (x - np.mean(x, axis = 0, keepdims = True)) / (np.std(x, axis = 0, keepdims = True) + 1e-6)
        
        # perform wavelet transform
        t, dt = np.linspace(0, 1, self.n_wavelets, retstep=True)
        fs = 1/dt
        freq = np.linspace(1, fs/2, self.n_wavelets)
        widths = self.morlet_w*fs / (2*freq*np.pi)
        axes = np.arange(0, np.shape(x)[1])
        transformed = [x]
        for axis in axes:
            sig = x[:, axis]
            transformed.append(np.transpose(np.abs(signal.cwt(sig, signal.morlet2, widths, w=self.morlet_w))))

        transformed = np.concatenate(transformed, axis = 1) # (time, channels)
        return torch.from_numpy(transformed).half().contiguous()

def main(tsv_dir, split, nshard, rank, feat_dir):
    n_wavelets = 25
    morlet_w = 1.
    reader = WaveletFeatureReader(morlet_w, n_wavelets)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
