# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
import numpy as np

from feature_utils import get_path_iterator, dump_feature

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


class HubertMotionFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_motion(self, path, ref_len=None):
        x = np.load(path)
        if ref_len is not None and abs(ref_len - len(x)) > 160:
            logging.warning(f"ref {ref_len} != read {len(x)} ({path})")
        return x

    def get_feats(self, path, ref_len=None):
        x = self.read_motion(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = torch.transpose(x, -1, -2)
                x = F.layer_norm(x, (x.size(dim = -1),))
                x = torch.transpose(x, -1, -2)
            x = torch.unsqueeze(x, 0)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk, :]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
            
        return torch.cat(feat, 1).squeeze(0)


def main(tsv_dir, split, ckpt_path, layer, nshard, rank, feat_dir, max_chunk):
    reader = HubertMotionFeatureReader(ckpt_path, layer, max_chunk)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("ckpt_path")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
