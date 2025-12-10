
import unittest
import torch
import sys
import os

# Add fbgemm_gpu to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fbgemm_gpu.split_table_batched_embeddings_ops_training_common import generate_vbe_metadata
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode

class VBEMetadataTest(unittest.TestCase):
    def test_input_validation(self):
        offsets = torch.tensor([0, 10])
        feature_dims_cpu = torch.tensor([10, 10]) # 2 features
        device = torch.device("cpu")
        pooling_mode = PoolingMode.SUM

        # Case 1: Empty batch_size_per_feature_per_rank
        with self.assertRaisesRegex(ValueError, "batch_size_per_feature_per_rank cannot be empty"):
            generate_vbe_metadata(offsets, [], pooling_mode, feature_dims_cpu, device)

        # Case 2: Mismatch number of features
        batch_size_per_feature_per_rank = [[1, 1]] # 1 feature
        with self.assertRaisesRegex(ValueError, "does not match number of features"):
            generate_vbe_metadata(offsets, batch_size_per_feature_per_rank, pooling_mode, feature_dims_cpu, device)

        # Case 3: Mismatch number of ranks
        batch_size_per_feature_per_rank = [[1, 1], [1]] # 2 features, but 2nd has 1 rank instead of 2
        with self.assertRaisesRegex(ValueError, "does not match expected number of ranks"):
            generate_vbe_metadata(offsets, batch_size_per_feature_per_rank, pooling_mode, feature_dims_cpu, device)

if __name__ == '__main__':
    unittest.main()
