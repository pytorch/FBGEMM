import json
import os

import ai_codesign.nonprod.zhengwangmeta.torchrec_tbe.save_binary_extension as save_binary # @manual
import torch
from torchrec.datasets.random import RandomRecDataset

class RandomInputGenerator():
    """
    Random Input Generator for the benchmarking the performance of the BatchedFusedEmbedding API of TorchRec.
    """

    def __init__(self, args, rank=0, nBatch=16) -> None:
        self._args = args
        self.input_keys = ["table_{}".format(i) for i in range(args.nTable)]
        self.nBatch = nBatch
        # self.dataset = RandomRecDataset(
        #     keys=self.input_keys,
        #     batch_size=args.local_batch_size,
        #     hash_size=args.table_length,
        #     ids_per_feature=args.pooling_factor,
        #     num_dense=1,
        #     num_batches=1,
        # )

        self.kjt_list = []
        for _ in range(nBatch):
            dataset = RandomRecDataset(
                keys=self.input_keys,
                batch_size=args.local_batch_size,
                hash_size=args.table_length,
                ids_per_feature=args.pooling_factor,
                num_dense=1,
                num_batches=1,
            )
            batch = next(iter(dataset))
            self.kjt_list.append(batch.sparse_features.to(rank))
        self.iter=iter(self.kjt_list)



    def next(self):
        try:
            sparse_input = next(self.iter)
            return sparse_input
        except StopIteration:
            self.iter=iter(self.kjt_list)
            sparse_input = next(self.iter)
            return sparse_input



class DistributedInputGenerator():
    """
    Distributed random generated inputs from RandomInputGenerator.
    """
    def __init__(self, input_generator, sharded_model) -> None:
        self.dist_input_list = []
        self.nBatch = input_generator.nBatch

        for _ in range(self.nBatch):
            sparse_input = input_generator.next()
            module_ctx = sharded_model.create_context()
            disted_input = sharded_model.input_dist(module_ctx, sparse_input).wait().wait()
            self.dist_input_list.append((module_ctx, disted_input))

        self.iter=iter(self.dist_input_list)

    def next(self):
        try:
            module_ctx, dist_input = next(self.iter)
            return module_ctx, dist_input
        except StopIteration:
            self.iter=iter(self.dist_input_list)
            module_ctx, dist_input = next(self.iter)
            return module_ctx, dist_input

    def reset_iter(self):
        self.iter=iter(self.dist_input_list)

    def save_input(self, dir_name, rank, nBatch=None):
        nBatch = self.nBatch if nBatch is None else nBatch

        index_list = []
        offset_list = []
        index_len = []
        offset_len = []
        self.reset_iter()
        for _ in range(nBatch):
            _, dist_input = self.next()
            # TODO: support multiple sharding methods
            kjt = dist_input[0]
            indices = kjt.values().cpu().long()
            offsets = kjt.offsets().cpu().long()

            index_list.append(indices)
            offset_list.append(offsets)
            index_len.append(len(indices))
            offset_len.append(len(offsets))

        index_list = torch.cat(index_list, dim=0)
        offset_list = torch.cat(offset_list, dim=0)
        # print("index: rank:{}".format(rank), index_list)
        # print("offset: rank:{}".format(rank), offset_list)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        index_list_file = os.path.join(dir_name, "indices_{}.bin".format(rank))
        offset_list_file = os.path.join(dir_name, "offsets_{}.bin".format(rank))
        save_binary.save_tensor(index_list, index_list_file, index_list.numel()) # index val
        save_binary.save_tensor(offset_list, offset_list_file, offset_list.numel()) # offset val

        len_dict = {"index": index_len, "offset": offset_len}
        len_file = os.path.join(dir_name, "index_and_offset_length_{}.json".format(rank))
        with open(len_file, "w") as f:
            json.dump(len_dict, f)  # index and offset length for each batch
