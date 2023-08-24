import argparse
import os

import torch
import torch.multiprocessing as mp
from torch import distributed as dist

from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import (
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.shard import shard
from torchrec.distributed.types import (
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection

import random
from ai_codesign.nonprod.zhengwangmeta.torchrec_tbe.random_input_generator import DistributedInputGenerator
from ai_codesign.nonprod.zhengwangmeta.put_based_nvshmem_tbe.sharding_parameter_recorder import ShardingParameterRecorder
import ai_codesign.nonprod.zhengwangmeta.torchrec_tbe.save_binary_extension as save_binary # @manual
from ai_codesign.nonprod.zhengwangmeta.put_based_nvshmem_tbe.ranking_model_arc_parser import ModelArcParser, OemaeDataLoader
from tqdm import tqdm



parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
parser.add_argument(
    "--nDev",
    type=int,
    default=8,
    help="number of GPUs",
)
parser.add_argument(
    "--sharding_type",
    type=str,
    default="tw",
    choices=["tw", "cw", "rw", "dp"],
    help="sharding type"
)
parser.add_argument(
    "--exp",
    type=str,
    default="fwd",
    choices=["tbe", "fwd", "bwd", "correct_check", "profile"],
    help="[only tbe] or [only fwd] or [fwd+bwd] or [correctness check] or [profiling]"
)
parser.add_argument(
    "--n_loop",
    type=int,
    default=16384,
    help="number of iterations for throughput test"
)
parser.add_argument(
    "--config_file",
    type=str,
    default=None,
    help="Config file of oemae model"
)



def train_process(rank, nDev, args) -> None:
    random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    backend = 'nccl'
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend, rank=rank, world_size=nDev)
    dist.barrier()

    print("{}: finish init dist".format(rank))

    config_file = args.config_file if args.config_file is not None else os.path.join(os.environ['HOME'], "XXX.yaml")
    parser = ModelArcParser(config_file)
    oemae_dataloader = OemaeDataLoader(rank, nBatch=16)

    # create embedding bag
    eb_configs = [
        EmbeddingBagConfig(
            name = table_name,
            embedding_dim = table_info["embedding_dim"],
            num_embeddings = table_info["num_embeddings"],
            feature_names = table_info["features"],
        )
        for table_name, table_info in parser.table_dict.items()
    ]

    model = FusedEmbeddingBagCollection(
        tables=eb_configs,
        optimizer_type=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.01},
        device=torch.device("meta"),
        # location=EmbeddingLocation.DEVICE
    )

    # Define sharding constraints
    sharding_dict = {
        "tw": ShardingType.TABLE_WISE.value,
        "cw": ShardingType.COLUMN_WISE.value,
        "rw": ShardingType.ROW_WISE.value,
        "dp": ShardingType.DATA_PARALLEL.value,
    }
    sharding_constraints = {
        table_name: ParameterConstraints(
        sharding_types=[sharding_dict[args.sharding_type]],
        ) for table_name in parser.table_dict.keys()
    }

    # get sharding plan
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=oemae_dataloader.batch_size,
        storage_reservation=HeuristicalStorageReservation(percentage=0.01),
        constraints=sharding_constraints,
    )
    plan = planner.collective_plan(
        model, get_default_sharders(), dist.GroupMember.WORLD
    )
    if rank == 0:
        print(plan)
    default_group = dist.group.WORLD

    # shard the embedding tables
    sharded_model = shard(
        module=model,
        env=ShardingEnv.from_process_group(default_group),
        plan=plan.get_plan_for_module(""),
        device=device,
    )

    # # ============================ Input Dataloader ================================
    dist_input_generator = DistributedInputGenerator(oemae_dataloader, sharded_model)

    # Save the sharding information for calling the fbgemm-tbe kernel
    output_dir = os.path.join(os.environ['HOME'], "tmp/test_1")
    SPR = ShardingParameterRecorder(sharded_model, plan=plan)
    SPR.print_parameter_list()
    SPR.save_paramter_list(output_dir, rank)


    dist.barrier()
    if args.exp == "correct_check":
        # # Save the index and offset of random generated input batches
        output_data_dir = os.path.join(os.environ['HOME'], "tmp/test_1/data")
        dist_input_generator.save_input(output_data_dir, rank)

        # # # Save embedding table parameters
        # # TODO: support multiple sharding methods
        fbgemm_tbe = sharded_model._lookups[0]._emb_modules[0]._emb_module
        emb_weights = fbgemm_tbe.weights_dev.to('cpu')
        weight_file_name = os.path.join(os.environ['HOME'], "tmp/test_1/weight_{}.bin".format(rank))
        save_binary.save_tensor(emb_weights, weight_file_name, emb_weights.numel()) # index val

        # Comput and save the result
        result_dir = os.path.join(os.environ['HOME'], "tmp/test_1/result")
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        dist_input_generator.reset_iter()

        for i in range(4): # Test 4 input batches
            ctx, disted_input = dist_input_generator.next()
            unpermuted_embedding = sharded_model.compute(ctx, disted_input)
            output = sharded_model.output_dist(ctx, unpermuted_embedding).wait()
            output.values().sum().backward()
            torch.cuda.synchronize()
            flat_tensor = output.values().flatten().cpu()
            file_name = os.path.join(result_dir, "torchrec_all_to_all_result_{}_{}.pt".format(rank, i))
            torch.save(flat_tensor, file_name)

        # Save updated dev_weight
        new_weight = sharded_model._lookups[0]._emb_modules[0]._emb_module.weights_dev
        save_length = int(new_weight.numel() * 0.05) # only save 5% of the weights due to the significant memory footprint of the full weight tensor
        flat_tensor = new_weight.flatten()[0:save_length].cpu()
        file_name = os.path.join(result_dir, "torchrec_all_to_all_bwd_result_{}.pt".format(rank))
        torch.save(flat_tensor, file_name)

    elif args.exp == "profile":
        in_dim = 2048
        out_dim = 4096
        linear_layer = torch.nn.Linear(in_dim, out_dim).to(device)
        input_tensor = torch.randn(out_dim, in_dim).to(device)

        # loop_range = range(nwait + nwarmup + nactive + 1)
        loop_range = range(25)
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=5, # During this phase profiler is not active.
                warmup=10, # During this phase profiler starts tracing, but the results are discarded.
                active=10, # During this phase profiler traces and records data.
                repeat=1), # Specifies an upper bound on the number of cycles.
            with_stack=True
        ) as profiler:
            for _ in loop_range:
                _ = linear_layer(input_tensor)

            for _ in loop_range:
                ctx, disted_input = dist_input_generator.next()
                unpermuted_embedding = sharded_model.compute(ctx, disted_input)
                output = sharded_model.output_dist(ctx, unpermuted_embedding).wait()
                output.values().sum().backward()
                profiler.step()

            profiler.export_chrome_trace("manifold://gpu_traces/tree/nvshmem/{}.json".format(os.getpid()))


    else:  # Throughput Profilining
        n_warm_up = 256
        n_loop = args.n_loop
        loop_range = range(n_loop)
        if rank == 0:
            loop_range = tqdm(loop_range)

        # warm up
        if args.exp == "fwd":
            for _ in range(n_warm_up):
                ctx, disted_input = dist_input_generator.next()
                unpermuted_embedding = sharded_model.compute(ctx, disted_input)
                output = sharded_model.output_dist(ctx, unpermuted_embedding).wait()
                torch.cuda.synchronize()
        elif args.exp == "bwd":
            for _ in range(n_warm_up):
                ctx, disted_input = dist_input_generator.next()
                unpermuted_embedding = sharded_model.compute(ctx, disted_input)
                output = sharded_model.output_dist(ctx, unpermuted_embedding).wait()
                output.values().sum().backward()
                torch.cuda.synchronize()
        elif args.exp == "tbe":
            for _ in loop_range:
                ctx, disted_input = dist_input_generator.next()
                unpermuted_embedding = sharded_model.compute(ctx, disted_input)
                torch.cuda.synchronize()
        dist.barrier()

        # Large GEMM
        linear_layer = torch.nn.Linear(4096, 1024).to(device)
        input_tensor = torch.randn(1024, 4096).to(device)
        linear_layer(input_tensor)

        # TBE throughput test
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in loop_range:
            if args.exp == "fwd":
                ctx, disted_input = dist_input_generator.next()
                unpermuted_embedding = sharded_model.compute(ctx, disted_input)
                output = sharded_model.output_dist(ctx, unpermuted_embedding).wait()
            elif args.exp == "bwd":
                ctx, disted_input = dist_input_generator.next()
                unpermuted_embedding = sharded_model.compute(ctx, disted_input)
                output = sharded_model.output_dist(ctx, unpermuted_embedding).wait()
                output.values().sum().backward()
            elif args.exp == "tbe":
                ctx, disted_input = dist_input_generator.next()
                unpermuted_embedding = sharded_model.compute(ctx, disted_input)
                # torch.cuda.synchronize()
            # torch.cuda.synchronize()
        end_event.record()
        torch.cuda.synchronize()
        total_latency = start_event.elapsed_time(end_event)
        avg_latency = total_latency / (n_loop) # ms
        throughput = 1000 / avg_latency # batch/s
        print("rank:{}, avg_latency:{:.3f} ms/iter, throughput:{:.3f} iters/s".format(rank, avg_latency, throughput))

    if rank == 0:
        print("Exp Config: {}, batch_size:{}, nTable:{}".format(args.exp, oemae_dataloader.batch_size, len(parser.table_dict.keys())))


if __name__ == "__main__":
    args = parser.parse_args()
    nDev = args.nDev

    processes = []
    mp.set_start_method("spawn")

    print("Start Test...")

    for rank in range(nDev):
        p = mp.Process(target=train_process, args=(rank, nDev, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Test Done.")
