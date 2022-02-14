# Running `batch_benchmark_run.py`
This script acts as a wrapper around the existing `split_table_batched_embeddings_benchmark.py`
benchmark to execute multiple benchmark instances and aggregate the results.

Options for each execution are to be specified in individual lines of an input file that is
passed to the script via the `--command-file` argument.  To accommodate various build
configurations, the command used to invoke `split_table_batched_embeddings_benchmark` instances
is passed to the script via the `--benchmark-command` argument.

An example of a typical execution is:
```
python batch_benchmark_run.py --benchmark-command "python split_table_batched_embeddings_benchmark.py" --command-file batch_input.txt
```

which will provide something like the following output:
```
Running command 0: [<command_0_arguments>]
...
<command_0_output>
...
Running command 1: [<command_1_arguments>]
...
<command_1_output>
...
Number of commands run: 2
Average FWD BW: 1197.9126493108731 GB/s
        FWDBWD BW: 859.5188964175346 GB/s
```

Any commands failed will be reported to ease debugging.

## Expected use-case
This script is intended to be used in conjunction with synthetic datasets provided
in the [dlrm_datasets repository](https://github.com/facebookresearch/dlrm_datasets).
Simply clone this repository to obtain the datasets.

Datasets in this repository provide inputs to the `split_table_batched_embeddings_benchmark.py`
benchmark and can be specified with the `--requests_data_file` argument.  A subset of tables
provided in the input dataset can be used for benchmarking through the `--tables` arguemnt.

Please note that in order to use this feature, dimensions of the tables in the dataset
must conform to the corresponding arguments of the benchmark; these being the following:
* `--batch-size`
* `--num-tables`
* `--num-embeddings`
* `--bag-size`

Hence, a typical line in the input file to `batch_benchmark_run.py` will look something like the following:
```
device --requests_data_file ./fbgemm_t856_bs65536.pt --batch-size 65536 --num-tables 1 --tables "44" --num-embeddings 6618839 --bag-size 194
```

An error will be shown if any of these arguments do not align with the data provided.  This is
in order to ensure proper accounting when metric reporting is performed.
