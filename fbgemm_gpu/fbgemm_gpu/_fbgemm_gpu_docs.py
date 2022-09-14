import torch
import fbgemm_gpu


def add_docs(method, docstr):
    method.__doc__ = docstr


add_docs(
    torch.ops.fbgemm.jagged_2d_to_dense,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.jagged_1d_to_dense,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.dense_to_jagged,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.jagged_to_padded_dense,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_elementwise_add,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_dense_elementwise_add_jagged_output,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.jagged_dense_elementwise_mul,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.batched_dense_vec_jagged_2d_mul,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.stacked_jagged_1d_to_dense,
    """Args:
                {input}
            Keyword args:
                {out}""",
)


add_docs(
    torch.ops.fbgemm.stacked_jagged_2d_to_dense,
    """Args:
                {input}
            Keyword args:
                {out}""",
)
