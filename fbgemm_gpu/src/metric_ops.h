#include <ATen/ATen.h>

namespace fbgemm_gpu {

at::Tensor batch_auc(
    const int64_t num_tasks,
    const at::Tensor& indices,
    const at::Tensor& labels,
    const at::Tensor& weights);

} // namespace fbgemm_gpu
