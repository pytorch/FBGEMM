import torch
import torch.nn as nn

class HistogramBinningCalibrationByFeature(nn.Module):
    def __init__(self, num_segments, num_bins):
        super().__init__()
        # HistogramBinningCalibrationByFeature
        self._num_segments = num_segments
        self._num_bins = num_bins
        _num_interval = (self._num_segments + 1) * self._num_bins
        _lower_bound = 0
        _upper_bound = 1
        l, u = _lower_bound, _upper_bound
        w = (u - l) / self._num_bins
        self.register_buffer("_boundaries", torch.arange(l + w, u - w / 2, w))
        self.register_buffer(
            "_bin_num_examples",
            torch.empty([_num_interval], dtype=torch.float64).fill_(0.0),
        )  # ConstantFill
        self.register_buffer(
            "_bin_num_positives",
            torch.empty([_num_interval], dtype=torch.float64).fill_(0.0),
        )  # ConstantFill
        self.register_buffer("_bin_ids", torch.arange(_num_interval))

        self._iteration = 0

    def forward_only(self, segment_value, segment_lengths, logit):
        origin_prediction = torch.sigmoid(logit - 0.9162907600402832)
        # HistogramBinningCalibrationByFeature
        _3251 = torch.reshape(origin_prediction, (-1,))  # Reshape
        dense_segment_value = torch.zeros(logit.numel(), dtype=segment_value.dtype)
        offsets = torch.cumsum(segment_lengths, 0) - segment_lengths[0]
        dense_segment_value[offsets] = segment_value[offsets] + 1
        _3257 = dense_segment_value
        _3253 = torch.reshape(segment_lengths, (-1,))
        _3258 = torch.reshape(_3257, (-1,))  # Reshape
        _3259 = _3258.long()  # Cast
        _3260 = torch.empty_like(_3253, dtype=torch.int64).fill_(0)  # ConstantFill
        _3261 = torch.empty_like(_3253, dtype=torch.int64).fill_(1)  # ConstantFill
        _3262 = torch.gt(_3259, self._num_segments)  # GT
        _3263 = torch.gt(_3260, _3259)  # GT
        _3264 = _3253 == _3261  # EQ
        _3265 = torch.where(_3262, _3260, _3259)  # Conditional
        _3266 = torch.where(_3263, _3260, _3265)  # Conditional
        _3267 = torch.where(_3264, _3266, _3260)  # Conditional
        _3268 = torch.bucketize(_3251, boundaries=self._boundaries)  # Bucketize
        _3269 = _3268.long()  # Cast
        _3270 = _3267 * self._num_bins  # Mul
        _3271 = _3269 + _3270  # Add
        _3272 = _3271.int()  # Cast
        _3273 = self._bin_num_positives[_3272.long()]  # Gather
        _3274 = self._bin_num_examples[_3272.long()]  # Gather
        _3275 = _3273 / _3274  # Div
        _3276 = _3275.float()  # Cast
        _3277 = _3276 * 0.9995 + _3251 * 0.0005  # WeightedSum
        _3278 = torch.gt(_3274, 10000.0)  # GT
        _3279 = torch.where(_3278, _3277, _3251.float())  # Conditional
        prediction = torch.reshape(_3279, (-1, 1))  # Reshape
        return prediction


if __name__ == "__main__":
    data_type = torch.float32
    logit = torch.tensor([[-0.0018], [0.0085], [0.0090], [0.0003], [0.0029]]).type(data_type)
    segment_value = torch.tensor([40, 31, 32, 13, 31])
    lengths = torch.tensor([[1], [1], [1], [1], [1]])
    num_bins = 5000
    num_segments = 42
    hb = HistogramBinningCalibrationByFeature(num_segments, num_bins)
    calibrated_prediction = torch.squeeze(hb.forward_only(segment_value, lengths, logit))
    expected_calibrated_prediction = torch.tensor([0.2853, 0.2875, 0.2876, 0.2858, 0.2863]).type(data_type)
    torch.testing.assert_allclose(calibrated_prediction, expected_calibrated_prediction, rtol=1e-03, atol=1e-03,)
    print("Success!")
