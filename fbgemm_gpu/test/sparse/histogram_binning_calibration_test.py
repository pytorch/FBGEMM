#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors[56]

import random
import unittest

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

from .common import extend_test_class, open_source

if open_source:
    # pyre-ignore[21]
    from test_utils import gpu_unavailable
else:
    import fbgemm_gpu.sparse_ops  # noqa: F401, E402
    from fbgemm_gpu.test.test_utils import gpu_unavailable


class HistogramBinningCalibrationTest(unittest.TestCase):
    @given(data_type=st.sampled_from([torch.bfloat16, torch.half, torch.float32]))
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_histogram_binning_calibration(self, data_type: torch.dtype) -> None:
        num_bins = 5000

        logit = torch.tensor([[-0.0018], [0.0085], [0.0090], [0.0003], [0.0029]]).type(
            data_type
        )

        bin_num_examples = torch.empty([num_bins], dtype=torch.float64).fill_(0.0)
        bin_num_positives = torch.empty([num_bins], dtype=torch.float64).fill_(0.0)

        calibrated_prediction, bin_ids = torch.ops.fbgemm.histogram_binning_calibration(
            logit=logit,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            positive_weight=0.4,
            lower_bound=0.0,
            upper_bound=1.0,
            bin_ctr_in_use_after=10000,
            bin_ctr_weight_value=0.9995,
        )

        expected_calibrated_prediction = torch.tensor(
            [[0.2853], [0.2875], [0.2876], [0.2858], [0.2863]]
        ).type(data_type)
        expected_bin_ids = torch.tensor(
            [1426, 1437, 1437, 1428, 1431], dtype=torch.long
        )

        error_tolerance = 1e-03
        if data_type == torch.bfloat16:
            # Due to smaller significand bits.
            error_tolerance = 1e-02

            expected_bin_ids = torch.tensor(
                [1426, 1438, 1438, 1430, 1430], dtype=torch.long
            )

        torch.testing.assert_close(
            calibrated_prediction,
            expected_calibrated_prediction,
            rtol=error_tolerance,
            atol=error_tolerance,
        )

        self.assertTrue(
            torch.equal(
                bin_ids.long(),
                expected_bin_ids,
            )
        )

        if torch.cuda.is_available():
            (
                calibrated_prediction_gpu,
                bin_ids_gpu,
            ) = torch.ops.fbgemm.histogram_binning_calibration(
                logit=logit.cuda(),
                bin_num_examples=bin_num_examples.cuda(),
                bin_num_positives=bin_num_positives.cuda(),
                positive_weight=0.4,
                lower_bound=0.0,
                upper_bound=1.0,
                bin_ctr_in_use_after=10000,
                bin_ctr_weight_value=0.9995,
            )

            torch.testing.assert_close(
                calibrated_prediction_gpu,
                expected_calibrated_prediction.cuda(),
                rtol=error_tolerance,
                atol=error_tolerance,
            )

            self.assertTrue(
                torch.equal(
                    bin_ids_gpu.long(),
                    expected_bin_ids.cuda(),
                )
            )

    @given(
        data_type=st.sampled_from([torch.bfloat16, torch.half, torch.float32]),
        segment_value_type=st.sampled_from([torch.int, torch.long]),
        segment_length_type=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_histogram_binning_calibration_by_feature(
        self,
        data_type: torch.dtype,
        segment_value_type: torch.dtype,
        segment_length_type: torch.dtype,
    ) -> None:
        num_bins = 5000
        num_segments = 42

        logit = torch.tensor([-0.0018, 0.0085, 0.0090, 0.0003, 0.0029]).type(data_type)

        segment_value = torch.tensor([40, 31, 32, 13, 31]).type(segment_value_type)
        lengths = torch.tensor([[1], [1], [1], [1], [1]]).type(segment_length_type)

        num_interval = num_bins * (num_segments + 1)
        bin_num_examples = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)
        bin_num_positives = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)

        (
            calibrated_prediction,
            bin_ids,
        ) = torch.ops.fbgemm.histogram_binning_calibration_by_feature(
            logit=logit,
            segment_value=segment_value,
            segment_lengths=lengths,
            num_segments=num_segments,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            num_bins=num_bins,
            positive_weight=0.4,
            lower_bound=0.0,
            upper_bound=1.0,
            bin_ctr_in_use_after=10000,
            bin_ctr_weight_value=0.9995,
        )

        expected_calibrated_prediction = torch.tensor(
            [0.2853, 0.2875, 0.2876, 0.2858, 0.2863]
        ).type(data_type)
        expected_bin_ids = torch.tensor(
            [206426, 161437, 166437, 71428, 161431], dtype=torch.long
        )

        error_tolerance = 1e-03
        if data_type == torch.bfloat16:
            # Due to smaller significand bits.
            error_tolerance = 1e-02

            expected_bin_ids = torch.tensor(
                [206426, 161438, 166438, 71430, 161430], dtype=torch.long
            )

        torch.testing.assert_close(
            calibrated_prediction,
            expected_calibrated_prediction,
            rtol=error_tolerance,
            atol=error_tolerance,
        )

        self.assertTrue(
            torch.equal(
                bin_ids.long(),
                expected_bin_ids,
            )
        )

        if torch.cuda.is_available():
            (
                calibrated_prediction_gpu,
                bin_ids_gpu,
            ) = torch.ops.fbgemm.histogram_binning_calibration_by_feature(
                logit=logit.cuda(),
                segment_value=segment_value.cuda(),
                segment_lengths=lengths.cuda(),
                num_segments=num_segments,
                bin_num_examples=bin_num_examples.cuda(),
                bin_num_positives=bin_num_positives.cuda(),
                num_bins=num_bins,
                positive_weight=0.4,
                lower_bound=0.0,
                upper_bound=1.0,
                bin_ctr_in_use_after=10000,
                bin_ctr_weight_value=0.9995,
            )

            torch.testing.assert_close(
                calibrated_prediction_gpu,
                expected_calibrated_prediction.cuda(),
                rtol=error_tolerance,
                atol=error_tolerance,
            )

            self.assertTrue(
                torch.equal(
                    bin_ids_gpu.long(),
                    expected_bin_ids.cuda(),
                )
            )

    @given(
        data_type=st.sampled_from([torch.bfloat16, torch.half, torch.float32]),
        segment_value_type=st.sampled_from([torch.int, torch.long]),
        segment_length_type=st.sampled_from([torch.int, torch.long]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_generic_histogram_binning_calibration_by_feature(
        self,
        data_type: torch.dtype,
        segment_value_type: torch.dtype,
        segment_length_type: torch.dtype,
    ) -> None:
        num_bins = 5000
        num_segments = 42

        logit = torch.tensor([-0.0018, 0.0085, 0.0090, 0.0003, 0.0029]).type(data_type)

        segment_value = torch.tensor([40, 31, 32, 13, 31]).type(segment_value_type)
        lengths = torch.tensor([[1], [1], [1], [1], [1]]).type(segment_length_type)

        num_interval = num_bins * (num_segments + 1)
        bin_num_examples = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)
        bin_num_positives = torch.empty([num_interval], dtype=torch.float64).fill_(0.0)

        lower_bound = 0.0
        upper_bound = 1.0
        w = (upper_bound - lower_bound) / num_bins
        bin_boundaries = torch.arange(
            lower_bound + w, upper_bound - w / 2, w, dtype=torch.float64
        )

        (
            calibrated_prediction,
            bin_ids,
        ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            logit=logit,
            segment_value=segment_value,
            segment_lengths=lengths,
            num_segments=num_segments,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            bin_boundaries=bin_boundaries,
            positive_weight=0.4,
            bin_ctr_in_use_after=10000,
            bin_ctr_weight_value=0.9995,
        )

        expected_calibrated_prediction = torch.tensor(
            [0.2853, 0.2875, 0.2876, 0.2858, 0.2863]
        ).type(data_type)
        expected_bin_ids = torch.tensor(
            [206426, 161437, 166437, 71428, 161431], dtype=torch.long
        )

        error_tolerance = 1e-03
        if data_type == torch.bfloat16:
            # Due to smaller significand bits.
            error_tolerance = 1e-02

            expected_bin_ids = torch.tensor(
                [206426, 161438, 166438, 71430, 161430], dtype=torch.long
            )

        torch.testing.assert_close(
            calibrated_prediction,
            expected_calibrated_prediction,
            rtol=error_tolerance,
            atol=error_tolerance,
        )

        self.assertTrue(
            torch.equal(
                bin_ids.long(),
                expected_bin_ids,
            )
        )

        if torch.cuda.is_available():
            (
                calibrated_prediction_gpu,
                bin_ids_gpu,
            ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
                logit=logit.cuda(),
                segment_value=segment_value.cuda(),
                segment_lengths=lengths.cuda(),
                num_segments=num_segments,
                bin_num_examples=bin_num_examples.cuda(),
                bin_num_positives=bin_num_positives.cuda(),
                bin_boundaries=bin_boundaries.cuda(),
                positive_weight=0.4,
                bin_ctr_in_use_after=10000,
                bin_ctr_weight_value=0.9995,
            )

            torch.testing.assert_close(
                calibrated_prediction_gpu,
                expected_calibrated_prediction.cuda(),
                rtol=error_tolerance,
                atol=error_tolerance,
            )

            self.assertTrue(
                torch.equal(
                    bin_ids_gpu.long(),
                    expected_bin_ids.cuda(),
                )
            )

    @unittest.skipIf(*gpu_unavailable)
    @given(
        data_type=st.sampled_from([torch.bfloat16, torch.half, torch.float32]),
    )
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_generic_histogram_binning_calibration_by_feature_cpu_gpu(
        self,
        data_type: torch.dtype,
    ) -> None:
        num_logits = random.randint(8, 16)
        num_bins = random.randint(3, 8)
        num_segments = random.randint(3, 8)
        positive_weight = random.uniform(0.1, 1.0)
        bin_ctr_in_use_after = random.randint(0, 10)
        bin_ctr_weight_value = random.random()

        logit = torch.randn(num_logits).type(data_type)

        lengths = torch.randint(0, 2, (num_logits,))
        segment_value = torch.randint(-3, num_segments + 3, (sum(lengths),))

        num_interval = num_bins * (num_segments + 1)
        bin_num_positives = torch.randint(0, 10, (num_interval,)).double()
        bin_num_examples = (
            bin_num_positives + torch.randint(0, 10, (num_interval,)).double()
        )

        lower_bound = 0.0
        upper_bound = 1.0
        w = (upper_bound - lower_bound) / num_bins
        bin_boundaries = torch.arange(
            lower_bound + w, upper_bound - w / 2, w, dtype=torch.float64
        )

        (
            calibrated_prediction_cpu,
            bin_ids_cpu,
        ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            logit=logit,
            segment_value=segment_value,
            segment_lengths=lengths,
            num_segments=num_segments,
            bin_num_examples=bin_num_examples,
            bin_num_positives=bin_num_positives,
            bin_boundaries=bin_boundaries,
            positive_weight=positive_weight,
            bin_ctr_in_use_after=bin_ctr_in_use_after,
            bin_ctr_weight_value=bin_ctr_weight_value,
        )

        (
            calibrated_prediction_gpu,
            bin_ids_gpu,
        ) = torch.ops.fbgemm.generic_histogram_binning_calibration_by_feature(
            logit=logit.cuda(),
            segment_value=segment_value.cuda(),
            segment_lengths=lengths.cuda(),
            num_segments=num_segments,
            bin_num_examples=bin_num_examples.cuda(),
            bin_num_positives=bin_num_positives.cuda(),
            bin_boundaries=bin_boundaries.cuda(),
            positive_weight=positive_weight,
            bin_ctr_in_use_after=bin_ctr_in_use_after,
            bin_ctr_weight_value=bin_ctr_weight_value,
        )

        torch.testing.assert_close(
            calibrated_prediction_cpu,
            calibrated_prediction_gpu.cpu(),
            rtol=1e-03,
            atol=1e-03,
        )

        self.assertTrue(
            torch.equal(
                bin_ids_cpu,
                bin_ids_gpu.cpu(),
            )
        )


extend_test_class(HistogramBinningCalibrationTest)

if __name__ == "__main__":
    unittest.main()
