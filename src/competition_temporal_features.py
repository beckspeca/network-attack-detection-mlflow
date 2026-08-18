"""Competition-safe temporal features for test rows using train history only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from attached_temporal_features import ROLLING_SOURCE_COLUMNS, TIMESTAMP


def build_test_features_from_train_history(
    train_history: pd.DataFrame,
    test_frame: pd.DataFrame,
    selected_flow_features: list[str],
    port_map: dict[str, int],
) -> pd.DataFrame:
    """Create test features from prior train events without feeding test rows into history.

    The v13/v15 training features are generated from train events only. The public and
    private test rows are interleaved with train rows in event time, so computing rolling
    windows from test rows alone creates a large density shift. This function queries the
    past train-event stream for every test row. Test rows never contribute to any feature,
    including features of later test rows.
    """

    train_values = train_history.loc[:, ROLLING_SOURCE_COLUMNS].copy()
    test_values = test_frame.loc[:, ROLLING_SOURCE_COLUMNS].copy()
    train_values[TIMESTAMP] = pd.to_datetime(train_values[TIMESTAMP])
    test_values[TIMESTAMP] = pd.to_datetime(test_values[TIMESTAMP])

    train_values["__is_train"] = True
    train_values["__test_position"] = -1
    test_values["__is_train"] = False
    test_values["__test_position"] = np.arange(len(test_values), dtype=np.int64)
    combined = pd.concat([train_values, test_values], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP, kind="mergesort").set_index(TIMESTAMP)

    is_train = combined["__is_train"].to_numpy(dtype=bool)
    test_positions = combined["__test_position"].to_numpy(dtype=np.int64)
    output_mask = test_positions >= 0
    output_positions = test_positions[output_mask]

    def take_test(values) -> np.ndarray:
        array = np.asarray(values)
        output = np.empty(len(test_frame), dtype=array.dtype)
        output[output_positions] = array[output_mask]
        return output

    temporal = pd.DataFrame(index=np.arange(len(test_frame)))
    train_timestamps = train_values[TIMESTAMP].sort_values().to_numpy(dtype="datetime64[ns]")
    query_timestamps = test_values[TIMESTAMP].to_numpy(dtype="datetime64[ns]")
    previous_positions = np.searchsorted(train_timestamps, query_timestamps, side="left") - 1
    time_since_previous = np.zeros(len(test_frame), dtype=np.float64)
    has_previous = previous_positions >= 0
    time_since_previous[has_previous] = (
        query_timestamps[has_previous] - train_timestamps[previous_positions[has_previous]]
    ) / np.timedelta64(1, "s")
    temporal["time_since_prev_flow"] = time_since_previous

    fwd = combined["Tot Fwd Pkts"].where(combined["__is_train"])
    bwd = combined["Tot Bwd Pkts"].where(combined["__is_train"])
    fwd_bytes = combined["TotLen Fwd Pkts"].where(combined["__is_train"])
    bwd_bytes = combined["TotLen Bwd Pkts"].where(combined["__is_train"])
    syn = combined["SYN Flag Cnt"].where(combined["__is_train"])
    total_packets = fwd + bwd
    total_bytes = fwd_bytes + bwd_bytes

    flow_count_10s = fwd.rolling("10s", closed="left").count()
    flow_count_60s = fwd.rolling("60s", closed="left").count()
    packet_count_10s = total_packets.rolling("10s", closed="left").sum()
    byte_count_10s = total_bytes.rolling("10s", closed="left").sum()
    syn_count_10s = syn.rolling("10s", closed="left").sum()
    temporal["flow_count_10s"] = take_test(flow_count_10s)
    temporal["flow_count_60s"] = take_test(flow_count_60s)
    temporal["packet_count_10s"] = take_test(packet_count_10s)
    temporal["byte_count_10s"] = take_test(byte_count_10s)
    temporal["syn_count_10s"] = take_test(syn_count_10s)
    temporal["packet_rate_10s"] = temporal["packet_count_10s"] / 10.0
    temporal["byte_rate_10s"] = temporal["byte_count_10s"] / 10.0

    dst_flow_count = combined.assign(__train_fwd=fwd).groupby("Dst Port")["__train_fwd"].transform(
        lambda series: series.rolling("10s", closed="left").count()
    )
    temporal["dst_flow_count_10s"] = take_test(dst_flow_count)

    fwd60 = fwd.rolling("60s", closed="left").sum()
    bwd60 = bwd.rolling("60s", closed="left").sum()
    fwd10 = fwd.rolling("10s", closed="left").sum()
    bwd10 = bwd.rolling("10s", closed="left").sum()
    packet60 = total_packets.rolling("60s", closed="left").sum()
    temporal["fwd_bwd_ratio_60s"] = take_test(fwd60) / (take_test(bwd60) + 1)
    temporal["fwd_bwd_ratio_10s"] = take_test(fwd10) / (take_test(bwd10) + 1)
    temporal["burst_ratio_10s"] = temporal["packet_count_10s"] / (take_test(packet60) + 1)

    categorical = pd.Categorical(combined["Dst Port"])
    onehot = pd.DataFrame(
        {
            index: ((categorical.codes == index) & is_train).astype(np.int8)
            for index in range(len(categorical.categories))
        },
        index=combined.index,
    )
    present10 = onehot.rolling("10s", closed="left").sum() > 0
    present60 = onehot.rolling("60s", closed="left").sum() > 0
    temporal["dstport_distinct_10s"] = take_test(present10.sum(axis=1))
    temporal["dstport_distinct_60s"] = take_test(present60.sum(axis=1))
    temporal["port_diversity_10s"] = temporal["dstport_distinct_10s"] / (
        temporal["flow_count_10s"] + 1
    )

    unanswered = ((combined["Tot Bwd Pkts"] == 0) & combined["__is_train"]).astype(np.int8)
    unanswered_count = unanswered.rolling("10s", closed="left").sum()
    temporal["syn_unanswered_rate_10s"] = take_test(unanswered_count) / (
        temporal["flow_count_10s"] + 1
    )
    temporal["syn_per_bwd_10s"] = temporal["syn_count_10s"] / (take_test(bwd10) + 1)

    flow = test_frame.loc[:, selected_flow_features].copy().reset_index(drop=True)
    flow["Dst Port"] = (
        test_frame["Dst Port"].astype("string").map(port_map).fillna(-1).astype(np.int16).to_numpy()
    )
    combined_features = pd.concat([flow, temporal], axis=1)
    return combined_features.replace([np.inf, -np.inf], np.nan).fillna(0)
