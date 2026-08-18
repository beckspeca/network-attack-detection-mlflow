"""Past-only rolling feature implementation adapted from the attached notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd


TIMESTAMP = "Timestamp"
ROLLING_SOURCE_COLUMNS = [
    TIMESTAMP,
    "Dst Port",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "SYN Flag Cnt",
]


def add_past_window_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create attachment-compatible rolling features without current/future rows."""
    values = frame.loc[:, ROLLING_SOURCE_COLUMNS].copy()
    values[TIMESTAMP] = pd.to_datetime(values[TIMESTAMP])
    window = values.set_index(TIMESTAMP).sort_index(kind="mergesort")
    features = pd.DataFrame(index=values.index)
    features["time_since_prev_flow"] = window.index.to_series().diff().dt.total_seconds().to_numpy()
    features["flow_count_10s"] = window["Tot Fwd Pkts"].rolling("10s", closed="left").count().to_numpy()
    features["flow_count_60s"] = window["Tot Fwd Pkts"].rolling("60s", closed="left").count().to_numpy()
    total_packets = window["Tot Fwd Pkts"] + window["Tot Bwd Pkts"]
    total_bytes = window["TotLen Fwd Pkts"] + window["TotLen Bwd Pkts"]
    features["packet_count_10s"] = total_packets.rolling("10s", closed="left").sum().to_numpy()
    features["byte_count_10s"] = total_bytes.rolling("10s", closed="left").sum().to_numpy()
    features["syn_count_10s"] = window["SYN Flag Cnt"].rolling("10s", closed="left").sum().to_numpy()
    features["packet_rate_10s"] = features["packet_count_10s"] / 10.0
    features["byte_rate_10s"] = features["byte_count_10s"] / 10.0
    features["dst_flow_count_10s"] = window.groupby("Dst Port")["Tot Fwd Pkts"].transform(
        lambda series: series.rolling("10s", closed="left").count()
    ).to_numpy()
    fwd60 = window["Tot Fwd Pkts"].rolling("60s", closed="left").sum()
    bwd60 = window["Tot Bwd Pkts"].rolling("60s", closed="left").sum()
    fwd10 = window["Tot Fwd Pkts"].rolling("10s", closed="left").sum()
    bwd10 = window["Tot Bwd Pkts"].rolling("10s", closed="left").sum()
    packet60 = total_packets.rolling("60s", closed="left").sum()
    features["fwd_bwd_ratio_60s"] = fwd60.to_numpy() / (bwd60.to_numpy() + 1)
    features["fwd_bwd_ratio_10s"] = fwd10.to_numpy() / (bwd10.to_numpy() + 1)
    features["burst_ratio_10s"] = features["packet_count_10s"] / (packet60.to_numpy() + 1)
    categorical = pd.Categorical(window["Dst Port"])
    port_codes = categorical.codes
    onehot = pd.DataFrame(
        {index: (port_codes == index).astype(np.int8) for index in range(len(categorical.categories))},
        index=window.index,
    )
    present10 = onehot.rolling("10s", closed="left").sum() > 0
    present60 = onehot.rolling("60s", closed="left").sum() > 0
    features["dstport_distinct_10s"] = present10.sum(axis=1).to_numpy()
    features["dstport_distinct_60s"] = present60.sum(axis=1).to_numpy()
    features["port_diversity_10s"] = features["dstport_distinct_10s"] / (features["flow_count_10s"] + 1)
    unanswered = (window["Tot Bwd Pkts"] == 0).astype(np.int8)
    window_flows = window["Tot Fwd Pkts"].rolling("10s", closed="left").count()
    unanswered_count = unanswered.rolling("10s", closed="left").sum()
    features["syn_unanswered_rate_10s"] = unanswered_count.to_numpy() / (window_flows.to_numpy() + 1)
    features["syn_per_bwd_10s"] = features["syn_count_10s"] / (bwd10.to_numpy() + 1)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_attachment_feature_frame(
    full_frame: pd.DataFrame,
    selected_flow_features: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Combine selected Flow columns and past-only rolling features."""
    temporal = add_past_window_features(full_frame)
    port_categories = sorted(full_frame["Dst Port"].astype("string").unique().tolist())
    port_map = {value: index for index, value in enumerate(port_categories)}
    flow = full_frame.loc[:, selected_flow_features].copy()
    flow["Dst Port"] = full_frame["Dst Port"].astype("string").map(port_map).fillna(-1).astype(np.int16)
    combined = pd.concat([flow, temporal], axis=1)
    return combined.replace([np.inf, -np.inf], np.nan).fillna(0), port_map
