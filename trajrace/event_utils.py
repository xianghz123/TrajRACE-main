from __future__ import annotations

from typing import Any, Dict, List


def length_to_event_value(segments: List[int], mode: str = "num_segments") -> int:
    """
    Convert a segment sequence into a length event value.

    Current simplest version:
    - mode == 'num_segments': use the number of segments directly
    """
    if mode == "num_segments":
        return len(segments)
    raise ValueError(f"Unsupported length mode: {mode}")


def build_event_record_from_sequence(
    seq_record: Dict[str, Any],
    length_mode: str = "num_segments",
) -> Dict[str, Any]:
    """
    Convert one segment-sequence trajectory into one event record.

    Input example:
    {
        "traj_id": "1372636858620000589",
        "start_timestamp": 1372636858,
        "segments": [101, 135, 209, 211, 305]
    }

    Output example:
    {
        "traj_id": "1372636858620000589",
        "start_timestamp": 1372636858,
        "start_event": {"e1": 101},
        "length_event": {"L": 5},
        "transition_events": [
            {"u": 101, "v": 135, "pos": 1, "traj_len": 5},
            {"u": 135, "v": 209, "pos": 2, "traj_len": 5},
            {"u": 209, "v": 211, "pos": 3, "traj_len": 5},
            {"u": 211, "v": 305, "pos": 4, "traj_len": 5}
        ]
    }
    """
    traj_id = seq_record["traj_id"]
    start_timestamp = seq_record.get("start_timestamp", None)
    segments = seq_record["segments"]

    if not isinstance(segments, list) or len(segments) < 2:
        raise ValueError(f"Invalid segment sequence for traj_id={traj_id}")

    traj_len = length_to_event_value(segments, mode=length_mode)

    start_event = {"e1": int(segments[0])}
    length_event = {"L": int(traj_len)}

    transition_events: List[Dict[str, Any]] = []
    for idx in range(len(segments) - 1):
        u = int(segments[idx])
        v = int(segments[idx + 1])
        transition_events.append(
            {
                "u": u,
                "v": v,
                "pos": idx + 1,   # 1-based position
                "traj_len": int(traj_len),
            }
        )

    return {
        "traj_id": traj_id,
        "start_timestamp": start_timestamp,
        "start_event": start_event,
        "length_event": length_event,
        "transition_events": transition_events,
    }


def build_event_records_from_sequences(
    seq_records: List[Dict[str, Any]],
    length_mode: str = "num_segments",
) -> List[Dict[str, Any]]:
    """
    Convert a list of segment-sequence trajectories into event records.
    """
    results: List[Dict[str, Any]] = []
    for rec in seq_records:
        try:
            event_rec = build_event_record_from_sequence(rec, length_mode=length_mode)
            results.append(event_rec)
        except Exception:
            # First simple version: skip bad records silently
            continue
    return results


def summarize_event_records(event_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build simple statistics for sanity checking.
    """
    num_traj = len(event_records)
    if num_traj == 0:
        return {
            "num_trajectories": 0,
            "avg_traj_len": 0.0,
            "avg_num_transition_events": 0.0,
            "num_start_events": 0,
            "num_length_events": 0,
            "num_transition_events": 0,
        }

    traj_lens = [int(rec["length_event"]["L"]) for rec in event_records]
    transition_counts = [len(rec["transition_events"]) for rec in event_records]

    return {
        "num_trajectories": num_traj,
        "avg_traj_len": sum(traj_lens) / num_traj,
        "avg_num_transition_events": sum(transition_counts) / num_traj,
        "num_start_events": num_traj,
        "num_length_events": num_traj,
        "num_transition_events": sum(transition_counts),
        "min_traj_len": min(traj_lens),
        "max_traj_len": max(traj_lens),
    }