#!/usr/bin/env python3
"""Screen a local raw tracking recording; write a review report, never activate it.

Example: python fit_operator_profile.py frames.jsonl.gz labels.json --output report.json
labels.json: {"user":"...", "source":"retarget_lab", "operator_confirmed_straight_arms":true,
 "holds":[{"name":"forward","start_s":1,"end_s":2,"use":"fit"}, ...,
          {"name":"independent","start_s":12,"end_s":13,"use":"validate"}]}
At least four spatially diverse fit holds and one unused validation hold are required.
Narrow arcs can remain too uncertain regardless of sample count and are rejected.

For the report-only three-pose alternative, labels also contain
 "method":"neutral_forward", "operator_confirmed_poses":true,
 "neutral":"neutral", "forward":"forward";
holds are neutral/forward with use:"fit" and a separate diagonal with use:"validate".
This mode never exports a profile, even when its pose-consistency checks pass.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mabel.whole_body_retargeting.profile_capture import assess_operator_capture, assess_neutral_forward_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="review report only, never a live profile path")
    args = parser.parse_args()
    labels = json.loads(args.labels.read_text())
    opener = gzip.open if args.trace.suffix == ".gz" else open
    with opener(args.trace, "rt") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if labels.get("method") == "neutral_forward":
        report = assess_neutral_forward_capture(
            rows, labels["holds"], source=labels.get("source"),
            neutral=labels.get("neutral", "neutral"), forward=labels.get("forward", "forward"),
            operator_confirmed_poses=labels.get("operator_confirmed_poses", False))
    else:
        report = assess_operator_capture(rows, labels["holds"], user=labels.get("user", "operator"),
                                         source=labels.get("source"),
                                         operator_confirmed_straight_arms=labels.get("operator_confirmed_straight_arms", False))
    report["trace_sha256"] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    if report["candidate_profile"]:
        report["candidate_profile"]["quality"]["trace_sha256"] = report["trace_sha256"]
    report["trace_file"] = str(args.trace.resolve())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"] + ": " + str(args.output))
    for error in report["errors"]:
        print("  " + error)
    return 0 if report["status"] != "rejected" else 2


if __name__ == "__main__":
    sys.exit(main())
