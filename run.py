#!/usr/bin/env python3
"""Run the compact Highway vehicle-trace pipeline."""

import argparse
from pathlib import Path

from highway_pipeline import HighwayProject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "trace", "all"))
    parser.add_argument("--trace", default="1", choices=("1", "2"), help="configured channel range")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "highway.json",
    )
    args = parser.parse_args()

    project = HighwayProject(args.config)
    if args.command == "prepare":
        print(f"prepared {project.prepare(args.trace)} H5 files")
    elif args.command == "trace":
        print("\n".join(project.trace(args.trace)))
    else:
        count, summaries = project.run(args.trace)
        print(f"prepared {count} H5 files")
        print("\n".join(summaries))


if __name__ == "__main__":
    main()
