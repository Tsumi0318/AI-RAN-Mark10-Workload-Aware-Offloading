from __future__ import annotations

import argparse

from mark10.experiments import STAGES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--pool-id", type=int)
    args = parser.parse_args()
    if args.pool_id is not None:
        if args.stage != "binding":
            parser.error("--pool-id is only valid with --stage binding")
        STAGES[args.stage](task_pool_ids=[args.pool_id])
    else:
        STAGES[args.stage]()


if __name__ == "__main__":
    main()
