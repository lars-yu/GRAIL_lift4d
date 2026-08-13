"""CLI entry point for 2D HOI generation.

Usage:
    python -m grail.cli.gen2d --dataset ComAsset --category cordless_drill --results_dir results
"""

from grail.pipelines.gen_2dhoi import main

if __name__ == "__main__":
    main()
