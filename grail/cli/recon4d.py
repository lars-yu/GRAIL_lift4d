"""CLI entry point for 4D HOI reconstruction.

Usage:
    python -m grail.cli.recon4d --dataset ComAsset --category cordless_drill --results_dir results
"""

from grail.pipelines.recon_4dhoi import main

if __name__ == "__main__":
    main()
