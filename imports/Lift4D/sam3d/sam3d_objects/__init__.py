# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# Allow skipping initialization for lightweight tools
if not os.environ.get('LIDRA_SKIP_INIT'):
    try:
        import sam3d_objects.init
    except ModuleNotFoundError as exc:
        if exc.name != 'sam3d_objects.init':
            raise
