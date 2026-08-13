# Re-export robot configs for backward compatibility.
# Import from the robot-specific module directly for new code.
from gear_sonic.envs.manager_env.robots.g1 import *  # noqa: F401,F403
from gear_sonic.envs.manager_env.robots.h2 import *  # noqa: F401,F403
from gear_sonic.envs.manager_env.robots.x2 import *  # noqa: F401,F403
