#!/usr/bin/env python3
"""
Object Initial State Simulation Script

This script loads a mesh object from a dataset/category, runs a physics simulation
to determine a stable resting orientation under gravity, and saves the final
orientation to a pickle file for use by render_blender_scene.py.

The simulation drops the object from a height and lets it settle on the ground,
then captures the final stable orientation.

Usage:
    python sim_object_init_state.py --dataset ComAsset --category barbell --output_dir results/initial_states

Requirements:
    - warp physics simulation library
    - trimesh for mesh loading
"""

import argparse
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

wp.init()

import warp.sim
import warp.sim.render

from grail.core.dataset import category2object


class ObjectInitialStateSimulator:
    def __init__(
        self,
        object_path,
        drop_height=2.0,
        settling_time=5.0,
        stage_path=None,
        initial_rotation_perturbation=5.0,
    ):
        """
        Initialize the physics simulation for determining object's stable orientation

        Args:
            object_path (str): Path to the object mesh file
            drop_height (float): Height to drop object from (meters)
            settling_time (float): Time to let object settle (seconds)
            stage_path (str): Path to save USD file for visualization (optional)
            initial_rotation_perturbation (float): Random rotation perturbation in degrees
        """
        self.object_path = object_path
        self.drop_height = drop_height
        self.settling_time = settling_time
        self.stage_path = stage_path
        self.initial_rotation_perturbation = initial_rotation_perturbation

        # Simulation parameters
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Physics parameters
        self.ke = 1.0e5  # Contact stiffness
        self.kd = 100.0  # Contact damping
        self.kf = 10.0  # Friction stiffness
        self.density = 1000.0  # Object density (kg/m³)

        # Tracking
        self.body_positions = []
        self.body_orientations = []
        self.simulation_step = 0

        # Initialize simulation
        self._setup_simulation()

    def _setup_simulation(self):
        """Setup the physics simulation with the object and ground"""
        print(f"Setting up simulation for object: {self.object_path}")

        builder = wp.sim.ModelBuilder()

        # Load object mesh using warp's mesh loader
        try:
            mesh_points, mesh_indices = wp.sim.load_mesh(self.object_path, method="trimesh")
            self.object_mesh = wp.sim.Mesh(mesh_points, mesh_indices)
            print(
                f"Successfully loaded mesh with {len(mesh_points)} vertices and {len(mesh_indices)//3} triangles"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh from {self.object_path}: {str(e)}")

        # Calculate object bounding box to determine lowest point
        mesh_points_np = (
            mesh_points.numpy() if hasattr(mesh_points, "numpy") else np.array(mesh_points)
        )
        self.min_bounds = np.min(mesh_points_np, axis=0)
        self.max_bounds = np.max(mesh_points_np, axis=0)
        self.object_height = (
            self.max_bounds[1] - self.min_bounds[1]
        )  # Y is up in most coordinate systems
        self.lowest_point_offset = self.min_bounds[1]  # Offset of lowest point from object center

        # Adjust drop position so lowest point of object starts at drop_height above ground
        # If lowest_point_offset is negative, we need to add its absolute value to the drop height
        self.adjusted_drop_height = self.drop_height - self.lowest_point_offset

        # Generate random initial rotation perturbation
        perturbation_rad = math.radians(self.initial_rotation_perturbation)
        if self.initial_rotation_perturbation > 0:
            random_x = (
                (np.random.random() - 0.5) * 2 * perturbation_rad
            )  # Random angle in [-perturbation, +perturbation]
            random_y = (np.random.random() - 0.5) * 2 * perturbation_rad
            random_z = (np.random.random() - 0.5) * 2 * perturbation_rad
        else:
            # No perturbation
            random_x = random_y = random_z = 0.0

        print(
            f"Initial rotation perturbation (degrees): ({math.degrees(random_x):.1f}, {math.degrees(random_y):.1f}, {math.degrees(random_z):.1f})"
        )

        # Create combined rotation from individual axis rotations
        quat_x = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), random_x)
        quat_y = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), random_y)
        quat_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), random_z)

        # Combine rotations: Z * Y * X (reverse order for composition)
        combined_quat = wp.mul(wp.mul(quat_z, quat_y), quat_x)

        # Add the object body at adjusted drop height with perturbed rotation
        self.object_body = builder.add_body(
            origin=wp.transform(
                (0.0, self.adjusted_drop_height, 0.0),  # Start position (x, y, z)
                combined_quat,  # Perturbed initial rotation
            )
        )

        # Add the mesh shape to the body
        builder.add_shape_mesh(
            body=self.object_body,
            mesh=self.object_mesh,
            pos=wp.vec3(0.0, 0.0, 0.0),  # Relative to body origin
            scale=wp.vec3(1.0, 1.0, 1.0),  # No scaling
            ke=self.ke,
            kd=self.kd,
            kf=self.kf,
            density=self.density,
        )

        # Finalize the model
        self.model = builder.finalize()
        self.model.ground = True  # Enable ground collision

        # Setup integrator (XPBD is good for rigid body dynamics)
        self.integrator = wp.sim.XPBDIntegrator()

        # Setup USD renderer if stage path provided
        if self.stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, self.stage_path, scaling=0.5)
        else:
            self.renderer = None

        # Initialize states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Evaluate forward kinematics
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        # Setup CUDA graph if available
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self._simulate_step()
            self.graph = capture.graph

        print("Simulation setup complete")

    def _simulate_step(self):
        """Perform one simulation step"""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def get_object_transform(self):
        """Get current position and orientation of the object"""
        # Get body transforms from the current state
        body_q = self.state_0.body_q.numpy()  # Body positions and orientations

        # Extract position (x, y, z) and quaternion (x, y, z, w)
        position = body_q[0][:3]  # First (and only) body
        quaternion = body_q[0][3:]  # Quaternion

        return position.copy(), quaternion.copy()

    def store_current_transform(self):
        """Store current transform of the object"""
        position, quaternion = self.get_object_transform()
        self.body_positions.append(position)
        self.body_orientations.append(quaternion)

    def render(self):
        """Render the current frame to USD file"""
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=True):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

    def is_settled(self, position_threshold=0.001, orientation_threshold=0.001, check_frames=30):
        """
        Check if object has settled (stopped moving significantly)

        Args:
            position_threshold (float): Maximum position change to consider settled
            orientation_threshold (float): Maximum orientation change to consider settled
            check_frames (int): Number of recent frames to check for stability

        Returns:
            bool: True if object has settled
        """
        if len(self.body_positions) < check_frames:
            return False

        # Get recent positions and orientations
        recent_positions = np.array(self.body_positions[-check_frames:])
        recent_orientations = np.array(self.body_orientations[-check_frames:])

        # Check position stability
        pos_range = np.max(recent_positions, axis=0) - np.min(recent_positions, axis=0)
        pos_stable = np.all(pos_range < position_threshold)

        # Check orientation stability (using quaternion difference)
        orient_diffs = []
        for i in range(1, len(recent_orientations)):
            # Calculate quaternion difference magnitude
            q1 = recent_orientations[i - 1]
            q2 = recent_orientations[i]
            # Dot product gives cosine of half the angle between quaternions
            dot_product = np.abs(np.dot(q1, q2))
            dot_product = min(1.0, dot_product)  # Clamp to handle numerical errors
            angle_diff = 2.0 * np.arccos(dot_product)
            orient_diffs.append(angle_diff)

        orient_stable = len(orient_diffs) == 0 or np.max(orient_diffs) < orientation_threshold

        return pos_stable and orient_stable

    def simulate_until_settled(self, max_simulation_time=None):
        """
        Run simulation until object settles or max time is reached

        Args:
            max_simulation_time (float): Maximum simulation time in seconds

        Returns:
            dict: Final transform data
        """
        if max_simulation_time is None:
            max_simulation_time = self.settling_time

        max_steps = int(max_simulation_time / self.frame_dt)

        start_time = time.time()

        # Store initial state and render initial frame
        self.store_current_transform()
        self.render()

        # Run simulation
        for step in range(max_steps):
            # Perform simulation step
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self._simulate_step()

            # Update tracking
            self.store_current_transform()
            self.simulation_step += 1
            self.sim_time += self.frame_dt

            # Render frame
            self.render()

            # Check if settled every 10 steps
            if step % 10 == 0:
                if self.is_settled():
                    print(f"Object settled after {step} steps ({self.sim_time:.2f}s)")
                    break

                # Print progress
                position, quaternion = self.get_object_transform()
                print(
                    f"Step {step}: pos=({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), "
                    f"height={position[1]:.3f}m"
                )
        else:
            print(f"Simulation completed after {max_steps} steps ({max_simulation_time}s)")

        # Save USD file if renderer was used
        if self.renderer:
            self.renderer.save()
            print(f"USD file saved to: {self.stage_path}")

        # Get final transform
        final_position, final_quaternion = self.get_object_transform()

        print(
            f"Final position: ({final_position[0]:.3f}, {final_position[1]:.3f}, {final_position[2]:.3f})"
        )
        print(
            f"Final quaternion: ({final_quaternion[0]:.3f}, {final_quaternion[1]:.3f}, "
            f"{final_quaternion[2]:.3f}, {final_quaternion[3]:.3f})"
        )

        # Convert quaternion to euler angles for reference
        final_euler = self._quaternion_to_euler(final_quaternion)
        print(
            f"Final euler angles (deg): ({np.degrees(final_euler[0]):.1f}, "
            f"{np.degrees(final_euler[1]):.1f}, {np.degrees(final_euler[2]):.1f})"
        )

        # final_R = wp.quat_to_matrix(final_quaternion).numpy()

        return {
            "obj_R_quat": final_quaternion,
            "obj_R_euler": final_euler,
            # "final_position": final_position,
            # "final_quaternion": final_quaternion,
            # "final_euler_radians": final_euler,
            # "final_euler_degrees": np.degrees(final_euler),
            # "simulation_time": simulation_time,
            # "num_steps": self.simulation_step,
            # "object_path": self.object_path,
            # "drop_height": self.drop_height,
            # "drop_height_adjusted": self.adjusted_drop_height,
            # "object_bounds": {"min": self.min_bounds, "max": self.max_bounds},
            # "object_height": self.object_height,
            # "lowest_point_offset": self.lowest_point_offset,
            # "settling_time": max_simulation_time,
            # "all_positions": np.array(self.body_positions),
            # "all_orientations": np.array(self.body_orientations),
        }

    def _quaternion_to_euler(self, q):
        """
        Convert quaternion to euler angles (roll, pitch, yaw)

        Args:
            q (array): Quaternion [x, y, z, w]

        Returns:
            array: Euler angles [roll, pitch, yaw] in radians
        """
        x, y, z, w = q

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])


def simulate_object_initial_state(
    dataset,
    category,
    output_dir,
    drop_height=2.0,
    settling_time=5.0,
    skip_done=False,
    save_usd=True,
    initial_rotation_perturbation=5.0,
):
    """
    Simulate object to find stable initial orientation

    Args:
        dataset (str): Dataset name
        category (str): Object category
        output_dir (str): Directory to save results
        drop_height (float): Height to drop object from
        settling_time (float): Maximum time to simulate
        skip_done (bool): Skip if output file already exists
        save_usd (bool): Whether to save USD file for visualization
        initial_rotation_perturbation (float): Random rotation perturbation in degrees

    Returns:
        str: Path to saved pickle file
    """
    # Get object path
    try:
        object_path = category2object(f"data/{dataset}", category)
        print(f"Found object: {object_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to find object for dataset='{dataset}', category='{category}': {str(e)}"
        )

    # Create output directory
    os.makedirs(f"{output_dir}/{dataset}/{category}", exist_ok=True)
    output_file = f"{output_dir}/{dataset}/{category}/initial_state.pickle"

    # Setup USD file path if requested
    stage_path = None
    if save_usd:
        stage_path = f"{output_dir}/{dataset}/{category}/initial_state_simulation.usd"

    # Check if already done
    if skip_done and os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        return output_file

    # Run simulation
    simulator = ObjectInitialStateSimulator(
        object_path=object_path,
        drop_height=drop_height,
        settling_time=settling_time,
        stage_path=stage_path,
        initial_rotation_perturbation=initial_rotation_perturbation,
    )

    result_data = simulator.simulate_until_settled()

    # Save results
    with open(output_file, "wb") as f:
        pickle.dump(result_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Initial state data saved to: {output_file}")
    return output_file


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Simulate object to determine stable initial orientation"
    )

    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., ComAsset, Terrain). Meshes are looked up under data/<dataset>/<category>/.",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Object category (e.g., barbell, chair)",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/generation/initial_states",
        help="Directory to save initial state data",
    )

    # Simulation parameters
    parser.add_argument(
        "--drop_height",
        type=float,
        default=0.1,
        help="Height to drop object from (meters)",
    )
    parser.add_argument(
        "--settling_time",
        type=float,
        default=5.0,
        help="Maximum time to simulate for settling (seconds)",
    )
    parser.add_argument(
        "--initial_rotation_perturbation",
        type=float,
        default=5.0,
        help="Random rotation perturbation in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results",
    )

    # Options
    parser.add_argument(
        "--skip_done", action="store_true", help="Skip if output file already exists"
    )
    parser.add_argument(
        "--save_usd",
        action="store_true",
        default=False,
        help="Save USD file for visualization (default: True)",
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Determine USD saving preference
    save_usd = args.save_usd

    try:
        # Set random seed if provided
        if args.seed is not None:
            np.random.seed(args.seed)
            print(f"Using random seed: {args.seed}")

        with wp.ScopedDevice(args.device):
            simulate_object_initial_state(
                dataset=args.dataset,
                category=args.category,
                output_dir=args.output_dir,
                drop_height=args.drop_height,
                settling_time=args.settling_time,
                skip_done=args.skip_done,
                save_usd=save_usd,
                initial_rotation_perturbation=args.initial_rotation_perturbation,
            )

    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
