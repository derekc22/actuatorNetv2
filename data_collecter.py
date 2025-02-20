import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import pandas as pd
import torch
from collections import deque
from actuator_net import ActuatorNet
import sys
import random

#!/usr/bin/env python3
import argparse

def setArgs():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument("-c", "--collect", action="store_true", help="Collect data")
    parser.add_argument("-s", "--shelf", action="store_true", help="Collect shelf data")
    parser.add_argument("-d", "--drop", action="store_true", help="Collect drop data")
    parser.add_argument("-k", "--kickoff", action="store_true", help="Collect kickoff data")
    parser.add_argument("-w", "--wall", action="store_true", help="Collect wall push data")
    parser.add_argument("-p", "--perturb", action="store_true", help="Collect perturbation data")
    parser.add_argument("-t", "--time", type=int, help="Set simulation duration")
    parser.add_argument("-n", "--trials", type=int, help="Set number of trials")
    
    args = parser.parse_args()
    return args


def _setup_joint_indices(m):
    """Map joints to their MuJoCo indices"""
    joint_names = [
        "L_hip_joint", "L_hip2_joint", "L_thigh_joint", "L_calf_joint", "L_toe_joint",
        "R_hip_joint", "R_hip2_joint", "R_thigh_joint", "R_calf_joint", "R_toe_joint"
    ]
    joint_info = {}
    for j_name in joint_names:
        jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j_name)
        act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j_name)
        joint_info[j_name] = {
            'joint_id': jnt_id,
            'act_id': act_id,
            'qpos_adr': m.jnt_qposadr[jnt_id],
            'qvel_adr': m.jnt_dofadr[jnt_id],
        }
    return joint_info

def _init_joint_buffers(joint_info):
    """Initialize history buffers for each joint"""
    joint_buffers = {}
    for j_name in joint_info.keys():
        joint_buffers[j_name] = deque(maxlen=3)
    return joint_buffers


def _track_positions(j_name, pos_err, vel, current_state):
    """Store joint state in tracking dict"""
    current_state[f'{j_name}_pos_err'] = pos_err
    current_state[f'{j_name}_vel'] = vel
    return current_state



def _apply_kickoff_initialization(m, d):
    """Set crouched position and initial upward velocity"""
    # Set slightly crouched pose
    joint_info = _setup_joint_indices(m)
    for j_name in joint_info:
        if 'calf' in j_name:
            d.qpos[joint_info[j_name]['qpos_adr']] = -1.0  # Bent knees
        elif 'thigh' in j_name:
            d.qpos[joint_info[j_name]['qpos_adr']] = 0.7
    
    d.qvel[0:3] = [0, 0, 4]  # Upward velocity
    d.qvel[4] = -2.0  # Angular velocity for flip



def _apply_wall_push(m, d):
    """Simulate wall interaction through initial velocity"""
    # Start near virtual wall position
    # d.qpos[0] = -1.5  # Start 1.5m left of origin
    
    rand_dir = random.randint(0, 3)
    d.qvel[rand_dir] = 3.0  # Push rightward into "wall"


def _apply_perturbation(m, d):
    """Apply random perturbations throughout the simulation."""
    # Random force and torque impulses
    # force = [random.uniform(-80, 80), random.uniform(-80, 80), random.uniform(-80, 80)]
    # torque = [random.uniform(-15, 15), random.uniform(-15, 15), random.uniform(-15, 15)]

    total_mass = sum(m.body_mass)*9.81
    force = [random.uniform(-total_mass, total_mass), random.uniform(-total_mass, total_mass), random.uniform(-total_mass, total_mass)]
    torque = [random.uniform(-total_mass, total_mass), random.uniform(-total_mass, total_mass), random.uniform(-total_mass, total_mass)]
    
    # Get the trunk body ID
    trunk_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    
    # Apply the force and torque (overwrite previous values, don't accumulate)
    d.xfrc_applied[trunk_body_id] = force + torque
    
    return d


def _inverse_kinematics_control(t, m, d, joint_info):
    """Inverse kinematics foot trajectory tracking with proper joint Jacobian extraction."""
    _current_state = {}
    kp = 5.0  # Higher proportional gain for torque authority
    kd = 4.0   # Increased damping to prevent oscillations
    alpha = 1.0  # Aggressive error scaling
    init_amp = 5.0  # Normal working amplitude
    init_freq = 1.0
    damping = 0.1  # DLS stability factor

    for side in ['L', 'R']:
        # Get foot site and current position
        foot_site_name = f"{side}_toe_sensor"
        site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, foot_site_name)
        current_pos = d.site(site_id).xpos.copy()

        # Trajectory generation
        phase_offset = 0.0 if side == 'L' else np.pi
        x = init_amp * np.sin(2 * np.pi * init_freq * t + phase_offset)
        z = -0.85 + init_amp * np.sin(2 * np.pi * init_freq * t + phase_offset)
        desired_pos = np.array([x, current_pos[1], z])

        # Compute position error
        error = desired_pos - current_pos

        # Get leg joints and their velocity addresses
        leg_joints = [
            f"{side}_hip_joint", f"{side}_hip2_joint",
            f"{side}_thigh_joint", f"{side}_calf_joint", f"{side}_toe_joint"
        ]
        vel_adrs = [joint_info[j_name]['qvel_adr'] for j_name in leg_joints]

        # Compute full Jacobian and extract columns for leg joints
        Jt_full = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, Jt_full, None, site_id)
        Jt_leg = Jt_full[:, vel_adrs]  # Extract relevant columns (3x5)

        # Damped Least Squares pseudo-inverse
        J_JT = Jt_leg @ Jt_leg.T
        regularization = damping**2 * np.eye(3)
        J_pinv = Jt_leg.T @ np.linalg.inv(J_JT + regularization)

        # Compute joint angle updates
        delta_theta = alpha * (J_pinv @ error)

        # Apply PD control to each joint
        for i, j_name in enumerate(leg_joints):
            info = joint_info[j_name]
            
            # Get current state
            current_angle = d.qpos[info['qpos_adr']]
            current_vel = d.qvel[info['qvel_adr']]
            
            # Compute desired angle with clamping
            desired_angle = current_angle + delta_theta[i]
            joint_range = m.jnt_range[info['joint_id']]
            if joint_range[0] < joint_range[1]:  # Check valid limits
                desired_angle = np.clip(desired_angle, joint_range[0], joint_range[1])
            
            # PD torque calculation
            pos_err = desired_angle - current_angle
            torque = kp * pos_err + kd * (-current_vel)
            
            # Apply control
            d.ctrl[info['act_id']] = torque
            _current_state = _track_positions(j_name, pos_err, current_vel, _current_state)

    # Add perturbation if enabled
    if args.perturb:
        d = _apply_perturbation(m, d)

    return m, d, _current_state



def create_state(model_path, save_dir):
        m = mujoco.MjModel.from_xml_path(model_path)
        d = mujoco.MjData(m)
        base_save_dir = Path(save_dir)
        base_save_dir.mkdir(exist_ok=True)
        
        # Set up joint info and history buffers
        joint_info = _setup_joint_indices(m)
        # joint_buffers = _init_joint_buffers(joint_info)

        # Current state tracking
        # _current_state = {}

        # Geom IDs
        floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        trunk_geom_ids = [] 
        left_foot_geom_ids = []
        right_foot_geom_ids = []
        for geom_id in range(m.ngeom):
            body_id = m.geom_bodyid[geom_id]
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body_id)
            # if body_name in ["trunk"]:
            if body_name == "trunk": # THIS LINE MAY BE WRONG/BROKEN
                trunk_geom_ids.append(geom_id)
            elif body_name == "L_toe":
                left_foot_geom_ids.append(geom_id)
            elif body_name == "R_toe":
                right_foot_geom_ids.append(geom_id)
        geom_ids = {
            'floor': floor_geom_id,
            'trunk': trunk_geom_ids,
            'left_foot': left_foot_geom_ids,
            'right_foot': right_foot_geom_ids
        }

        return (m, d, base_save_dir, geom_ids, joint_info) #joint_buffers)


def _get_torque_for_joint(j_name, sensor_data):
    """Extract torque value for a joint"""
    parts = j_name.split('_')
    sensor_name = f"{parts[0]}_{parts[1]}_actuatorfrc"
    return sensor_data.get(sensor_name)

def _get_sensor_data(d, _current_state):
    """Collect all relevant sensor data"""
    sensor_data = {}
    joints = ['hip', 'hip2', 'thigh', 'calf', 'toe']
    sides = ['L', 'R']
    for side in sides:
        for joint in joints:
            sensor_name = f'{side}_{joint}_actuatorfrc'
            sensor_data[sensor_name] = d.sensor(sensor_name).data[0]
    sensor_data.update(_current_state)
    return sensor_data


def _euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (radians) to a unit quaternion."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qw, qx, qy, qz]

def _set_initial_orientation(m, d):
    # Find root joint's qpos address
    root_joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")

    if root_joint_id == -1:
        raise ValueError("Root joint not found.")
    root_qpos_adr = m.jnt_qposadr[root_joint_id]

    # Find weld constraint ID
    world_weld_eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "world_root")

    # Disable weld constraint to allow free motion
    if world_weld_eq_id != -1:
        d.eq_active[world_weld_eq_id] = 0
    
    # Set random initial orientation (small perturbations)
    roll = np.random.uniform(-np.pi/12, np.pi/12)
    pitch = np.random.uniform(-np.pi/12, np.pi/12)
    yaw = np.random.uniform(-np.pi/12, np.pi/12)
    quat = _euler_to_quaternion(roll, pitch, yaw)
    
    # Update position and orientation in root joint's qpos
    start_idx = root_qpos_adr
    drop_height = np.random.uniform(0.5, 1)
    d.qpos[start_idx:start_idx+3] = [0, 0, drop_height]  # x, y, z
    d.qpos[start_idx+3:start_idx+7] = quat  # quaternion


def _run_simulation(m, d, control_callback, sample_callback, duration, visualize, drop_biped, start_condition, termination_condition):
    """Generic simulation loop to reduce code duplication."""

    # Randomize initial orientation
    if drop_biped:
        _set_initial_orientation(m, d)

    next_sample_time = 0.0
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(m, d)
    
    start_condition_met = False
    termination_condition_met = False
    sim_time = 0.0

    try:
        while sim_time < duration and not termination_condition_met:
            m, d, _current_state = control_callback(sim_time)
            mujoco.mj_step(m, d)
            sim_time = d.time

            # Check start condition (e.g., foot contact)
            if not start_condition_met:
                start_condition_met = start_condition()
                if start_condition_met:
                    next_sample_time = sim_time
                    print(f"Start condition met at {sim_time:.3f}s")

            # Check termination condition (e.g., trunk contact)
            termination_condition_met = termination_condition()

            # Sample data if conditions are met
            if sim_time >= next_sample_time and start_condition_met:
                sample_callback(d, _current_state)
                next_sample_time += 0.0025

            if viewer:
                viewer.sync()
    finally:
        if viewer:
            viewer.close()
            time.sleep(0.5) # Allow time for viewer to close


def collect_data(m, d, save_dir, duration, visualize, drop_biped, start_on_foot_contact, end_on_trunk_contact, geom_ids, joint_info):
    """Main data collection routine (refactored to use _run_simulation)"""
    joint_data = []
    joint_buffers = _init_joint_buffers(joint_info)

    floor_geom_id = geom_ids['floor']
    trunk_geom_ids = geom_ids['trunk']
    left_foot_geom_ids = geom_ids['left_foot']
    right_foot_geom_ids = geom_ids['right_foot']
    
    # Define conditions
    def start_condition():
        if not start_on_foot_contact:
            return True  # Immediate start
        for i in range(d.ncon):
            contact = d.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            if ((geom1 == floor_geom_id and (geom2 in left_foot_geom_ids or geom2 in right_foot_geom_ids)) or 
                (geom2 == floor_geom_id and (geom1 in left_foot_geom_ids or geom1 in right_foot_geom_ids))):
                return True
        return False

    def termination_condition():
        if end_on_trunk_contact:
            for i in range(d.ncon):
                contact = d.contact[i]
                geom1, geom2 = contact.geom1, contact.geom2
                if ((geom1 == floor_geom_id and geom2 in trunk_geom_ids) or 
                    (geom2 == floor_geom_id and geom1 in trunk_geom_ids)):
                    print(f"Trunk contact detected at {d.time:.3f}s")
                    return True
        return False

    # Define sampling callback
    def sample_callback(d, _current_state):
        sensor_data = _get_sensor_data(d, _current_state)
        for j_name in joint_info.keys():
            pos_err = sensor_data.get(f'{j_name}_pos_err', 0.0)
            vel = sensor_data.get(f'{j_name}_vel', 0.0)
            torque = _get_torque_for_joint(j_name, sensor_data)
            joint_buffers[j_name].append((pos_err, vel, torque))
        
        if all(len(buf) >= 3 for buf in joint_buffers.values()):
            input_features = []
            target_torques = []
            for j_name in joint_info.keys():
                buf = joint_buffers[j_name]
                input_features.extend([buf[2][0], buf[2][1], buf[1][0], buf[1][1], buf[0][0], buf[0][1]])
                target_torques.append(buf[2][2])
            joint_data.append((input_features, target_torques))

    # Run simulation
    _run_simulation(
        m=m,
        d=d,
        control_callback=lambda t: _inverse_kinematics_control(t, m, d, joint_info),
        sample_callback=sample_callback,
        duration=duration,
        visualize=visualize,
        drop_biped=drop_biped,
        start_condition=start_condition,
        termination_condition=termination_condition,
    )

    # Save data (existing code remains unchanged)
    if save_dir:
        save_dir.mkdir(exist_ok=True, parents=True)
        if joint_data:
            inputs = torch.tensor([d[0] for d in joint_data], dtype=torch.float32)
            targets = torch.tensor([d[1] for d in joint_data], dtype=torch.float32)
            file_path = save_dir / "concatenated_data.pt"
            if file_path.exists():
                existing = torch.load(file_path, weights_only=True)
                inputs = torch.cat([existing['inputs'], inputs], dim=0)
                targets = torch.cat([existing['targets'], targets], dim=0)
            torch.save({'inputs': inputs, 'targets': targets}, file_path)


def evaluate_model(model_path, duration, visualize, drop_biped, start_on_foot_contact, end_on_trunk_contact, geom_ids, joint_info, base_save_dir):
    import matplotlib
    matplotlib.use('Agg')  # Set backend to non-interactive
    import matplotlib.pyplot as plt
    import pickle

    floor_geom_id = geom_ids['floor']
    trunk_geom_ids = geom_ids['trunk']
    left_foot_geom_ids = geom_ids['left_foot']
    right_foot_geom_ids = geom_ids['right_foot']

    """Evaluate model using DRY simulation loop"""
    model = ActuatorNet(input_dim=60, output_dim=10)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    torque_data = {j_name: {'predicted': [], 'desired': []} for j_name in joint_info.keys()}
    joint_buffers = _init_joint_buffers(joint_info)

    # Define conditions
    def start_condition():
        if not start_on_foot_contact:
            return True  # Immediate start
        for i in range(d.ncon):
            contact = d.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            if ((geom1 == floor_geom_id and (geom2 in left_foot_geom_ids or geom2 in right_foot_geom_ids)) or 
                (geom2 == floor_geom_id and (geom1 in left_foot_geom_ids or geom1 in right_foot_geom_ids))):
                return True
        return False

    def termination_condition():
        if end_on_trunk_contact:
            for i in range(d.ncon):
                contact = d.contact[i]
                geom1, geom2 = contact.geom1, contact.geom2
                if ((geom1 == floor_geom_id and geom2 in trunk_geom_ids) or 
                    (geom2 == floor_geom_id and geom1 in trunk_geom_ids)):
                    print(f"Trunk contact detected at {d.time:.3f}s")
                    return True
        return False

    # Define sampling callback
    def sample_callback(d, _current_state):
        sensor_data = _get_sensor_data(d, _current_state)
        for j_name in joint_info.keys():
            pos_err = sensor_data.get(f'{j_name}_pos_err', 0.0)
            vel = sensor_data.get(f'{j_name}_vel', 0.0)
            torque = _get_torque_for_joint(j_name, sensor_data)
            joint_buffers[j_name].append((pos_err, vel, torque))
        
        if all(len(buf) >= 3 for buf in joint_buffers.values()):
            input_features = []
            for j_name in joint_info.keys():
                buf = joint_buffers[j_name]
                input_features.extend([
                    buf[2][0], buf[2][1], # Current pos error and velocity
                    buf[1][0], buf[1][1], # Previous pos error and velocity
                    buf[0][0], buf[0][1]  # Pre-previous pos error and velocity
                ])
            input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                predicted = model(input_tensor).squeeze().tolist()
            for idx, j_name in enumerate(joint_info.keys()):
                torque_data[j_name]['predicted'].append(predicted[idx])
                torque_data[j_name]['desired'].append(joint_buffers[j_name][2][2])

    # Run simulation
    _run_simulation(
        m=m,
        d=d,
        control_callback=lambda t: _inverse_kinematics_control(t, m, d, joint_info),
        sample_callback=sample_callback,
        duration=duration,
        visualize=visualize,
        drop_biped=drop_biped,
        start_condition=start_condition,
        termination_condition=termination_condition
    )

    # Plot results
    torque_comparison_dir = base_save_dir / 'torque_comparisons'
    torque_comparison_dir.mkdir(exist_ok=True)
    for j_name in joint_info.keys():
        desired = torque_data[j_name]['desired']
        predicted = torque_data[j_name]['predicted']
        if desired and predicted:

            # Save the figure
            # with open('figure.pickle', 'wb') as f:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(desired, label='Desired')
            ax.plot(predicted, label='Predicted')
            ax.set_title(f'Torque Comparison: {j_name}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Torque (Nm)')
            ax.legend()
            
            # Save the figure to a pickle file
            pickle_file = torque_comparison_dir / f'torque_{j_name}.pickle'
            with open(pickle_file, 'wb') as f:
                pickle.dump(fig, f)
            
            # Optionally, save a static image for viewing later
            fig.savefig(torque_comparison_dir / f'torque_{j_name}.png')
            plt.close(fig)

    return torque_data   



def run_data_collection_suite(
        m, 
        d, 
        trials_per_pattern, 
        duration, 
        visualize, 
        drop_biped, 
        start_on_foot_contact, 
        end_on_trunk_contact, 
        base_save_dir, 
        geom_ids, 
        joint_info,
        mode
    ):
    """Orchestrate data collection for all control patterns."""
    method_dir = base_save_dir / 'inverse_kinematics'

    for trial in range(trials_per_pattern):
        print(f"\nCollecting {mode} - Trial {trial+1}/{trials_per_pattern}")
        mujoco.mj_resetData(m, d)

        # Mode-specific initializations
        if mode == 'kickoff':
            _apply_kickoff_initialization(m, d)
        elif mode == 'wall_push':
            _apply_wall_push(m, d)
        elif mode == 'perturbation':
            pass  # Handled in control callback

        collect_data(
            m=m,
            d=d,
            save_dir=method_dir,
            duration=duration,
            visualize=visualize,
            drop_biped=drop_biped,
            start_on_foot_contact=start_on_foot_contact,
            end_on_trunk_contact=end_on_trunk_contact,
            geom_ids=geom_ids,
            joint_info=joint_info,
        )





if __name__ == "__main__":
    args = setArgs()
    m, d, base_save_dir, geom_ids, joint_info = create_state(model_path='./biped_simple_final.xml', save_dir='./collected_data')

    if args.collect:
        # Original modes
        if args.shelf:
            run_data_collection_suite(
                m=m, d=d, trials_per_pattern=5, duration=args.time,
                visualize=True, drop_biped=False, start_on_foot_contact=False,
                end_on_trunk_contact=False, base_save_dir=base_save_dir,
                geom_ids=geom_ids, joint_info=joint_info, mode='inverse_kinematics'
            )

        if args.drop:
            run_data_collection_suite(
                m=m, d=d, trials_per_pattern=args.trials, duration=args.time,
                visualize=True, drop_biped=True, start_on_foot_contact=True,
                end_on_trunk_contact=True, base_save_dir=base_save_dir,
                geom_ids=geom_ids, joint_info=joint_info, mode='drop'
            )

        # New modes
        if args.kickoff:
            run_data_collection_suite(
                m=m, d=d, trials_per_pattern=args.trials, duration=args.time,
                visualize=True, drop_biped=True, start_on_foot_contact=True,
                end_on_trunk_contact=True, base_save_dir=base_save_dir,
                geom_ids=geom_ids, joint_info=joint_info, mode='kickoff'
            )

        if args.wall:
            run_data_collection_suite(
                m=m, d=d, trials_per_pattern=args.trials, duration=args.time,
                visualize=True, drop_biped=True, start_on_foot_contact=False,
                end_on_trunk_contact=False, base_save_dir=base_save_dir,
                geom_ids=geom_ids, joint_info=joint_info, mode='wall_push'
            )

        if args.perturb:
            run_data_collection_suite(
                m=m, d=d, trials_per_pattern=args.trials, duration=args.time,
                visualize=True, drop_biped=True, start_on_foot_contact=False,
                end_on_trunk_contact=False, base_save_dir=base_save_dir,
                geom_ids=geom_ids, joint_info=joint_info, mode='perturbation'
            )

    else:
        evaluate_model(
            model_path='./trained_models/concatenated/concatenated_model.pth',
            duration=args.time,
            visualize=True,
            drop_biped=True,
            start_on_foot_contact=False,
            end_on_trunk_contact=True,
            base_save_dir=base_save_dir,
            geom_ids=geom_ids,
            joint_info=joint_info,
        )
