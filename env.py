import numpy as np
from gymnasium import Env, spaces
import matplotlib.pyplot as plt


class MultiAgentDroneEnv(Env):
    def __init__(self, agent_positions, target_positions, grid_size, radius_field_of_view, target_threshold, max_steps, likelihood_decay):
        self.agent_positions = agent_positions
        self.target_positions = target_positions
        self.grid_size = grid_size
        self.radius_field_of_view = radius_field_of_view
        self.target_threshold = target_threshold
        self.max_steps = max_steps
        self.likelihood_decay = likelihood_decay

        self.num_agents = len(agent_positions)
        self.num_targets = len(target_positions)
        self.detected_targets = set()

        # Initialize ROI mask and field of view offsets
        self.roi_mask = np.zeros(grid_size, dtype=bool)
        self.roi_radii = []  # Store the ROI radius of each target
        
        self.field_of_view_offsets = [
            (dx, dy)
            for dx in range(-radius_field_of_view, radius_field_of_view + 1)
            for dy in range(-radius_field_of_view, radius_field_of_view + 1)
            if np.sqrt(dx**2 + dy**2) <= radius_field_of_view
        ]

        self.action_space = spaces.MultiDiscrete([7] * self.num_agents)
        self.observation_space = spaces.Dict({
            "positions": spaces.Box(
                low=np.zeros((self.num_agents, 2), dtype=np.int32),
                high=np.tile(np.array(grid_size) - 1, (self.num_agents, 1)),
                shape=(self.num_agents, 2),
                dtype=np.int32
            ),
            "directions": spaces.Box(low=0, high=360, shape=(self.num_agents,), dtype=np.float32),
            "map": spaces.Box(low=0.0, high=1.0, shape=grid_size + (3,), dtype=np.float32)
        })

        self.state = None
        self.steps = 0
        self.log_likelihood_ratios = np.zeros(self.grid_size)
        self.mutual_information = 0  # Initialize mutual information
        self.np_random = None  # Random number generator for seed-based consistency

    def _get_observation(self):
        positions = self.state["positions"].flatten() / np.array(self.grid_size).max()
        directions = self.state["directions"] / 360.0
        map_flat = self.state["map"].flatten()
        return np.concatenate([positions, directions, map_flat])

    def _initialize_roi_mask(self):
        """Initialize the ROI mask with random radii."""
        self.roi_radii.clear()
        for target in self.target_positions:
            cx, cy = target
            radius = self.np_random.uniform(2 * self.radius_field_of_view, 4 * self.radius_field_of_view)
            self.roi_radii.append(radius)
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    if np.sqrt((x - cx)**2 + (y - cy)**2) <= radius:
                        self.roi_mask[x, y] = True

    def reset(self, seed=None):
        """Reset the environment and initialize the RNG."""
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        self.steps = 0
        self.detected_targets = set()

        # Reinitialize ROI mask to ensure deterministic results
        self.roi_mask.fill(False)
        self._initialize_roi_mask()

        initial_positions = np.array([(x, y) for x, y, _ in self.agent_positions], dtype=np.int32)
        initial_directions = np.array([d for _, _, d in self.agent_positions], dtype=np.float32)

        probabilistic_map = np.ones(self.grid_size + (3,)) / 3

        self.state = {
            "positions": initial_positions,
            "directions": initial_directions,
            "map": probabilistic_map
        }
        self.log_likelihood_ratios = np.zeros(self.grid_size)
        self.mutual_information = 0
        return self._get_observation(), {}

    def render(self, mode, agent_paths=None):
        """Render the environment for debugging or visualization."""
        if mode == 'human':
            grid = np.full(self.grid_size, '.', dtype=str)
            for x, y in self.target_positions:
                grid[x, y] = 'T'
            for i, (x, y) in enumerate(self.state["positions"]):
                grid[x, y] = f'A{i}'
            print("\n".join(" ".join(row) for row in grid))
            print(f"Mutual Information: {self.mutual_information:.2f}")
            print(f"Steps: {self.steps}, Detected Targets: {len(self.detected_targets)} / {self.num_targets}")
        
        elif mode == 'rgb_array':
            fig, ax = plt.subplots(figsize=(8, 8))
            map_data = np.flipud(self.state["map"].max(axis=-1))  # Visualizing the highest probability across layers
            ax.imshow(map_data, cmap='gray', interpolation='none', extent=[0, self.grid_size[1], 0, self.grid_size[0]])

            # Plot targets and their ROIs
            for (x, y), radius in zip(self.target_positions, self.roi_radii):
                circle = plt.Circle((y, x), radius, color='red', alpha=0.3, label='ROI')
                ax.add_artist(circle)
                ax.plot(y, x, 'rx', markersize=12, label='Target')

            # Plot agent positions and paths
            for i, (x, y) in enumerate(self.state["positions"]):
                ax.plot(y, x, 'bo', markersize=8, label=f'Agent {i}')
                if agent_paths and i in agent_paths:
                    path = np.array(agent_paths[i])
                    ax.plot(path[:, 1], path[:, 0], 'b--', linewidth=1, alpha=0.7)

            plt.title("Multi-Agent Drone Environment")
            ax.set_xticks(range(0, self.grid_size[1], max(1, self.grid_size[1] // 10)))
            ax.set_yticks(range(0, self.grid_size[0], max(1, self.grid_size[0] // 10)))
            ax.set_xlim(0, self.grid_size[1])
            ax.set_ylim(0, self.grid_size[0])
            plt.legend(loc='lower right')
            plt.grid(True, which='major', linestyle='--', linewidth=0.5)
            plt.show(block=False)
            plt.pause(0.1)
        
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

    def step(self, actions):
        """Take a step in the environment."""
        self.steps += 1
        
        previous_positions = self.state["positions"].copy()
        penalty = self._update_positions_and_directions(actions)
        
        # Log movements
        movements = [
            f"Agent {i} hovered at {prev_pos}." if action == 6 else f"Agent {i} moved from {prev_pos} to {curr_pos}."
            for i, (prev_pos, action, curr_pos) in enumerate(zip(previous_positions, actions, self.state["positions"]))
        ]
        print(f"Timestep {self.steps}:\n" + "\n".join(movements))
        
        self._update_information()

        # Detect targets
        global_likelihood = np.sum(self.log_likelihood_ratios)
        if global_likelihood > self.target_threshold * self.grid_size[0] * self.grid_size[1]:
            print("Global target detection triggered.")
            self.detected_targets.add("global")

        
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.log_likelihood_ratios[x, y] > self.target_threshold:
                    print(f"Target detected at cell ({x}, {y}) with log likelihood ratio {self.log_likelihood_ratios[x, y]:.4f}")
             
            
            '''
            if idx not in self.detected_targets and self.log_likelihood_ratios[tx, ty] > self.target_threshold:
                print(f"Target detected at cell: {target}")
                self.detected_targets.add(idx)
            '''
        reward = self._compute_reward() + penalty
        done = self.steps >= self.max_steps or len(self.detected_targets) == self.num_targets
        if done:
            print(f"Termination Condition Reached: {'Max steps' if self.steps >= self.max_steps else 'All targets detected'}")
        
        # Early termination for stagnation
        if self.mutual_information < 0.01 and self.steps > 10:
            print("Exploration stagnated, ending episode.")
            done = True
        
        return self._get_observation(), reward, done, {}

    def _update_positions_and_directions(self, actions):
        """Update positions and directions based on actions."""
        angle_changes = [0, 60, 120, 180, -60, -120]
        new_positions = np.zeros_like(self.state["positions"])
        new_directions = np.zeros_like(self.state["directions"])
        penalty = 0
        
        # Compute current fields of view from the field_of_view_offests
        current_fov = [set() for _ in range(self.num_agents)]
        for i, (x, y) in enumerate(self.state["positions"]):
            for dx, dy in self.field_of_view_offsets:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]:
                    current_fov[i].add((new_x, new_y))

        for i, action in enumerate(actions):
            if action == 6:  # Hover
                new_positions[i] = self.state["positions"][i]
                new_directions[i] = self.state["directions"][i]
                continue

            angle_change = angle_changes[action]
            new_direction = (self.state["directions"][i] + angle_change) % 360
            direction_radians = np.deg2rad(new_direction)

            R = self.radius_field_of_view
            epsilon = self.np_random.uniform(0, 0.1 * R)
            movement_distance = 2 * R - epsilon
            
            start_position = self.state["positions"][i]
            end_position = start_position + np.array([
                movement_distance * np.cos(direction_radians),
                movement_distance * np.sin(direction_radians)
                ])
            
            # Check whether agent is trying to move out of bounds
            if not (0 <= end_position[0] < self.grid_size[0] and 0 <= end_position[1] < self.grid_size[1]):
                print(f"Agent {i} attempted to move out of bounds, so stayed in place.")
                new_positions[i] = self.state["positions"][i]
                new_directions[i] = self.state["directions"][i]
                penalty -= 5  # Penalty for out-of-bounds attempt
                continue

            # Simulate the motion path
            path_points = []
            for t in np.linspace(0, 1, 10):
                point = start_position + t * (end_position - start_position)
                path_points.append(point)
            
            # Compute the FOV for the motion path
            path_fov = set()
            for point in path_points:
                for dx, dy in self.field_of_view_offsets:
                    path_x, path_y = int(point[0]) + dx, int(point[1]) + dy
                    if 0 <= path_x < self.grid_size[0] and 0 <= path_y < self.grid_size[1]:
                        path_fov.add((path_x, path_y))
                        
            # Check for overlap with other agents' FOVs (during motion)
            overlap = any(path_fov & current_fov[j] for j in range(self.num_agents) if j != i)
            
            # Check if the agent is trying to move out of bounds or if its FOV will overlap during motion
            if np.any(start_position != end_position) and overlap:
                print(f"Agent {i} would have overlapped FOV/collided with another agent during motion, so stayed in place.")
                new_positions[i] = self.state["positions"][i]
                new_directions[i] = self.state["directions"][i]
                penalty -= 10  # Penalty for motion-based FOV overlap
            else:
                new_positions[i] = np.round(end_position).astype(int)
                new_directions[i] = new_direction

        self.state["positions"] = new_positions
        self.state["directions"] = new_directions

        return penalty

    def _update_information(self):
        """Update the probabilistic map, likelihood ratios, and compute mutual information."""
        self.mutual_information = 0
        self.log_likelihood_ratios *= self.likelihood_decay

        for position in self.state["positions"]:
            x, y = position
            field_of_view_cells = []
            for dx, dy in self.field_of_view_offsets:
                new_x, new_y = x + dx, y + dy
                
                if not (0 <= new_x < self.grid_size[0] and 0 <= new_y < self.grid_size[1]):
                    continue
                
                field_of_view_cells.append((new_x, new_y))
                
                v, u, o = self.state["map"][new_x, new_y]
                
                m_g = 4
                m_z = 10  # High value of m_z needed for accurate target detection
                
                w_g = self.np_random.normal(0, 1)  # white noise for G sensor
                w_z = self.np_random.normal(0, 1)  # white noise for Z sensor
                
                # Check ROI and target presence
                is_roi = self.roi_mask[new_x, new_y]
                is_target = (new_x, new_y) in self.target_positions
                
                g_k = (m_g if is_roi else 0) + w_g  # Signal strength for ROI
                z_k = (m_z if is_target else 0) + w_z  # Signal strength for target
            

                g_given_V = np.exp(-g_k**2 / 2)
                g_given_UO = np.exp(-(g_k - m_g)**2 / 2)
                z_given_VU = np.exp(-z_k**2 / 2)
                z_given_O = np.exp(-(z_k - m_z)**2 / 2)

                k_V = z_given_VU * g_given_V * v
                k_U = z_given_VU * g_given_UO * u
                k_O = z_given_O * g_given_UO * o
                total_k = k_V + k_U + k_O + 1e-8 # Small value to avoid division by zero

                p_V = k_V / total_k
                p_U = k_U / total_k
                p_O = k_O / total_k
                
                # Normalizing the probabilities (probably unnecessary but still added)
                p_sum = p_V + p_U + p_O
                p_V /= p_sum
                p_U /= p_sum
                p_O /= p_sum


                self.state["map"][new_x, new_y, 0] = p_V
                self.state["map"][new_x, new_y, 1] = p_U
                self.state["map"][new_x, new_y, 2] = p_O

                # Measurement Update
                #log_measurement_update = np.log(z_given_O) - np.log(z_given_VU)
                log_measurement_update = m_z * (z_k - m_z / 2)  # Measurement likelihood update
                
                '''
                # Motion Update (for stationary targets, this is irrelevant, as forecast posterior equals the prior)
                log_motion_update = 0  # Default motion update for stationary targets
                '''
                    
                # Combine updates into the log likelihood ratio
                log_likelihood_ratio_update = log_measurement_update # + log_motion_update
                self.log_likelihood_ratios[new_x, new_y] += log_likelihood_ratio_update

                
                if self.log_likelihood_ratios[new_x, new_y] > self.target_threshold:
                    print(f"Target detected at cell: ({new_x, new_y}) with log likelihood ratio {self.log_likelihood_ratios[new_x, new_y]}")
                
                #print(f"Log Likelihood ratios for targets:")
                #for idx, target in enumerate(self.target_positions):
                    #tx, ty = target
                    #print(f"Target {idx} at ({tx}, {ty}): {self.log_likelihood_ratios[tx, ty]:.4f}")


                H_C = -np.sum([p * np.log2(p + 1e-8) for p in [p_V, p_U, p_O]])
                H_C_given_GZ = -np.sum([k * np.log2(k / total_k + 1e-8) for k in [k_V, k_U, k_O]])
                self.mutual_information += H_C - H_C_given_GZ
                
            #print(f"Agent field of view: {field_of_view_cells}")
        

    def _compute_reward(self):
        """Compute reward based on mutual information and target detection."""
        normalized_mi = self.mutual_information / (self.num_agents * len(self.field_of_view_offsets))
        reward = normalized_mi

        for idx, target in enumerate(self.target_positions):
            if idx in self.detected_targets:
                reward -= 2  # Target already detected
            elif self.log_likelihood_ratios[target] > self.target_threshold:
                reward += 10  # New target detected
                self.detected_targets.add(idx)

        reward -= 1  # Flight time penalty
        return reward


# Main script for testing (random policy just to check whether or not the environment is running)
if __name__ == "__main__":
    env = MultiAgentDroneEnv(
        agent_positions=[(1, 1, 60), (99, 99, 240)],
        target_positions=[(30, 30), (70, 70)],
        grid_size=(100, 100),
        radius_field_of_view=5,
        target_threshold=np.log(19),
        max_steps=100,
        likelihood_decay=0.1
    )

    state, _ = env.reset(seed=42)
    agent_paths = {i: [env.state["positions"][i]] for i in range(env.num_agents)}

    for _ in range(env.max_steps):
        actions = env.action_space.sample()   # Just taking a random action at this point, no RL involved here at this moment.
        state, reward, done, _ = env.step(actions)
        for i, pos in enumerate(env.state["positions"]):
            agent_paths[i].append(pos)
        env.render(mode='rgb_array', agent_paths=agent_paths)
        print(f"Reward: {reward}")
        if done:
            break
