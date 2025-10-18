import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
import math

class MCTSNode:
    """Node for Monte Carlo Tree Search"""
    
    def __init__(self, state, parent=None, move=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior
        
        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.value = 0.0
        
    def expanded(self):
        return len(self.children) > 0
    
    def value_estimate(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, exploration_weight=1.0):
        if self.visit_count == 0:
            return float('inf')
        
        # UCB formula
        exploitation = self.value_estimate()
        exploration = exploration_weight * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def best_child(self, exploration_weight=1.0):
        if not self.children:
            return None
        
        return max(self.children, key=lambda child: child.ucb_score(exploration_weight))
    
    def add_child(self, state, move, prior):
        child = MCTSNode(state, self, move, prior)
        self.children.append(child)
        return child

class MCTSSolver:
    """Monte Carlo Tree Search solver for sliding puzzles"""
    
    def __init__(self, model, board_size, num_simulations=400, exploration_weight=1.0):
        self.model = model
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
    def get_blank_position(self, state):
        """Find blank tile position"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if state[i][j] == 0:
                    return i, j
        return -1, -1
    
    def move_tile(self, state, direction):
        """Move blank tile in given direction"""
        i, j = self.get_blank_position(state)
        di, dj = direction
        
        new_i, new_j = i + di, j + dj
        if 0 <= new_i < self.board_size and 0 <= new_j < self.board_size:
            new_state = [row[:] for row in state]
            new_state[i][j], new_state[new_i][new_j] = new_state[new_i][new_j], new_state[i][j]
            return new_state
        return None
    
    def get_valid_moves(self, state):
        """Get all valid moves from current state"""
        valid_moves = []
        for move_idx, move in enumerate(self.moves):
            if self.move_tile(state, move) is not None:
                valid_moves.append(move_idx)
        return valid_moves
    
    def state_to_tensor(self, state, board_size):
        """Convert state to neural network input format"""
        vector = []
        for i in range(board_size):
            for j in range(board_size):
                tile = state[i][j]
                vector.append(tile / (board_size * board_size - 1))
                vector.append(i / (board_size - 1))
                vector.append(j / (board_size - 1))
        return torch.FloatTensor(vector).unsqueeze(0)
    
    def create_target_state(self):
        """Create solved puzzle state"""
        state = [[0] * self.board_size for _ in range(self.board_size)]
        num = 1
        for i in range(self.board_size):
            for j in range(self.board_size):
                if i == self.board_size - 1 and j == self.board_size - 1:
                    state[i][j] = 0
                else:
                    state[i][j] = num
                    num += 1
        return state
    
    def is_solved(self, state):
        """Check if state is solved"""
        return state == self.create_target_state()
    
    def run_mcts(self, initial_state, temperature=1.0):
        """Run MCTS from initial state"""
        root = MCTSNode(initial_state)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.expanded():
                node = node.best_child(self.exploration_weight)
                search_path.append(node)
            
            # Expansion
            if not self.is_solved(node.state):
                # Get policy and value from neural network
                state_tensor = self.state_to_tensor(node.state, self.board_size)
                with torch.no_grad():
                    # Set model to eval mode for inference
                    self.model.eval()
                    policy_logits, value = self.model(state_tensor, torch.LongTensor([self.board_size]))
                    policy_probs = torch.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
                
                # Add children for valid moves
                valid_moves = self.get_valid_moves(node.state)
                total_valid_prior = sum(policy_probs[move] for move in valid_moves)
                
                for move in valid_moves:
                    if total_valid_prior > 0:
                        prior = policy_probs[move] / total_valid_prior
                    else:
                        prior = 1.0 / len(valid_moves)
                    
                    new_state = self.move_tile(node.state, self.moves[move])
                    node.add_child(new_state, move, prior)
                
                # Use the value from neural network
                value = value.item()
            else:
                # Puzzle is solved
                value = 1.0
            
            # Backpropagation
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value = -value  # Alternate perspective for opponent
        
        # Calculate move probabilities
        visit_counts = np.array([child.visit_count for child in root.children])
        
        if temperature == 0:
            # Choose move with highest visit count
            best_idx = np.argmax(visit_counts)
            move_probs = np.zeros(len(root.children))
            move_probs[best_idx] = 1.0
        else:
            # Apply temperature
            visit_counts = visit_counts ** (1.0 / temperature)
            move_probs = visit_counts / np.sum(visit_counts)
        
        return move_probs, root
    
    def solve_with_mcts(self, initial_state, max_steps=200):
        """Solve puzzle using MCTS"""
        state = initial_state
        solution = []
        
        for step in range(max_steps):
            if self.is_solved(state):
                return solution, True
            
            move_probs, root = self.run_mcts(state, temperature=0)
            best_idx = np.argmax(move_probs)
            best_move = root.children[best_idx].move
            
            state = self.move_tile(state, self.moves[best_move])
            solution.append(best_move)
        
        return solution, False

class SelfPlayAgent:
    """Agent that plays against itself to generate training data"""
    
    def __init__(self, model, board_size, num_simulations=400):
        self.model = model
        self.board_size = board_size
        self.mcts = MCTSSolver(model, board_size, num_simulations)
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def generate_game(self, max_steps=300, temperature_schedule=None):
        """Generate one self-play game"""
        if temperature_schedule is None:
            temperature_schedule = {
                'initial': 1.0,
                'final': 0.1,
                'decay_steps': 100
            }
        
        # Start from solved state and scramble
        state = self.mcts.create_target_state()
        scramble_moves = random.randint(20, 60)  # Scramble the board
        
        for _ in range(scramble_moves):
            valid_moves = self.mcts.get_valid_moves(state)
            if valid_moves:
                move = random.choice(valid_moves)
                state = self.mcts.move_tile(state, self.moves[move])
        
        initial_state = [row[:] for row in state]  # Copy initial state
        states = []
        mcts_policies = []
        values = []
        
        step = 0
        while step < max_steps and not self.mcts.is_solved(state):
            # Calculate temperature for this step
            if step < temperature_schedule['decay_steps']:
                temperature = (temperature_schedule['initial'] - 
                             (temperature_schedule['initial'] - temperature_schedule['final']) * 
                             step / temperature_schedule['decay_steps'])
            else:
                temperature = temperature_schedule['final']
            
            # Run MCTS to get policy
            move_probs, root = self.mcts.run_mcts(state, temperature)
            
            # Store training data
            states.append(state)
            
            # Convert move probabilities to policy vector
            policy_vector = np.zeros(4)
            for i, child in enumerate(root.children):
                policy_vector[child.move] = move_probs[i]
            mcts_policies.append(policy_vector)
            
            # Choose move according to policy
            if len(move_probs) > 0:
                move_idx = np.random.choice(len(move_probs), p=move_probs)
                move = root.children[move_idx].move
                state = self.mcts.move_tile(state, self.moves[move])
            
            step += 1
        
        # Calculate values based on outcome
        solved = self.mcts.is_solved(state)
        for i in range(len(states)):
            # Value is 1 if puzzle is eventually solved, -1 otherwise
            # But we can also use step-based discounting
            if solved:
                # Closer to solution gets higher value
                steps_remaining = len(states) - i
                value = max(0.1, 1.0 - steps_remaining / len(states))
            else:
                value = -1.0  # Negative value for failure
            
            values.append(value)
        
        return list(zip(states, mcts_policies, values))

class SelfPlayDataset(Dataset):
    """Dataset for self-play training data"""
    
    def __init__(self, data):
        self.data = data
    
    def state_to_vector(self, state, board_size):
        """Convert state to neural network input vector"""
        vector = []
        for i in range(board_size):
            for j in range(board_size):
                tile = state[i][j]
                vector.append(tile / (board_size * board_size - 1))
                vector.append(i / (board_size - 1))
                vector.append(j / (board_size - 1))
        return vector
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, policy, value, board_size = self.data[idx]
        state_vector = self.state_to_vector(state, board_size)
        
        return (
            torch.FloatTensor(state_vector),
            torch.FloatTensor(policy),
            torch.FloatTensor([value]),
            torch.LongTensor([board_size])
        )

class AlphaZeroLoss(nn.Module):
    """Loss function for AlphaZero-style training"""
    
    def __init__(self, policy_weight=1.0, value_weight=1.0, entropy_weight=0.01):
        super(AlphaZeroLoss, self).__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
    
    def forward(self, policy_pred, value_pred, policy_target, value_target):
        # Policy loss
        policy_loss = self.policy_criterion(policy_pred, policy_target)
        
        # Value loss
        value_loss = self.value_criterion(value_pred.squeeze(), value_target.squeeze())
        
        # Entropy regularization
        policy_probs = torch.softmax(policy_pred, dim=1)
        entropy = -torch.sum(policy_probs * torch.log(policy_probs + 1e-8), dim=1).mean()
        
        total_loss = (self.policy_weight * policy_loss + 
                     self.value_weight * value_loss - 
                     self.entropy_weight * entropy)
        
        return total_loss, policy_loss, value_loss, entropy

class AdaptiveSelfPlayNetwork(nn.Module):
    """Neural network that adapts to different board sizes"""
    
    def __init__(self, max_board_size=8, hidden_layers=[512, 256, 128]):
        super(AdaptiveSelfPlayNetwork, self).__init__()
        self.max_board_size = max_board_size
        self.input_size = max_board_size * max_board_size * 3
        
        # Dynamic input projection layer
        self.input_projection = nn.Linear(self.input_size, 512)
        
        # Hidden layers
        layers = []
        prev_size = 512
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(prev_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  # 4 moves
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        # Board size embedding
        self.size_embedding = nn.Embedding(10, 32)
        self.size_projection = nn.Linear(32, prev_size)
    
    def forward(self, x, board_size=None):
        # Project input to fixed size
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Ensure input is the right size, pad if necessary
        current_size = x.size(1)
        if current_size < self.input_size:
            # Pad with zeros
            padding = torch.zeros(x.size(0), self.input_size - current_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif current_size > self.input_size:
            # Truncate (shouldn't happen)
            x = x[:, :self.input_size]
        
        x = self.input_projection(x)
        x = self.hidden_layers(x)
        
        # Add board size information
        if board_size is not None:
            size_embed = self.size_embedding(board_size)
            if size_embed.dim() == 3:
                size_embed = size_embed.squeeze(1)
            size_projected = self.size_projection(size_embed)
            x = x + size_projected
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value

class Phase4Trainer:
    """Phase 4: Advanced Self-Play Training for 7x7 and 8x8"""
    
    def __init__(self, config):
        self.config = config
        self.board_sizes = config["board_sizes"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4)
        )
        
        self.criterion = AlphaZeroLoss(
            policy_weight=1.0,
            value_weight=1.0,
            entropy_weight=config.get("entropy_weight", 0.01)
        )
        
        # Training state
        self.iteration = 0
        self.history = defaultdict(list)
        self.best_value_loss = float('inf')
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=config["replay_buffer_size"])
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('self_play_data', exist_ok=True)
    
    def _initialize_model(self):
        """Initialize adaptive model for different board sizes"""
        max_board_size = max(self.board_sizes)
        
        model = AdaptiveSelfPlayNetwork(
            max_board_size=max_board_size,
            hidden_layers=self.config.get("hidden_layers", [512, 256, 128])
        ).to(self.device)
        
        # Load pre-trained weights if available
        pretrained_path = self.config.get("pretrained_path")
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pre-trained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try to load with strict=False to handle architecture changes
                try:
                    model.load_state_dict(checkpoint, strict=False)
                except:
                    print("Could not load pre-trained weights, starting from scratch")
        
        return model
    
    def generate_self_play_data(self, iteration):
        """Generate self-play data for current iteration"""
        print(f"🎮 Generating self-play data for iteration {iteration}...")
        
        games_per_size = self.config["games_per_iteration"] // len(self.board_sizes)
        all_data = []
        
        # Set model to eval mode for self-play
        self.model.eval()
        
        for board_size in self.board_sizes:
            print(f"  Generating {games_per_size} games for {board_size}x{board_size}")
            
            agent = SelfPlayAgent(
                self.model,
                board_size,
                num_simulations=self.config["mcts_simulations"]
            )
            
            temperature_schedule = self.config.get("temperature_schedule", {
                "initial": 1.0,
                "final": 0.1,
                "decay_steps": 50
            })
            
            successful_games = 0
            for game_idx in tqdm(range(games_per_size), desc=f"{board_size}x{board_size}"):
                try:
                    game_data = agent.generate_game(
                        max_steps=200,  # Reduced for stability
                        temperature_schedule=temperature_schedule
                    )
                    
                    # Add board size and save to replay buffer
                    for state, policy, value in game_data:
                        self.replay_buffer.append((state, policy, value, board_size))
                        all_data.append((state, policy, value, board_size))
                    
                    successful_games += 1
                
                except Exception as e:
                    print(f"Error in game {game_idx}: {e}")
                    continue
            
            print(f"  Successfully generated {successful_games}/{games_per_size} games for {board_size}x{board_size}")
        
        # Set model back to train mode
        self.model.train()
        
        # Save self-play data for this iteration
        if all_data:
            data_file = f"self_play_data/iteration_{iteration}.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(all_data, f)
        
        print(f"✅ Generated {len(all_data)} self-play samples")
        return all_data
    
    def train_iteration(self, iteration_data):
        """Train for one iteration on self-play data"""
        print(f"🎯 Training iteration {self.iteration}...")
        
        # Skip if no data
        if not iteration_data and len(self.replay_buffer) == 0:
            print("❌ No training data available. Skipping training iteration.")
            return 0.0, 0.0, 0.0, 0.0
        
        # Create dataset from replay buffer (sample from recent data)
        if len(self.replay_buffer) > self.config["batch_size"] * 10:
            # Sample from replay buffer
            training_data = random.sample(
                list(self.replay_buffer), 
                min(len(self.replay_buffer), self.config["batch_size"] * 50)
            )
        else:
            training_data = iteration_data
        
        dataset = SelfPlayDataset(training_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.model.train()
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0
        
        pbar = tqdm(dataloader, desc="Training")
        
        for batch_idx, (states, policy_targets, value_targets, sizes) in enumerate(pbar):
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            sizes = sizes.to(self.device)
            
            # Convert policy targets to class indices for CrossEntropyLoss
            _, policy_indices = torch.max(policy_targets, 1)
            
            self.optimizer.zero_grad()
            
            policy_pred, value_pred = self.model(states, sizes.squeeze())
            
            # Calculate loss
            loss, p_loss, v_loss, ent = self.criterion(
                policy_pred, value_pred, policy_indices, value_targets
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get("max_grad_norm", 1.0)
            )
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            policy_loss += p_loss.item()
            value_loss += v_loss.item()
            entropy += ent.item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Policy': f'{p_loss.item():.4f}',
                    'Value': f'{v_loss.item():.4f}'
                })
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_policy_loss = policy_loss / num_batches
        avg_value_loss = value_loss / num_batches
        avg_entropy = entropy / num_batches
        
        # Update history
        self.history['train_loss'].append(avg_loss)
        self.history['policy_loss'].append(avg_policy_loss)
        self.history['value_loss'].append(avg_value_loss)
        self.history['entropy'].append(avg_entropy)
        
        print(f"Iteration {self.iteration}: "
              f"Loss: {avg_loss:.4f} | "
              f"Policy: {avg_policy_loss:.4f} | "
              f"Value: {avg_value_loss:.4f} | "
              f"Entropy: {avg_entropy:.4f}")
        
        return avg_loss, avg_policy_loss, avg_value_loss, avg_entropy
    
    def evaluate_model(self, num_eval_games=20):
        """Evaluate current model performance"""
        print("📊 Evaluating model performance...")
        
        success_rates = {}
        avg_moves = {}
        
        # Set model to eval mode for evaluation
        self.model.eval()
        
        for board_size in self.board_sizes:
            print(f"  Evaluating {board_size}x{board_size}...")
            successes = 0
            total_moves = 0
            
            agent = SelfPlayAgent(
                self.model,
                board_size,
                num_simulations=self.config["mcts_simulations"] // 2  # Faster evaluation
            )
            
            for _ in tqdm(range(num_eval_games), desc=f"Eval {board_size}x{board_size}"):
                # Start from a scrambled state
                state = agent.mcts.create_target_state()
                scramble_moves = random.randint(10, 20)  # Reduced scrambling
                
                for _ in range(scramble_moves):
                    valid_moves = agent.mcts.get_valid_moves(state)
                    if valid_moves:
                        move = random.choice(valid_moves)
                        state = agent.mcts.move_tile(state, agent.moves[move])
                
                # Try to solve
                solution, solved = agent.mcts.solve_with_mcts(state, max_steps=100)
                
                if solved:
                    successes += 1
                    total_moves += len(solution)
            
            success_rate = successes / num_eval_games
            avg_move_count = total_moves / successes if successes > 0 else float('inf')
            
            success_rates[board_size] = success_rate
            avg_moves[board_size] = avg_move_count
            
            print(f"    {board_size}x{board_size}: Success rate: {success_rate:.2%}, "
                  f"Avg moves: {avg_move_count:.1f}")
        
        # Set model back to train mode
        self.model.train()
        
        # Update history
        for size in success_rates:
            self.history[f'success_rate_{size}'].append(success_rates[size])
            self.history[f'avg_moves_{size}'].append(avg_moves[size])
        
        return success_rates, avg_moves
    
    def run_training(self):
        """Run complete self-play training pipeline"""
        print("=" * 80)
        print("🚀 PHASE 4: ADVANCED SELF-PLAY TRAINING (7x7-8x8)")
        print("=" * 80)
        
        start_time = time.time()
        
        for iteration in range(self.config["self_play_iterations"]):
            self.iteration = iteration
            
            print(f"\n🔄 Iteration {iteration + 1}/{self.config['self_play_iterations']}")
            print("-" * 60)
            
            # Step 1: Generate self-play data
            iteration_data = self.generate_self_play_data(iteration)
            
            # Step 2: Train on self-play data (if any)
            if iteration_data or len(self.replay_buffer) > 0:
                train_loss, policy_loss, value_loss, entropy = self.train_iteration(iteration_data)
            else:
                print("❌ No training data available. Skipping training.")
                continue
            
            # Step 3: Evaluate periodically
            if (iteration + 1) % self.config.get("eval_interval", 5) == 0:
                success_rates, avg_moves = self.evaluate_model()
                
                # Save model if it's the best so far
                overall_success = sum(success_rates.values()) / len(success_rates)
                if overall_success > self.history.get('best_success_rate', 0):
                    self.history['best_success_rate'] = overall_success
                    self.save_model("numpuz_selfplay_best.pth")
                    print(f"🏆 New best model! Overall success rate: {overall_success:.2%}")
            
            # Save checkpoint
            if (iteration + 1) % self.config.get("checkpoint_interval", 10) == 0:
                self.save_model(f"numpuz_selfplay_iteration_{iteration+1}.pth")
                self.save_training_history()
        
        total_time = time.time() - start_time
        print(f"\n✅ Self-play training completed in {total_time/3600:.2f} hours")
        
        # Save final model
        self.save_model("numpuz_selfplay_final.pth")
        self.save_training_history()
        
        # Final evaluation
        print("\n🎯 Final Evaluation:")
        success_rates, avg_moves = self.evaluate_model(num_eval_games=50)
        
        return self.history
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': self.iteration,
            'config': self.config,
            'history': dict(self.history),
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, f"models/{filename}")
        print(f"💾 Model saved: models/{filename}")
    
    def save_training_history(self):
        """Save training history and plots"""
        history_file = "models/phase4_selfplay_history.json"
        
        with open(history_file, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot self-play training curves"""
        if not self.history:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        if 'train_loss' in self.history:
            ax1.plot(self.history['train_loss'], label='Total Loss', linewidth=2)
            ax1.plot(self.history['policy_loss'], label='Policy Loss', alpha=0.7)
            ax1.plot(self.history['value_loss'], label='Value Loss', alpha=0.7)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('Self-Play Training Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Success rates
        for key in self.history:
            if key.startswith('success_rate_'):
                size = key.split('_')[-1]
                ax2.plot(self.history[key], label=f'{size}x{size}', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rates by Board Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Average moves
        for key in self.history:
            if key.startswith('avg_moves_'):
                size = key.split('_')[-1]
                ax3.plot(self.history[key], label=f'{size}x{size}', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Average Moves')
        ax3.set_title('Average Moves to Solution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Entropy
        if 'entropy' in self.history:
            ax4.plot(self.history['entropy'], linewidth=2, color='purple')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Entropy')
            ax4.set_title('Policy Entropy')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/phase4_selfplay_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def run_phase4():
    """Run complete Phase 4 self-play training"""
    
    # Configuration for Phase 4
    config_phase4 = {
        "board_sizes": [7, 8],
        "self_play_iterations": 30,  # Reduced for testing
        "games_per_iteration": 40,   # Reduced for testing
        "mcts_simulations": 100,     # Reduced for testing
        "batch_size": 64,            # Reduced for testing
        "learning_rate": 0.0002,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "replay_buffer_size": 50000, # Reduced for testing
        "entropy_weight": 0.01,
        "eval_interval": 3,
        "checkpoint_interval": 5,
        "temperature_schedule": {
            "initial": 1.0,
            "final": 0.1,
            "decay_steps": 30
        },
        "hidden_layers": [256, 128, 64],  # Reduced for testing
        "pretrained_path": None
    }
    
    print("🎯 PHASE 4: ADVANCED SELF-PLAY TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Board sizes: {config_phase4['board_sizes']}")
    print(f"Iterations: {config_phase4['self_play_iterations']}")
    print(f"Games per iteration: {config_phase4['games_per_iteration']}")
    print(f"MCTS simulations: {config_phase4['mcts_simulations']}")
    print(f"Replay buffer: {config_phase4['replay_buffer_size']}")
    print("=" * 60)
    
    try:
        # Initialize and run trainer
        trainer = Phase4Trainer(config_phase4)
        history = trainer.run_training()
        
        if history is not None:
            print("\n" + "=" * 80)
            print("🎉 PHASE 4 COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📊 Final model: models/numpuz_selfplay_final.pth")
            
            # Print final success rates
            if 'best_success_rate' in history:
                print(f"🏆 Best success rate: {history['best_success_rate']:.2%}")
            
            print("=" * 80)
        
        return history
        
    except Exception as e:
        print(f"❌ Error during Phase 4 training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_phase4()