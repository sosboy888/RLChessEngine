import chess
import numpy as np
import torch.optim as optim
from mcts import MCTS, MCTSNode
from agent import ZeroNet
import torch

class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()
    
    def reset(self):
        self.board.reset()
        return self.get_state()

    def step(self, move):
        if move in self.board.legal_moves:
            self.board.push(move)
            reward = 0
            done = self.board.is_game_over()
            return self.get_state(), reward, done
        else:
            return self.get_state(), -1, False  # Penalty for illegal move
    
    def get_state(self):
    # Convert board to a tensor of shape (8, 8, 12)
        state = np.zeros((8, 8, 12), dtype=int)
        piece_map = self.board.piece_map()
        for square, piece in piece_map.items():
            plane = piece.piece_type - 1 + (6 if piece.color else 0)
            row, col = divmod(square, 8)
            state[row, col, plane] = 1
        return state.transpose(2, 0, 1)  # Transpose to (12, 8, 8)

    def legal_moves(self):
        return list(self.board.legal_moves)

    def is_terminal(self):
        return self.board.is_game_over()

    def simulate_game(self):
        # Simulate random play to evaluate the current state
        while not self.board.is_game_over():
            move = np.random.choice(list(self.board.legal_moves))
            self.board.push(move)
        if self.board.is_checkmate():
            return 1 if self.board.turn else -1
        return 0  # Draw
    
    def move_to_index(self, move):
        """Encodes a chess.Move into a unique index in the action space."""
        return move.from_square * 64 + move.to_square

    def index_to_move(self, index):
        """Decodes an index into a chess.Move."""
        from_square = index // 64
        to_square = index % 64
        move = chess.Move(from_square, to_square)
        return move if move in chess.Board().legal_moves else None

def move_to_index(move):
        """Encodes a chess.Move into a unique index in the action space."""
        return move.from_square * 64 + move.to_square

def index_to_move(index):
    """Decodes an index into a chess.Move."""
    from_square = index // 64
    to_square = index % 64
    move = chess.Move(from_square, to_square)
    return move if move in chess.Board().legal_moves else None


def self_play(env, model, num_games=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure model is on the same device
    model = model.to(device)
    data = []
    for _ in range(num_games):
        state = env.reset()
        done = False
        while not done:
            mcts = MCTS(model, env)
            root = MCTSNode(state)
            mcts.search(root)

            # Map move probabilities to a policy array
            legal_moves = list(env.legal_moves())
            policy = np.zeros(4672)  # Action space size for chess
            for move in legal_moves:
                move_index = move_to_index(move)
                policy[move_index] = root.children[move].visits if move in root.children else 0

            # Normalize policy, handle edge case for sum == 0
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy /= policy_sum
            else:
                policy = np.ones_like(policy) / len(policy)

            # Check for no legal moves
            if len(legal_moves) == 0:
                # Game is over due to checkmate or stalemate
                break

            # Sample a legal move
            legal_move_indices = [move_to_index(move) for move in legal_moves]
            legal_policy = np.array([policy[index] for index in legal_move_indices])
            legal_policy /= np.sum(legal_policy)
            selected_index = np.random.choice(len(legal_policy), p=legal_policy)
            selected_move = legal_moves[selected_index]

            # Apply the move and get the next state
            next_state, _, done = env.step(selected_move)

            # Save data for training (state, policy, placeholder reward)
            data.append((state, policy, 0))  # Placeholder reward
            state = next_state
    return data




def train(model, optimizer, data):
    model.train()
    for state, policy_target, value_target in data:
        # Move state, policy_target, and value_target to the same device as the model
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model.device)  # Move to device
        policy_target = torch.FloatTensor(policy_target).to(model.device)  # Move to device
        value_target = torch.FloatTensor([value_target]).to(model.device)  # Move to device

        optimizer.zero_grad()
        policy_pred, value_pred = model(state_tensor)
        
        # Compute the loss
        policy_loss = -torch.sum(policy_target * torch.log(policy_pred))
        value_loss = (value_pred - value_target) ** 2
        loss = policy_loss + value_loss
        
        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Initialize environment and model, move model to the appropriate device (GPU or CPU)
    env = ChessEnvironment()
    model = ZeroNet().to(device)  # Move the model to GPU or CPU
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for iteration in range(400):  # Training iterations
        print(f"Iteration {iteration}")
        data = self_play(env, model, num_games=20)  # Ensure self_play function returns data correctly
        train(model, optimizer, data)  # Ensure the 'train' function works with GPU

    # Test the trained model
    env.reset()
    while not env.is_terminal():
        state = env.get_state()
        
        # Move the state tensor to the same device as the model
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Get the policy (also moved to GPU)
        policy, _ = model(state_tensor)

        # Map policy to legal moves
        legal_moves = list(env.legal_moves())
        legal_move_indices = [move_to_index(move) for move in legal_moves]
        legal_policy = [policy[0, index].item() for index in legal_move_indices]

        # Select the best legal move
        best_move_index = torch.argmax(torch.tensor(legal_policy)).item()
        selected_move = legal_moves[best_move_index]

        # Apply the selected move
        env.step(selected_move)
        print(env.board)

    # Save the model after training
    torch.save(model.state_dict(), 'chess_model.pth')


