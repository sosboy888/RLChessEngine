import math
import torch
import chess

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.prior = 0

    def ucb_score(self, total_visits, c=1.0):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * self.prior * math.sqrt(total_visits) / (1 + self.visits)

class MCTS:
    def __init__(self, model, env):
        self.model = model
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    def search(self, root, num_simulations=100):
        for _ in range(num_simulations):
            node = root
            path = []

            # Selection
            while node.children:
                total_visits = sum(child.visits for child in node.children.values())
                node = max(node.children.values(), key=lambda n: n.ucb_score(total_visits))
                path.append(node)

            # Expansion
            if not node.children and not self.env.is_terminal():
                # Expand children with move indices mapped to policy values
                policy, _ = self.model(torch.FloatTensor(self.env.get_state()).unsqueeze(0).to(self.device))

                # Move tensor to CPU and convert to NumPy for indexing
                policy = policy.detach().cpu().numpy()  # Use .cpu() before converting to numpy

                for move in self.env.legal_moves():
                    move_index = move_to_index(move)
                    self.env.board.push(move)
                    new_state = self.env.get_state()
                    self.env.board.pop()
                    node.children[move] = MCTSNode(new_state, node)
                    node.children[move].prior = policy[0, move_index]  # Use the mapped index

            # Simulation
            leaf = node
            outcome = self.env.simulate_game()

            # Backpropagation
            for node in path:
                node.visits += 1
                node.value += outcome


def move_to_index(move):
        """Encodes a chess.Move into a unique index in the action space."""
        return move.from_square * 64 + move.to_square

def index_to_move(index):
    """Decodes an index into a chess.Move."""
    from_square = index // 64
    to_square = index % 64
    return chess.Move(from_square, to_square)