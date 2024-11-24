import chess
import chess.pgn
import chess.engine
import torch
import io
from agent import ZeroNet
from main import ChessEnvironment

# Initialize the Stockfish engine
stockfish_path = "stockfish-8-win/Windows/stockfish_8_x64.exe"  # Path to your Stockfish binary
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

# Load your trained model
model = ZeroNet()  # Ensure this matches your model's architecture
model.load_state_dict(torch.load('chess_model.pth'))
model.eval()

# Function to convert chess board state to model input
def fen_to_tensor(fen):
    board = chess.Board(fen)
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11,  # Black pieces
        '.': -1  # Empty square (we'll leave this as zero for simplicity)
    }

    # Initialize an empty tensor with the shape [12, 8, 8]
    board_tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

    for i in range(8):
        for j in range(8):
            piece = board.piece_at(i * 8 + j)
            if piece:
                piece_type = piece.symbol()
                # Set the appropriate piece channel to 1
                board_tensor[piece_map.get(piece_type, -1), i, j] = 1  # Assign 1 if piece is present

    # Add batch dimension to the tensor [1, 12, 8, 8]
    return board_tensor.unsqueeze(0)  # Unsqueeze to make it [1, 12, 8, 8]

def board_to_input(board):
    fen = board.fen()  # Get FEN string from the board
    return fen_to_tensor(fen)  # Convert FEN to tensor and add batch dimension

def move_to_index(move):
        """Encodes a chess.Move into a unique index in the action space."""
        return move.from_square * 64 + move.to_square

def index_to_move(index):
    """Decodes an index into a chess.Move."""
    from_square = index // 64
    to_square = index % 64
    move = chess.Move(from_square, to_square)
    return move if move in chess.Board().legal_moves else None


# Start a new game
board = chess.Board()

# Initialize a list to keep track of the moves (for PGN generation)
game_moves = []

while not board.is_game_over():
    # Get model's move (using your trained model)
    state = board_to_input(board)  # This now returns a tensor suitable for the model
    policy, _ = model(state)

    # Convert the model's output to a move
    legal_moves = list(board.legal_moves)
    legal_move_indices = [move_to_index(move) for move in legal_moves]
    legal_policy = [policy[0, index].item() for index in legal_move_indices]
    best_move_index = torch.argmax(torch.tensor(legal_policy)).item()
    selected_move = legal_moves[best_move_index]

    # Play the move
    board.push(selected_move)
    game_moves.append(selected_move)  # Append the move to the list
    print("Model Move:", selected_move)
    print(board)

    # Let Stockfish play its move
    result = engine.play(board, chess.engine.Limit(time=2.0))  # Stockfish plays within a time limit of 2 seconds
    board.push(result.move)
    game_moves.append(result.move)  # Append Stockfish's move to the list
    print("Stockfish Move:", result.move)
    print(board)

# Close the Stockfish engine
game = chess.pgn.Game()

# Add moves to the game object
node = game.add_variation(game_moves[0])
for move in game_moves[1:]:
    node = node.add_variation(move)

# Save the PGN to a file
with open("game.pgn", "w") as f:
    exporter = chess.pgn.FileExporter(f)
    game.accept(exporter)

# Print the result of the game
print("Game Over!")
print(board.result())
