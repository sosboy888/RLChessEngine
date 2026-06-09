# RLChessEngine

A chess engine built from scratch in Python, implementing an AlphaZero-style self-play training loop. A convolutional neural network learns both a move policy and a board evaluation (value) from games it plays against itself, guided by Monte Carlo Tree Search.

---

## How It Works

The architecture is directly inspired by DeepMind's AlphaZero paper. The core idea is that the engine improves purely through self-play — no handcrafted evaluation functions or opening books.

### The Three Components

**ZeroNet (`agent.py`)** — the neural network

A dual-headed CNN that takes a board position as input and outputs two things simultaneously:
- **Policy head:** a probability distribution over all 4,672 possible chess moves, representing which moves look promising
- **Value head:** a scalar in [-1, 1] representing who is winning from this position

The board is encoded as a (12, 8, 8) tensor — 12 planes of 8x8, one plane per piece type per colour (6 piece types × 2 colours).

```
Input: (12, 8, 8) board tensor
  → Conv2d(12→256) → ReLU → Conv2d(256→256) → ReLU
      ├── Policy head: Conv2d → Flatten → Linear(128, 4672) → Softmax
      └── Value head:  Conv2d → Flatten → Linear(64, 256) → Linear(256, 1) → Tanh
```

**MCTS (`mcts.py`)** — the search algorithm

Monte Carlo Tree Search uses the neural network to guide exploration. Rather than searching every possible move, MCTS selects moves that balance exploitation (moves the network thinks are good) with exploration (moves not yet tried). After running many simulations from the current position, it returns a visit count distribution over legal moves — this becomes the training target for the policy head.

**Self-play loop (`main.py`)** — data generation and training

The engine plays games against itself. For each position encountered, MCTS produces a move distribution that is stored alongside the board state. After each game, the network is trained on this data using:
- **Policy loss:** cross-entropy between the network's policy output and the MCTS visit distribution
- **Value loss:** MSE between the network's value output and the actual game outcome (+1 win, -1 loss, 0 draw)

This loop runs for 400 iterations, generating 20 self-play games per iteration.

---

## Repository Structure

```
├── agent.py          # ZeroNet: dual-headed CNN (policy + value)
├── mcts.py           # Monte Carlo Tree Search with UCB exploration
├── main.py           # ChessEnvironment, self-play loop, training loop
├── train.py          # Standalone training utilities
├── test.py           # Model evaluation and testing
├── chess_model.pth   # Trained model weights
├── game.pgn          # Example game recorded in PGN format
└── stockfish-8-win/  # Stockfish 8 binary for evaluation
```

---

## Installation

```bash
git clone https://github.com/sosboy888/my-chess-engine
cd my-chess-engine

pip install torch python-chess numpy
```

---

## Usage

### Train from scratch

```bash
python main.py
```

This runs 400 training iterations, generating 20 self-play games per iteration. The trained model is saved to `chess_model.pth` on completion. Training on CPU will be slow — a GPU is recommended.

### Test the trained model

```bash
python test.py
```

Loads `chess_model.pth` and plays a game, printing the board state after each move.

---

## Architecture Details

### Board Representation

Each position is encoded as a (12, 8, 8) tensor:

| Planes | Piece |
|---|---|
| 0–5 | White: Pawn, Knight, Bishop, Rook, Queen, King |
| 6–11 | Black: Pawn, Knight, Bishop, Rook, Queen, King |

Each plane is a binary 8x8 grid — 1 where that piece type exists, 0 elsewhere.

### Action Space

Moves are encoded as `from_square * 64 + to_square`, giving an action space of 4,096 entries (padded to 4,672 to match standard AlphaZero encoding). Only legal moves are considered during play.

### Training

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Iterations | 400 |
| Self-play games per iteration | 20 |
| Policy loss | Cross-entropy |
| Value loss | MSE |

---

## Design Decisions

**Why AlphaZero-style?** Traditional chess engines like Stockfish use handcrafted evaluation functions built from decades of chess knowledge. AlphaZero showed that a neural network trained purely through self-play can reach superhuman strength. This project builds that system from scratch to understand how it works at the implementation level.

**Why L1 loss for value?** The value target is either +1, -1, or 0 — a small integer range. MSE works fine here; the loss landscape is smooth.

**Why MCTS over pure neural network play?** The raw network policy is noisy, especially early in training. MCTS improves decision quality by simulating ahead and combining network guidance with actual search, producing much stronger move distributions as training targets.

---

## Limitations

- The current training run (400 iterations, 20 games each) produces a weak engine — AlphaZero trained for millions of games on specialised TPUs. This is a proof-of-concept implementation.
- No opening book or endgame tablebase.
- Promotion moves (pawn → queen/rook/bishop/knight) use simplified move encoding that may miss some edge cases.
- The engine plays random moves during the very first iterations before the network has learned anything useful.

---

## References

- Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero), DeepMind, 2017
- Browne et al., *A Survey of Monte Carlo Tree Search Methods*, IEEE TCIAIG, 2012
- [python-chess](https://python-chess.readthedocs.io/) — chess move generation and board representation
- [Stockfish 8](https://stockfishchess.org/) — used for engine evaluation
