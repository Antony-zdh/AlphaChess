# Alpha Chess

This is a RL project that aims to build a chess AI from scratch.

## Milestones

- [x] **1. The Arena**: Set up the chess environment and the basic game loop. Use a random agent to play against the user.
- [x] **2. Visualization**: Set up a visualization tool to display the game board real time.
- [x] **3. The Brain**: Design a small yet capable Neural Network in PyTorch. We need to make sure the architecture is reasonable and strong enough.
- [x] **4. Data Pipeline**: Set up a data pipeline to collect game data from web. These data are curated for Supervised Fine Tuning.
- [x] **5. Training Infrastructure**: Set up the training infrastructure. At this point, we do not need to support RL yet, but we need to prepare for SFT.
- [x] **6. Supervised Fine Tuning**: Train the Neural Network with curated data from Data Pipeline.
- [ ] **7. RL Infrastructure**: Set up the RL infrastructure. We need to support both on-policy and off-policy algorithms. We have to dedicate the update algorithm. During RL, the AI will play against itself to collect experience.
- [ ] **8. RL Training**: Train the Neural Network with RL algorithms.
- [ ] **9. Testing and Evaluation**: Test the trained Neural Network with the Arena. A finetuned AI (Standard AI of similar size) from web should be selected as the opponent for our AI to play with. Win rate, loss rate, and draw rate should be recorded.

## Architecture 
(num_filters = 128, input dimension = 19x8x8)

- **Input CNN Block**: 19x8x8 -> 128x8x8
- **10 Residual Blocks**: num_filters = 128
- **Policy Head**: 128 -> 2
- **Value Head**: 128 -> 1

### Input Representation
| Channel(s) | Data | Description |
| :--- | :--- | :--- |
| 0 - 5 | Current Player Pieces | `P, N, B, R, Q, K` |
| 6 - 11 | Opponent Pieces | `p, n, b, r, q, k` |
| 12 | Side to Move | All 1s for White, 0s for Black |
| 13 - 16 | Castling | 4 planes (WK, WQ, BK, BQ) |
| 17 | En Passant | Your current binary map |
| 18 | 50-move clock | Normalized (0.0 to 1.0) |

## Data Source
- **Lichess**: https://lichess.org/

**Target Players** (Top 10 GMs, 500 games each):
- `DrNykterstein` (Magnus Carlsen)
- `Dr_Tiger` (Hikaru Nakamura)
- `Alireza2003` (Alireza Firouzja)
- `FabianoCaruanaa` (Fabiano Caruana)
- `STL_Aronian` (Levon Aronian)
- `RebeccaHarris` (Daniel Naroditsky)
- `Penguingim1` (Andrew Tang)
- `Nihalsarin2004` (Nihal Sarin)
- `Zhigalko_Sergei` (Sergei Zhigalko)
- `Night-King96` (Oleksandr Bortnyk)

**Status**: Successful Download: 3988/5000 

# Usage
## Training
```bash
bash scripts/train.sh
```

## Hyperparameters
- **Model Path**: `models/alpha_chess_epoch_16.pth` (Likely overfitted after epoch 16)
- **Training Epochs**: 20
- **Batch Size**: 256
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss for Policy, MSELoss for Value
- **Learning Rate Scheduler**: CosineAnnealingLR

## Inference
```bash
bash scripts/run.sh
```