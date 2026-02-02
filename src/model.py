import torch.nn as nn
import torch
import numpy as np
import chess

"""
Channel(s), Data,                  Description
0 - 5,      Current Player Pieces, "P, N, B, R, Q, K"
6 - 11,     Opponent Pieces,       "p, n, b, r, q, k"
12,         Side to Move,          "All 0s for White, 1s for Black (or vice versa)"
13 - 16,    Castling,              "4 planes (WK, WQ, BK, BQ)"
17,         En Passant,            "Your current binary map"
18,         50-move clock,         "Normalized (0.0 to 1.0)"

Looking at this tensor, you may wonder how is Side to Move useful?
This channel breaks the symmetry, since the white always moves first,
and the chess is therefore asymmetrical game. 
"""

class AlphaChess(nn.Module):
    """ This is the main AlphaChess Model combining all components """
    def __init__(self, input_channels=19, num_filters=128, num_res_blocks=10, board_size=8):
        super().__init__()
        self.cnn = CNN(input_channels, num_filters)
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_res_blocks)])
        self.policy_head = PolicyHead(num_filters, board_size)
        self.value_head = ValueHead(num_filters, board_size)

    
    def forward(self, x):
        out = self.cnn(x)
        for res_block in self.res_blocks:
            out = res_block(out)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value


class ResBlock(nn.Module):
    """ This ResBlock is the Main Body Block """
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual # Skip connection
        out = nn.ReLU()(out)
        return out
    
    
class CNN(nn.Module):
    """ This CNN is the Input Block """
    def __init__(self, input_channels, num_filters):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)

    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = nn.ReLU()(out)
        return out
    

class PolicyHead(nn.Module):
    """ This is the Policy Head """
    def __init__(self, in_channels, board_size):
        super().__init__()
        # 1x1 convolution to reduce channel dimensions, summarizing features
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels=2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        # Fully connected layer to output move probabilities, 8x8 possible from and to squares respectively
        self.fc = nn.Linear(2 * board_size * board_size, (board_size * board_size) * (board_size * board_size))

    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = nn.ReLU()(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out # No softmax here; will be applied in loss function
    

class ValueHead(nn.Module):
    """ This is the Value Head """
    def __init__(self, in_channels, board_size):
        super().__init__()
        # 1x1 convolution to reduce channel dimensions
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels=1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        # Fully connected layers to output -1 to 1 value, indicating loss or win
        self.fc1 = nn.Linear(board_size * board_size, 128)
        self.fc2 = nn.Linear(128, 1)

    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = nn.ReLU()(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out = self.fc2(out)
        out = torch.tanh(out) # Output between -1 and 1
        return out


class BoardEncoder:
    """ Encodes the chess board into a 19x8x8 tensor for NN input. """
    def __init__(self):
        pass


    def encode(self, board: chess.Board) -> torch.Tensor:
        """
        Converts board to a 19x8x8 tensor.
        Always from the current player's perspective.
        """
        # 1. Determine perspective
        who_moves = board.turn 

        # 2. Initialize tensor with zeros
        tensor = np.zeros((19, 8, 8), dtype=np.float32)

        # Function to get piece mask for a specific color
        def fill_pieces(color, channel_offset):
            piece_map = board.piece_map()
            for square, piece in piece_map.items():
                if piece.color == color:
                    piece_type = piece.piece_type
                    row = 7 - chess.square_rank(square)
                    col = chess.square_file(square)
                    tensor[channel_offset + piece_type - 1, row, col] = 1.0

        # 3. Fill piece positions
        # Channels 0-5: Current player's pieces
        fill_pieces(who_moves, 0)

        # Channels 6-11: Opponent pieces
        fill_pieces(not who_moves, 6)

        # 4. Side to move (Channel 12)
        if who_moves == chess.WHITE:
            tensor[12, :, :] = 0.0
        else:
            tensor[12, :, :] = 1.0

        # 5. Castling rights (Channels 13-16)
        # Current player's King-side, Current player's Queen-side, Opponent's King-side, Opponent's Queen-side
        tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(who_moves) else 0.0
        tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(who_moves) else 0.0
        tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(not who_moves) else 0.0
        tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(not who_moves) else 0.0

        # 6. En passant target square (Channel 17)
        # Need to cover all cases
        if board.ep_square is not None:
            row = 7 - chess.square_rank(board.ep_square)
            col = chess.square_file(board.ep_square)
            tensor[17, row, col] = 1.0

        # 7. 50-move rule counter (Channel 18)
        tensor[18, :, :] = board.halfmove_clock / 50.0

        return torch.tensor(tensor)