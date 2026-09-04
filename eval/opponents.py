import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import AlphaChess, BoardEncoder
from src.mcts import BatchMCTS


# ---------------------------------------------------------------------------
# PyTorch port of the chess-deep-rl ResNet (channels_first, runs on MPS/GPU)
# ---------------------------------------------------------------------------

class _DeepRLResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class _DeepRLNet(nn.Module):
    N_FILTERS   = 256
    N_RESIDUAL  = 19
    POLICY_SIZE = 4672   # 8*8*73

    def __init__(self) -> None:
        super().__init__()
        F = self.N_FILTERS
        # Stem
        self.conv_in = nn.Conv2d(8, F, 3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(F, eps=1e-3, momentum=0.01)
        # Residual tower
        self.res = nn.ModuleList([_DeepRLResBlock(F) for _ in range(self.N_RESIDUAL)])
        # Policy head: conv(256→2,1x1) + BN + relu + flatten + linear(304,4672,sigmoid)
        self.p_conv = nn.Conv2d(F, 2, 1)
        self.p_bn   = nn.BatchNorm2d(2, eps=1e-3, momentum=0.01)
        self.p_fc   = nn.Linear(2 * 8 * 19, self.POLICY_SIZE)
        # Value head: conv(256→1,1x1) + BN + relu + linear(152,256) + relu + linear(256,1,tanh)
        self.v_conv = nn.Conv2d(F, 1, 1)
        self.v_bn   = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
        self.v_fc1  = nn.Linear(1 * 8 * 19, 256)
        self.v_fc2  = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res:
            x = block(x)
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = torch.sigmoid(self.p_fc(p.flatten(1)))
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v.flatten(1)))))
        return p, v


def _load_deep_rl_weights(net: _DeepRLNet, h5_path: str) -> None:
    """Transfer weights from the original Keras NCHW h5 model to a _DeepRLNet."""
    try:
        from tf_keras.models import load_model
    except ImportError:
        from tensorflow.keras.models import load_model

    keras_model = load_model(h5_path, compile=False)
    layers = [l for l in keras_model.layers if l.get_weights()]

    def _conv(k: np.ndarray) -> torch.Tensor:
        # Keras (kH,kW,Cin,Cout) → PyTorch (Cout,Cin,kH,kW)
        return torch.from_numpy(k.transpose(3, 2, 0, 1).copy())

    def _bn(keras_w: list, pt_bn: nn.BatchNorm2d) -> None:
        gamma, beta, mean, var = keras_w
        pt_bn.weight.data.copy_(torch.from_numpy(gamma))
        pt_bn.bias.data.copy_(torch.from_numpy(beta))
        pt_bn.running_mean.copy_(torch.from_numpy(mean))
        pt_bn.running_var.copy_(torch.from_numpy(var))

    # Stem (layers[0]=conv, layers[1]=bn)
    net.conv_in.weight.data.copy_(_conv(layers[0].get_weights()[0]))
    _bn(layers[1].get_weights(), net.bn_in)

    # 19 residual blocks, each occupying 4 keras layers (conv,bn,conv,bn)
    for i, block in enumerate(net.res):
        base = 2 + i * 4
        block.conv1.weight.data.copy_(_conv(layers[base].get_weights()[0]))
        _bn(layers[base + 1].get_weights(), block.bn1)
        block.conv2.weight.data.copy_(_conv(layers[base + 2].get_weights()[0]))
        _bn(layers[base + 3].get_weights(), block.bn2)

    # Policy head (keras layer 78): [conv_k, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, fc_k, fc_b]
    pw = layers[78].get_weights()
    net.p_conv.weight.data.copy_(_conv(pw[0]))
    net.p_conv.bias.data.copy_(torch.from_numpy(pw[1]))
    _bn(pw[2:6], net.p_bn)
    net.p_fc.weight.data.copy_(torch.from_numpy(pw[6].T.copy()))   # (in,out)→(out,in)
    net.p_fc.bias.data.copy_(torch.from_numpy(pw[7]))

    # Value head (keras layer 79): [conv_k, conv_b, bn_gamma, bn_beta, bn_mean, bn_var, fc1_k, fc1_b, fc2_k, fc2_b]
    vw = layers[79].get_weights()
    net.v_conv.weight.data.copy_(_conv(vw[0]))
    net.v_conv.bias.data.copy_(torch.from_numpy(vw[1]))
    _bn(vw[2:6], net.v_bn)
    net.v_fc1.weight.data.copy_(torch.from_numpy(vw[6].T.copy()))
    net.v_fc1.bias.data.copy_(torch.from_numpy(vw[7]))
    net.v_fc2.weight.data.copy_(torch.from_numpy(vw[8].T.copy()))
    net.v_fc2.bias.data.copy_(torch.from_numpy(vw[9]))


# ---------------------------------------------------------------------------
# PyTorch port of chess-ai (DenseNet-style, channels_last → NCHW on MPS/GPU)
# ---------------------------------------------------------------------------
# Graph (NHWC in Keras → NCHW in PyTorch):
#   conv1(12→32)+BN+ReLU → a1
#   conv2(32→64)+BN+ReLU → a2
#   conv3(64→256)+BN+ReLU → a3
#   cat(a3,a3)=512 → conv4(512→256)+BN+ReLU → a4
#   cat(a4,a2)=320 → conv5(320→256)+BN+ReLU → a5
#   cat(a5,a1)=288 → conv6(288→256)+BN+ReLU → a6
#   1×1conv(256→256)+BN → 1×1conv(256→64)+BN → 1×1conv(64→1)+BN
#   → spatial-softmax over H×W

class _ChessAINet(nn.Module):
    EPS = 1e-3
    MOM = 0.01  # 1 - keras momentum (0.99)

    def __init__(self) -> None:
        super().__init__()
        E, M = self.EPS, self.MOM
        self.conv1 = nn.Conv2d(12,  32,  3, padding=1)
        self.bn1   = nn.BatchNorm2d(32,  eps=E, momentum=M)
        self.conv2 = nn.Conv2d(32,  64,  3, padding=1)
        self.bn2   = nn.BatchNorm2d(64,  eps=E, momentum=M)
        self.conv3 = nn.Conv2d(64,  256, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(256, eps=E, momentum=M)
        self.conv4 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(256, eps=E, momentum=M)
        self.conv5 = nn.Conv2d(320, 256, 3, padding=1)
        self.bn5   = nn.BatchNorm2d(256, eps=E, momentum=M)
        self.conv6 = nn.Conv2d(288, 256, 3, padding=1)
        self.bn6   = nn.BatchNorm2d(256, eps=E, momentum=M)
        # Dense layers as 1×1 convolutions
        self.fc1   = nn.Conv2d(256, 256, 1)
        self.bn7   = nn.BatchNorm2d(256, eps=E, momentum=M)
        self.fc2   = nn.Conv2d(256, 64,  1)
        self.bn8   = nn.BatchNorm2d(64,  eps=E, momentum=M)
        self.fc3   = nn.Conv2d(64,  1,   1)
        self.bn9   = nn.BatchNorm2d(1,   eps=E, momentum=M)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, 8, 8) NCHW
        a1 = F.relu(self.bn1(self.conv1(x)))           # (B, 32,  8, 8)
        a2 = F.relu(self.bn2(self.conv2(a1)))          # (B, 64,  8, 8)
        a3 = F.relu(self.bn3(self.conv3(a2)))          # (B, 256, 8, 8)
        a4 = F.relu(self.bn4(self.conv4(torch.cat([a3, a3], 1))))   # 512→256
        a5 = F.relu(self.bn5(self.conv5(torch.cat([a4, a2], 1))))   # 320→256
        a6 = F.relu(self.bn6(self.conv6(torch.cat([a5, a1], 1))))   # 288→256
        # Dense(relu) + BN stack — activation is *inside* each Dense layer in Keras
        d = self.bn7(F.relu(self.fc1(a6)))             # (B, 256, 8, 8)
        d = self.bn8(F.relu(self.fc2(d)))              # (B,  64, 8, 8)
        d = self.bn9(F.relu(self.fc3(d)))              # (B,   1, 8, 8)
        # Spatial softmax over H×W
        B, C, H, W = d.shape
        return torch.softmax(d.view(B, C, H * W), dim=2).view(B, C, H, W)


def _load_chess_ai_weights(net: _ChessAINet, h5_path: str) -> None:
    """Transfer weights from a Keras chess-ai h5 model to a _ChessAINet."""
    try:
        from tf_keras.models import load_model
    except ImportError:
        from tensorflow.keras.models import load_model

    keras_model = load_model(h5_path, compile=False)
    layers = [l for l in keras_model.layers if l.get_weights()]
    # Keras layer order: conv1 bn1 conv2 bn2 conv3 bn3 conv4 bn4 conv5 bn5 conv6 bn6
    #                    dense1 bn7 dense2 bn8 dense3 bn9   (18 weighted layers total)

    def _conv(k: np.ndarray) -> torch.Tensor:
        # Keras NHWC (kH,kW,Cin,Cout) → PyTorch NCHW (Cout,Cin,kH,kW)
        return torch.from_numpy(k.transpose(3, 2, 0, 1).copy())

    def _dense(k: np.ndarray) -> torch.Tensor:
        # Keras Dense (in,out) → PyTorch Conv2d(1x1) weight (out,in,1,1)
        return torch.from_numpy(k.T.copy()).unsqueeze(-1).unsqueeze(-1)

    def _bn(keras_w: list, pt_bn: nn.BatchNorm2d) -> None:
        gamma, beta, mean, var = keras_w
        pt_bn.weight.data.copy_(torch.from_numpy(gamma))
        pt_bn.bias.data.copy_(torch.from_numpy(beta))
        pt_bn.running_mean.copy_(torch.from_numpy(mean))
        pt_bn.running_var.copy_(torch.from_numpy(var))

    conv_layers = [net.conv1, net.conv2, net.conv3, net.conv4, net.conv5, net.conv6]
    bn_layers   = [net.bn1,   net.bn2,   net.bn3,   net.bn4,   net.bn5,   net.bn6]
    fc_layers   = [net.fc1, net.fc2, net.fc3]
    fc_bns      = [net.bn7, net.bn8, net.bn9]

    for i, (pt_conv, pt_bn) in enumerate(zip(conv_layers, bn_layers)):
        kw = layers[i * 2].get_weights()       # [kernel, bias]
        pt_conv.weight.data.copy_(_conv(kw[0]))
        pt_conv.bias.data.copy_(torch.from_numpy(kw[1]))
        _bn(layers[i * 2 + 1].get_weights(), pt_bn)

    for i, (pt_fc, pt_bn) in enumerate(zip(fc_layers, fc_bns)):
        kw = layers[12 + i * 2].get_weights()  # [kernel, bias]
        pt_fc.weight.data.copy_(_dense(kw[0]))
        pt_fc.bias.data.copy_(torch.from_numpy(kw[1]))
        _bn(layers[12 + i * 2 + 1].get_weights(), pt_bn)


@dataclass
class OpponentSpec:
    name: str
    opponent: "BaseOpponent"


class BaseOpponent:
    supports_batch = False

    def select_move(self, board: chess.Board) -> chess.Move:
        # Default: delegate to batch path for batch-capable opponents.
        if self.supports_batch:
            return self.select_moves_batch([board])[0]
        raise NotImplementedError

    def select_moves_batch(self, boards: List[chess.Board]) -> List[chess.Move]:
        return [self.select_move(board) for board in boards]


class AlphaChessOpponent(BaseOpponent):
    supports_batch = True

    def __init__(self, model_path: str, device: torch.device, greedy: bool = False) -> None:
        self.device = device
        self.greedy = greedy
        self.model = AlphaChess().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.encoder = BoardEncoder()

    def select_moves_batch(self, boards: List[chess.Board]) -> List[chess.Move]:
        state_tensors = [self.encoder.encode(board) for board in boards]
        states = torch.stack(state_tensors).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.model(states)

        moves = []
        for idx, board in enumerate(boards):
            legal_moves = list(board.legal_moves)
            legal_move_indices = [move.from_square * 64 + move.to_square for move in legal_moves]
            mask = torch.zeros_like(policy_logits[idx])
            mask[legal_move_indices] = 1.0
            masked_logits = policy_logits[idx].masked_fill(mask == 0, float("-inf"))
            if self.greedy:
                action = torch.argmax(masked_logits).item()
            else:
                probs = torch.softmax(masked_logits, dim=0)
                action = torch.multinomial(probs, 1).item()
            moves.append(self._action_to_move(board, action))

        return moves

    @staticmethod
    def _action_to_move(board: chess.Board, action: int) -> chess.Move:
        move_from = action // 64
        move_to = action % 64
        legal_moves = list(board.legal_moves)
        candidates = [m for m in legal_moves if m.from_square == move_from and m.to_square == move_to]
        if candidates:
            queen_promo = [m for m in candidates if m.promotion == chess.QUEEN]
            return queen_promo[0] if queen_promo else candidates[0]
        return legal_moves[0]


class AlphaZeroOpponent(BaseOpponent):
    """AlphaChess net + MCTS search (AlphaZero-style). Uses the policy head as a
    tree prior and the value head to evaluate leaves. Fits the batched eval
    harness: select_moves_batch runs one MCTS per board with batched leaf eval.
    """
    supports_batch = True

    def __init__(self, model_path: str, device: torch.device,
                 n_simulations: int = 400, temperature: float = 0.0,
                 c_puct: float = 4.0, greedy: bool = True) -> None:
        self.device = device
        self.model = AlphaChess().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.encoder = BoardEncoder()
        # greedy play -> temperature 0 (argmax over visit counts)
        temp = 0.0 if greedy else temperature
        self.mcts = BatchMCTS(
            model=self.model, encoder=self.encoder, device=device,
            n_simulations=n_simulations, c_puct=c_puct, temperature=temp)

    def select_moves_batch(self, boards: List[chess.Board]) -> List[chess.Move]:
        actions = self.mcts.search(boards)
        moves = []
        for board, a in zip(boards, actions):
            if a is None:  # terminal board (play_games shouldn't call on these)
                legal = list(board.legal_moves)
                moves.append(legal[0] if legal else chess.Move.null())
            else:
                moves.append(AlphaChessOpponent._action_to_move(board, a))
        return moves


class ChessAIKerasOpponent(BaseOpponent):
    def __init__(self, model_dir: str) -> None:
        try:
            from tf_keras.models import load_model
        except ImportError:
            from tensorflow.keras.models import load_model

        from_path = os.path.join(model_dir, "from.h5")
        to_path = os.path.join(model_dir, "to.h5")
        if not os.path.exists(from_path) or not os.path.exists(to_path):
            raise FileNotFoundError(f"Missing from.h5 or to.h5 in {model_dir}")

        self.from_model = load_model(from_path, compile=False)
        self.to_model = load_model(to_path, compile=False)

    def select_move(self, board: chess.Board) -> chess.Move:
        fen = board.fen()
        flip = board.turn == chess.BLACK
        if flip:
            fen = self._invert_fen(fen)

        arr = self._fen_to_matrix(fen)
        arr = arr.reshape((1,) + arr.shape)

        from_matrix = self.from_model.predict(arr, verbose=0).reshape((8, 8))
        to_matrix = self.to_model.predict(arr, verbose=0).reshape((8, 8))

        if flip:
            from_matrix = np.flip(from_matrix, axis=0)
            to_matrix = np.flip(to_matrix, axis=0)

        legal_moves = list(board.legal_moves)
        scores = []
        for move in legal_moves:
            uci = move.uci()[:4]
            from_row, from_col, to_row, to_col = self._uci_to_row_col(uci)
            scores.append(from_matrix[from_row][from_col] * to_matrix[to_row][to_col])

        scores = np.maximum(scores, 0)
        total = scores.sum()
        probs = scores / total if total > 0 else np.ones(len(scores)) / len(scores)
        return legal_moves[np.random.choice(len(legal_moves), p=probs)]

    @staticmethod
    def _uci_to_row_col(uci: str) -> Tuple[int, int, int, int]:
        sq_from = uci[:2]
        sq_to = uci[2:]

        def parse_square(sq: str) -> Tuple[int, int]:
            col = ord(sq[0]) - ord("a")
            row = 7 - (int(sq[1]) - 1)
            return row, col

        return parse_square(sq_from) + parse_square(sq_to)

    @staticmethod
    def _fen_to_matrix(fen: str) -> np.ndarray:
        fen = fen.split()[0]

        piece_dict = {
            "p": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "P": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "n": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "N": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "b": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "B": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "r": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "R": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "q": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "Q": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "k": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "K": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ".": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }

        row_arr = []
        rows = fen.split("/")
        for row in rows:
            arr = []
            for ch in str(row):
                if ch.isdigit():
                    for _ in range(int(ch)):
                        arr.append(piece_dict["."])
                else:
                    arr.append(piece_dict[ch])
            row_arr.append(arr)

        return np.array(row_arr)

    @staticmethod
    def _invert_fen(fen: str) -> str:
        fen = fen.split()[0]
        rows = fen.split("/")
        rows.reverse()
        for i in range(8):
            rows[i] = rows[i].swapcase()
        return "/".join(rows)


class ChessAITorchOpponent(BaseOpponent):
    """PyTorch port of chess-ai (from.h5 + to.h5) — runs on MPS/GPU via _ChessAINet."""

    def __init__(self, model_dir: str, device: torch.device) -> None:
        self.device = device
        self.from_net = _ChessAINet().to(device)
        self.to_net   = _ChessAINet().to(device)
        _load_chess_ai_weights(self.from_net, os.path.join(model_dir, "from.h5"))
        _load_chess_ai_weights(self.to_net,   os.path.join(model_dir, "to.h5"))
        self.from_net.eval()
        self.to_net.eval()

    def select_move(self, board: chess.Board) -> chess.Move:
        fen = board.fen()
        flip = board.turn == chess.BLACK
        if flip:
            fen = ChessAIKerasOpponent._invert_fen(fen)

        arr = ChessAIKerasOpponent._fen_to_matrix(fen)   # (8, 8, 12) NHWC float64
        # NHWC (8,8,12) → NCHW (1,12,8,8)
        x = torch.from_numpy(
            np.transpose(arr, (2, 0, 1))[np.newaxis].astype(np.float32)
        ).to(self.device)

        with torch.no_grad():
            from_out = self.from_net(x)   # (1, 1, 8, 8)
            to_out   = self.to_net(x)     # (1, 1, 8, 8)

        from_matrix = from_out[0, 0].cpu().numpy()   # (8, 8)
        to_matrix   = to_out[0, 0].cpu().numpy()     # (8, 8)

        if flip:
            from_matrix = np.flip(from_matrix, axis=0)
            to_matrix   = np.flip(to_matrix,   axis=0)

        legal_moves = list(board.legal_moves)
        scores = []
        for move in legal_moves:
            uci = move.uci()[:4]
            from_row, from_col, to_row, to_col = ChessAIKerasOpponent._uci_to_row_col(uci)
            scores.append(from_matrix[from_row][from_col] * to_matrix[to_row][to_col])

        scores = np.maximum(scores, 0)
        total = scores.sum()
        probs = scores / total if total > 0 else np.ones(len(scores)) / len(scores)
        return legal_moves[np.random.choice(len(legal_moves), p=probs)]


class ChessDeepRLOpponent(BaseOpponent):
    def __init__(self, model_path: str, code_dir: str) -> None:
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        try:
            from tf_keras.models import load_model
        except ImportError:
            from tensorflow.keras.models import load_model
        from chessEnv import ChessEnv
        from mapper import Mapping

        import config as deep_config

        self.config = deep_config
        self.chess_env = ChessEnv
        self.mapping = Mapping

        # Prefer the NHWC-converted model (required on CPU / Apple Silicon).
        # The NHWC model expects input transposed from (B,8,8,19) to (B,8,19,8).
        nhwc_path = os.path.splitext(model_path)[0] + "_nhwc.h5"
        if os.path.exists(nhwc_path):
            self.model = load_model(nhwc_path, compile=False)
            self._nhwc = True
        else:
            self.model = load_model(model_path, compile=False)
            self._nhwc = False

    def select_move(self, board: chess.Board) -> chess.Move:
        input_state = self.chess_env.state_to_input(board.fen())
        if self._nhwc:
            # Transpose from channels_first (B,C,H,W)=(B,8,8,19) to channels_last (B,H,W,C)=(B,8,19,8)
            input_state = np.transpose(input_state, (0, 2, 3, 1))
        policy, _ = self.model.predict(input_state, verbose=0)
        probabilities = policy[0].reshape(self.config.amount_of_planes, self.config.n, self.config.n)

        legal_moves = list(board.legal_moves)
        scores = np.array([
            probabilities[p][c][r]
            for m in legal_moves
            for p, r, c in [self._move_to_plane_index(m, board)]
        ], dtype=np.float64)
        scores = np.maximum(scores, 0)
        total = scores.sum()
        probs = scores / total if total > 0 else np.ones(len(scores)) / len(scores)
        return legal_moves[np.random.choice(len(legal_moves), p=probs)]

    def _move_to_plane_index(self, move: chess.Move, board: chess.Board) -> Tuple[int, int, int]:
        from_square = move.from_square
        to_square = move.to_square
        piece = board.piece_at(from_square)
        if piece is None:
            raise ValueError(f"No piece at {from_square}")

        if move.promotion and move.promotion != chess.QUEEN:
            piece_type, direction = self.mapping.get_underpromotion_move(
                move.promotion, from_square, to_square
            )
            plane_index = self.mapping.mapper[piece_type][1 - direction]
        else:
            if piece.piece_type == chess.KNIGHT:
                direction = self.mapping.get_knight_move(from_square, to_square)
                plane_index = self.mapping.mapper[direction]
            else:
                direction, distance = self.mapping.get_queenlike_move(from_square, to_square)
                plane_index = self.mapping.mapper[direction][abs(distance) - 1]

        row = from_square % 8
        col = 7 - (from_square // 8)
        return plane_index, row, col


class ChessDeepRLTorchOpponent(BaseOpponent):
    """PyTorch port of chess-deep-rl — runs on MPS/GPU via _DeepRLNet."""

    def __init__(self, model_path: str, code_dir: str, device: torch.device) -> None:
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)

        from chessEnv import ChessEnv
        from mapper import Mapping
        import config as deep_config

        self.config = deep_config
        self.chess_env = ChessEnv
        self.mapping = Mapping
        self.device = device

        self.net = _DeepRLNet().to(device)
        _load_deep_rl_weights(self.net, model_path)
        self.net.eval()

    def select_move(self, board: chess.Board) -> chess.Move:
        # state_to_input returns (1, 8, 8, 19); in NCHW this is (B, C=8, H=8, W=19) — correct for PyTorch
        input_state = self.chess_env.state_to_input(board.fen()).astype(np.float32)
        x = torch.from_numpy(input_state).to(self.device)

        with torch.no_grad():
            policy, _ = self.net(x)

        probs_np = policy[0].cpu().numpy()  # (4672,)
        probabilities = probs_np.reshape(self.config.amount_of_planes, self.config.n, self.config.n)

        legal_moves = list(board.legal_moves)
        scores = np.array([
            probabilities[p][c][r]
            for m in legal_moves
            for p, r, c in [self._move_to_plane_index(m, board)]
        ], dtype=np.float64)
        scores = np.maximum(scores, 0)
        total = scores.sum()
        move_probs = scores / total if total > 0 else np.ones(len(scores)) / len(scores)
        return legal_moves[np.random.choice(len(legal_moves), p=move_probs)]

    def _move_to_plane_index(self, move: chess.Move, board: chess.Board) -> Tuple[int, int, int]:
        from_square = move.from_square
        to_square = move.to_square
        piece = board.piece_at(from_square)
        if piece is None:
            raise ValueError(f"No piece at {from_square}")

        if move.promotion and move.promotion != chess.QUEEN:
            piece_type, direction = self.mapping.get_underpromotion_move(
                move.promotion, from_square, to_square
            )
            plane_index = self.mapping.mapper[piece_type][1 - direction]
        else:
            if piece.piece_type == chess.KNIGHT:
                direction = self.mapping.get_knight_move(from_square, to_square)
                plane_index = self.mapping.mapper[direction]
            else:
                direction, distance = self.mapping.get_queenlike_move(from_square, to_square)
                plane_index = self.mapping.mapper[direction][abs(distance) - 1]

        row = from_square % 8
        col = 7 - (from_square // 8)
        return plane_index, row, col


class StockfishOpponent(BaseOpponent):
    supports_batch = False

    # Approximate FIDE Elo per Skill Level (Stockfish 16; varies by version/hardware)
    SKILL_ELO: dict = {
        0: 800,  1: 900,  2: 1000, 3: 1100, 4: 1200,
        5: 1300, 6: 1400, 7: 1500, 8: 1600, 9: 1700,
        10: 1800, 11: 1900, 12: 2000, 13: 2100, 14: 2200,
        15: 2400, 16: 2600, 17: 2800, 18: 3000, 19: 3200, 20: 3500,
    }

    def __init__(self, stockfish_path: str, skill_level: int = 5, movetime: float = 0.05) -> None:
        import chess.engine
        self.skill_level = skill_level
        self.movetime = movetime
        self.approx_elo = self.SKILL_ELO.get(skill_level, 1500)
        self._engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self._engine.configure({"Skill Level": skill_level})

    def select_move(self, board: chess.Board) -> chess.Move:
        import chess.engine
        result = self._engine.play(board, chess.engine.Limit(time=self.movetime))
        return result.move

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()


def create_opponent(spec: str, device: torch.device) -> OpponentSpec:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty model spec")

    model_type = None
    model_path = spec
    if ":" in spec:
        prefix, rest = spec.split(":", 1)
        if prefix in {"alpha", "chess_ai", "chess_deep_rl", "deep_pink", "az"}:
            model_type = prefix
            model_path = rest

    if model_type is None:
        model_type = "alpha"

    if model_type == "stockfish":
        # spec: "stockfish:<skill_level>" or "stockfish:<path>:<skill_level>"
        parts = model_path.split(":")
        if len(parts) == 2 and parts[0].isdigit():
            sf_path, skill_level = "stockfish", int(parts[0])
        elif len(parts) == 2:
            sf_path, skill_level = parts[0], int(parts[1])
        else:
            sf_path, skill_level = model_path, 5
        opponent = StockfishOpponent(sf_path, skill_level)
        return OpponentSpec(name=f"stockfish_skill{skill_level}", opponent=opponent)

    if model_type == "az":
        # spec: "az:<path>" or "az:<path>:<n_sims>"
        parts = model_path.rsplit(":", 1)
        if len(parts) == 2 and parts[1].isdigit():
            path, n_sims = parts[0], int(parts[1])
        else:
            path, n_sims = model_path, 400
        name = os.path.splitext(os.path.basename(path))[0]
        return OpponentSpec(name=f"{name}_mcts{n_sims}",
                            opponent=AlphaZeroOpponent(path, device, n_simulations=n_sims))

    if model_type == "alpha":
        name = os.path.splitext(os.path.basename(model_path))[0]
        return OpponentSpec(name=name, opponent=AlphaChessOpponent(model_path, device))

    if model_type == "chess_ai":
        if model_path.endswith(".h5"):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path
        name = f"chess_ai_{os.path.basename(model_dir)}"
        return OpponentSpec(name=name, opponent=ChessAITorchOpponent(model_dir, device))

    if model_type == "chess_deep_rl":
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        code_dir = os.path.join(root_dir, "models", "chess_deep_rl", "chess-deep-rl", "code")
        name = f"chess_deep_rl_{os.path.splitext(os.path.basename(model_path))[0]}"
        return OpponentSpec(name=name, opponent=ChessDeepRLTorchOpponent(model_path, code_dir, device))

    raise NotImplementedError("deep_pink opponents are not wired yet")
