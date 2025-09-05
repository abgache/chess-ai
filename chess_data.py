import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import numpy as np
import os
import chess.pgn

class data():
    def __init__(self):
        self.pgn_files_list = os.listdir("games") # carlsen.pgn
        self.games_list = []
        for file in self.pgn_files_list:
            pgn = open(f"games/{file}")
            while True:
                self.tmp = chess.pgn.read_game(pgn)
                if self.tmp == None:
                    break
                self.games_list += [self.tmp]
        self.game_count = len(self.games_list)
    def board_to_vector(self, board: chess.Board):
        # Transforme un board en vecteur numpy de taille 64*6.
        mapping = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6
        }
        vec = np.zeros(64, dtype=int)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                value = mapping[piece.piece_type]
                if piece.color == chess.BLACK:
                    value *= -1
                vec[i] = value
        one_hot = np.zeros((64, 6), dtype=int)
        for i, v in enumerate(vec):
            if v != 0:
                idx = abs(v) - 1
                one_hot[i, idx] = 1 if v > 0 else -1
        return one_hot.flatten()
    def format(self, keep_looses=True):
        # Keep looses = True : if u loose the game, the model will just be trainned to play the opponent moves.
        x_data = []
        y_data = []
    
        def square_idx_to_6bits(idx: int):
            """Convertit un indice de case 0..63 en 6 bits: [rank(3 bits), file(3 bits)]."""
            if not (0 <= idx <= 63):
                raise ValueError("square index must be in 0..63")
            rank = idx // 8   # 0..7  (rank 1 -> 0, rank 8 -> 7)
            file = idx % 8    # 0..7  (a -> 0, h -> 7)
            rank_bits = [int(b) for b in format(rank, '03b')]
            file_bits = [int(b) for b in format(file, '03b')]
            return rank_bits + file_bits  # 6 éléments
    
        for game in self.games_list:
            board = game.board()
            for move in game.mainline_moves():
                vec = self.board_to_vector(board)
                x_data.append(vec)
    
                # Encode move en vecteur 12 bits (liste d'int 0/1)
                src6 = square_idx_to_6bits(move.from_square)
                dst6 = square_idx_to_6bits(move.to_square)
                y_vec12 = src6 + dst6
                y_data.append(y_vec12)
    
                board.push(move)
    
        return np.array(x_data), np.array(y_data, dtype=np.int8)
