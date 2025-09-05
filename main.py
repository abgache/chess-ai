# Chess ai V1
# by abgache hehe
# Trainned on an NVIDIA Zotac GeForce GTX 970 triple fan
# Input neurons : (8*8)*6 -> pour chaque case de l'echequier un vecteur sextidimentionnel 
# 0 = vide | 1 = pion | 2 = cavalier | 3 = fou | 4 = tour | 5 = reine | 6 = roi
# nombre positif : blancs | négatif : noirs 
# 1-0 win 0-1 Loose

import json
import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.models import load_model, Sequential
import numpy as np
import os
from time_log import time_log_module as tlm
import chess
import chess.pgn
from chess_data import data
from tensorflow.keras.optimizers import *

model_path = "chess.keras"
input_neurons = 64*6
output_neurons = 12

chess_pieces = [
    ["♙", "♘", "♗", "♖", "♕", "♔"], # white
    ["♟", "♞", "♝", "♜", "♛", "♚"]  # black
]

if __name__ == "__main__":
    print(f"{tlm()} Starting chess ai...")
    train_state = bool(input("Do you want to train the model or just use it [y/n]?\n(If you does not have any saved model the script will just close itself.)\n>>>"))

    # data things
    if train_state:
        print(f"{tlm()} Loading game data...")
        data = data()
        print(f"{tlm()} Formating data...")
        x, y = data.format(keep_looses=True)
        del data
    
    # Deep Neural caca
    mixed_precision.set_global_policy('mixed_float16')
    if not os.path.exists(model_path):
        print(f"{tlm()} Building the deep neural network...")
        model = Sequential([
            layers.Dense(4096, activation="relu", input_shape=(input_neurons,)),
            layers.Dense(4096, activation="relu"),
            layers.Dense(8192, activation="relu"),
            layers.Dense(1024, activation="relu"),
            layers.Dense(output_neurons, activation="sigmoid", dtype="float16")
        ])
    else:
        print(f"{tlm()} Loading the model...")
        model = load_model(model_path)
    if train_state:
        print(f"{tlm()} Trainning the model...")
        model.compile(optimizer="adam", loss="mse")
        model.summary()
        model.fit(x=x,
                y=y,
                batch_size=32,
                epochs=10)
    model.save(model_path)

    # --- helpers pour encodage 6bits / 12bits ---
    def square_idx_to_6bits(idx: int):
        """0..63 -> [rank(3 bits), file(3 bits)]"""
        if not (0 <= idx <= 63):
            raise ValueError("square index must be in 0..63")
        rank = idx // 8   # 0..7
        file = idx % 8    # 0..7
        bits = format(rank, '03b') + format(file, '03b')
        return np.array([int(b) for b in bits], dtype=np.int8)

    def move_to_vector12_from_indices(from_idx: int, to_idx: int):
        """from_idx, to_idx in 0..63 -> np.array shape (12,) of 0/1"""
        return np.concatenate([square_idx_to_6bits(from_idx), square_idx_to_6bits(to_idx)])

    def move_to_vector12(move: chess.Move):
        return move_to_vector12_from_indices(move.from_square, move.to_square)

    def vector12_to_move(vec12):
        """vec12: array-like length 12 -> chess.Move (from, to)"""
        vec12 = np.asarray(vec12).astype(int)
        if vec12.shape[0] != 12:
            raise ValueError("vec12 must be length 12")
        # src
        src_rank = int(''.join(str(b) for b in vec12[0:3]), 2)
        src_file = int(''.join(str(b) for b in vec12[3:6]), 2)
        dst_rank = int(''.join(str(b) for b in vec12[6:9]), 2)
        dst_file = int(''.join(str(b) for b in vec12[9:12]), 2)
        from_idx = src_rank * 8 + src_file
        to_idx = dst_rank * 8 + dst_file
        return chess.Move(from_idx, to_idx)
    def board_to_vector(board: chess.Board):
        """
        Convertit un chess.Board() en vecteur 64*6.
        Pour chaque case (64) on encode un vecteur 6D :
          0 = pion, 1 = cavalier, 2 = fou, 3 = tour, 4 = reine, 5 = roi
        Positif = blanc, Négatif = noir, 0 = vide
        """
        piece_map = board.piece_map()
        vec = np.zeros((64, 6), dtype=np.int8)

        for idx in range(64):
            piece = piece_map.get(idx)
            if piece is not None:
                p_type = piece.piece_type - 1  # pion=1 → 0, cavalier=2 → 1...
                vec[idx, p_type] = 1 if piece.color == chess.WHITE else -1

        return vec.flatten()  # shape (64*6,)


    def print_board_custom(board):
        """Affiche le board avec tes symboles chess_pieces et une grille alignée"""
        piece_map = board.piece_map()

        print("    a b c d e f g h")
        print("  +----------------+")
        for rank in range(7, -1, -1):
            row = f"{rank+1} |"
            for file in range(8):
                idx = rank * 8 + file
                piece = piece_map.get(idx)
                if piece is None:
                    row += ". "  # vide
                else:
                    color_idx = 0 if piece.color == chess.WHITE else 1
                    symbol = chess_pieces[color_idx][piece.piece_type - 1]
                    # ajout d'un espace pour forcer l'alignement
                    row += f"{symbol} "
            row += "|"
            print(row)
        print("  +----------------+")

    # --- play loop adapté au modèle 12-dim ---
    def play_against_model_12bits(model, threshold=0.5, verbose=True):
        """
        model: ton modèle Keras qui prédit shape (1,12) -> 12 valeurs (probas/logits)
        threshold: cutoff pour convertir probas -> bits (0/1)
        """
        board = chess.Board()
        print(board)

        while not board.is_game_over():
            # 1) coup du joueur
            while True:
                try:
                    user_move = input("Your move (e.g. e2e4): ")
                    move = chess.Move.from_uci(user_move)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move!")
                except Exception:
                    print("Invalid format!")

            if verbose:
                print("\nBoard after your move:")
                print_board_custom(board)

            if board.is_game_over():
                break

            # 2) coup du modèle
            x = board_to_vector(board)                     # shape (8,8,6)
            x_input = np.expand_dims(x, axis=0)            # (1,8,8,6)
            pred = model.predict(x_input, verbose=0)       # expect (1,12) or (12,)
            pred = np.squeeze(pred)
            if pred.shape[0] != 12:
                raise ValueError(f"Model output shape must be (12,) but got {pred.shape}")

            # convertir en bits selon threshold
            pred_bits = (pred >= threshold).astype(np.int8)

            try:
                ai_move = vector12_to_move(pred_bits)
            except Exception:
                ai_move = None

            # si illégal -> choisir le legal move dont le vecteur 12bits minimise la Hamming distance
            if ai_move not in board.legal_moves:
                bad_ai_move = ai_move
                legal_moves = list(board.legal_moves)
                if len(legal_moves) == 0:
                    # no legal moves, should be game over
                    break

                # precompute legal encodings
                legal_vecs = np.array([move_to_vector12(m) for m in legal_moves], dtype=np.int8)  # (L,12)
                # calc distance de Hamming (ou mismatch count)
                # pred_bits broadcasted -> compare
                diffs = np.sum(np.abs(legal_vecs - pred_bits), axis=1)  # (L,)
                best_idx = int(np.argmin(diffs))
                # parfois plusieurs ties -> récupère tous les indices min et pick random
                best_mask = diffs == diffs[best_idx]
                candidates = np.where(best_mask)[0]
                chosen = np.random.choice(candidates)
                ai_move = legal_moves[int(chosen)]

                if verbose:
                    print(f"\nPredicted raw bits invalid. ({bad_ai_move}) Hamming distances (min {diffs[best_idx]}). Selected legal move: {ai_move}")

            board.push(ai_move)

            if verbose:
                print("\nBoard after AI move:")
                print_board_custom(board)

        print("\nGame over:", board.result())
    play_against_model_12bits(model)


