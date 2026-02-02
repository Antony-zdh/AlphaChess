import sys
import os

def count_games(pgn_path):
    count = 0
    if not os.path.exists(pgn_path):
        print(f"File not found: {pgn_path}")
        return

    print(f"Counting games in {pgn_path}...")
    with open(pgn_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Every PGN game starts with detailed tags, usually [Event "..."]
            if line.startswith('[Event "'):
                count += 1
                
    print(f"Total Games Found: {count}")

if __name__ == "__main__":
    # You can change the path if you saved it elsewhere
    pgn_file = os.path.join("data", "raw", "GM_games.pgn")
    count_games(pgn_file)