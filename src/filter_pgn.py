"""Filter a streamed Lichess bulk PGN (stdin) to games where both players are
>= MIN_ELO Lichess rating. Writes filtered PGN to a file. Usage:

    zstd -d -c data/bulk/lichess_YYYY-MM.pgn.zst | python src/filter_pgn.py > data/GM_games_bulk.pgn

Streaming avoids materializing the multi-GB decompressed PGN on disk.
"""
import sys
import chess.pgn

MIN_ELO = 2500


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/GM_games_bulk.pgn"
    max_games = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = no limit
    n = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(sys.stdin)
            if game is None:
                break
            try:
                we = int(game.headers.get("WhiteElo", "0") or "0")
                be = int(game.headers.get("BlackElo", "0") or "0")
            except ValueError:
                skipped += 1
                continue
            if we >= MIN_ELO and be >= MIN_ELO:
                f.write(str(game))
                f.write("\n\n")
                n += 1
                if n % 5000 == 0:
                    print(f"  filtered {n} games...", file=sys.stderr)
                if max_games and n >= max_games:
                    break
            else:
                skipped += 1
    print(f"Done: {n} games written to {out_path} ({skipped} skipped).", file=sys.stderr)


if __name__ == "__main__":
    main()
