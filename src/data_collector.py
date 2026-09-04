import requests
import os 
import time


class DataCollector:
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        self.base_url = "https://lichess.org/api/games/user/"
        self.targets = [
            'DrNykterstein',   # Magnus Carlsen
            'Dr_Tiger',        # Hikaru Nakamura
            'Alireza2003',     # Alireza Firouzja
            'FabianoCaruanaa', # Fabiano Caruana (Verified with double 'a')
            'STL_Aronian',     # Levon Aronian
            'RebeccaHarris',   # Daniel Naroditsky
            'Penguingim1',     # Andrew Tang
            'Nihalsarin2004',  # Nihal Sarin
            'Zhigalko_Sergei', # Sergei Zhigalko
            'Night-King96'     # Oleksandr Bortnyk
        ]


    def download_games(self, max_games_per_player=500):
        """ Downloads games for target players and saves them as PGN files. """
        os.makedirs(self.output_dir, exist_ok=True)

        output_file = os.path.join(self.output_dir, 'GM_games_large.pgn')
        success_player_fetches = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for player in self.targets:
                print(f"Fetching games for {player}...")

                try:
                    # Params: 
                    # max: number of games
                    # perfType: filter for serious formats (exclude bullet/ultra-bullet if you want easier logic)
                    # tags: include PGN tags (True by default)
                    params = {
                        'max': max_games_per_player,
                        'perfType': 'blitz,rapid,classical',
                        'tags': 'true'
                    }

                    # Use stream=True to handle large responses
                    url = self.base_url + player
                    token = os.environ.get('LICHESS_TOKEN', '')
                    headers = {'Accept': 'application/x-chess-pgn'}
                    if token:
                        headers['Authorization'] = f'Bearer {token}'
                    response = requests.get(url, params=params, headers=headers, stream=True)
                    
                    if response.status_code == 200:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk.decode('utf-8'))
                        
                        # Add spacing between different players' games
                        f.write("\n\n")

                        success_player_fetches += 1
                        print(f"✅ Successfully fetched games for {player}")
                    elif response.status_code == 429:
                        print("Rate limit reached. Waiting 60 seconds...")
                        time.sleep(60)
                    else:
                        print(f"❌ Failed to fetch games for {player}. Status code: {response.status_code}")
                        
                except Exception as e:
                    print(f"❌ Error fetching games for {player}: {e}")
                    
                time.sleep(1)  # Be polite and avoid hitting rate limits

        print(f"✅ Successfully fetched games for {success_player_fetches}/{len(self.targets)} players")
        print(f"All games saved to {output_file}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Download GM games from Lichess")
    p.add_argument("--max_games", type=int, default=2000, help="Max games per player")
    p.add_argument("--output_dir", default="data", help="Output directory for PGN")
    args = p.parse_args()
    collector = DataCollector(output_dir=args.output_dir)
    collector.download_games(max_games_per_player=args.max_games)