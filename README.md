This is a RL project that aims to build a chess AI from scratch. 

Currently, the project milestones are designed as follows:
1. The Arena: Set up the chess environment and the basic game loop. Use a random agent to play against the user. ✔
2. Visualization: Set up a visualization tool to display the game board real time. ✔
3. Data Pipeline: Set up a data pipeline to collect game data from web. These data are curated for Supervised Fine Tuning.
4. The Brain: Design a small yet capable Neural Network in PyTorch. We need to make sure the architecture is reasonable and strong enough. 
5. Training Infrastructure: Set up the training infrastructure. At this point, we does not need to support RL yet, but we need to prepare for SFT. 
6. Supervised Fine Tuning: Train the Neural Network with curated data from Data Pipeline.
7. RL Infrastructure: Set up the RL infrastructure. We need to support both on-policy and off-policy algorithms. We have to dedicate the update algorithm. During RL, the AI will play against itself to collect experience.
8. RL Training: Train the Neural Network with RL algorithms.
9. Testing and Evaluation: Test the trained Neural Network with the Arena. A finetuned AI (Standard AI of similar size) from web should be selected as the opponent for our AI to play with. Win rate, loss rate, and draw rate should be recorded.