import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # Optional: for progress bar
import time # To add timestamp to filename

# --- Constants ---
GRID_SIZE = 10
NUM_SQUARES = GRID_SIZE * GRID_SIZE
# *** CHANGED: Reduced number of simulations ***
NUM_SIMULATIONS = 200_000
CONTINUE_PROBABILITY = 0.5

# --- Helper Functions (Unchanged) ---

def get_neighbors(r, c, grid_shape):
    """Gets valid neighbor coordinates (up, down, left, right) for a square."""
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
            neighbors.append((nr, nc))
    return neighbors

def get_target_neighbors(grid, player_id):
    """Finds all valid neighboring squares owned by opponents."""
    player_squares = np.argwhere(grid == player_id)
    potential_targets = set()

    for r, c in player_squares:
        neighbors = get_neighbors(r, c, grid.shape)
        for nr, nc in neighbors:
            # Check if the neighbor is within grid bounds (already handled by get_neighbors)
            # and belongs to a different player
            if grid[nr, nc] != player_id:
                potential_targets.add((nr, nc))

    return list(potential_targets)

# --- Simulation Function (Modified Challenger Selection) ---

def simulate_game():
    """Simulates a single game of 'The Floor' and returns the initial winner ID."""

    # 1. Initialize Grid: Assign unique initial IDs (0-99)
    grid = np.arange(NUM_SQUARES).reshape((GRID_SIZE, GRID_SIZE))
    # active_players set is implicitly managed by checking unique values in the grid

    while True: # Loop until only one player remains
        # *** MODIFIED: Challenger Selection Logic ***
        
        # Get current players and their territory sizes
        active_ids, counts = np.unique(grid, return_counts=True)

        # Check if the game is over
        if len(active_ids) <= 1:
            break # Exit main game loop if 0 or 1 player left

        single_square_players = []
        multi_square_players = []

        for player_id, count in zip(active_ids, counts):
            if count == 1:
                single_square_players.append(player_id)
            else:
                multi_square_players.append(player_id)

        # Prioritize single-square players if they exist
        if single_square_players:
            current_challenger_id = random.choice(single_square_players)
        else:
            # Should only happen if all remaining players have >1 square
            if not multi_square_players: 
                 # This case should ideally not be reached if len(active_ids) > 1
                 # Could happen in rare edge cases or if logic error elsewhere
                 print("Warning: No multi-square players found when expected.") 
                 break # Safety break
            current_challenger_id = random.choice(multi_square_players)
        # *** End of Modified Challenger Selection ***

        # Keep track of the player whose turn it is (can change if they lose)
        active_player_id_in_turn = current_challenger_id

        while True: # Loop for continuation within a turn

            # 3. Find potential targets for the active player
            target_squares = get_target_neighbors(grid, active_player_id_in_turn)

            if not target_squares:
                # No valid targets adjacent to the player's territory
                break # End this player's turn

            # 4. Randomly choose an adjoining opponent square
            target_r, target_c = random.choice(target_squares)
            defender_id = grid[target_r, target_c]

            # 5. Simulate Battle (50/50 odds) - Winner takes ALL loser's squares
            if random.random() < 0.5: # Attacker (active_player_id_in_turn) wins
                winner_id = active_player_id_in_turn
                loser_id = defender_id
                grid[grid == loser_id] = winner_id # Winner takes all loser territory

                # Check if the game is now over
                if len(np.unique(grid)) <= 1:
                     break # Exit continuation loop

                # 6. Decide whether to continue (50% chance)
                if random.random() >= CONTINUE_PROBABILITY:
                    break # Stop challenging this turn
                # else: continue the inner 'while True' loop

            else: # Defender wins
                winner_id = defender_id
                loser_id = active_player_id_in_turn
                grid[grid == loser_id] = winner_id # Winner takes all loser territory
                # The defender becomes the active player for potential continuation
                active_player_id_in_turn = winner_id

                # Check if the game is now over
                if len(np.unique(grid)) <= 1:
                     break # Exit continuation loop

                # 6. Decide whether to continue (50% chance)
                # The defender who just won now decides
                if random.random() >= CONTINUE_PROBABILITY:
                    break # Stop challenging this turn
                # else: continue the inner 'while True' loop

        # Check if the inner loop break was due to game ending
        if len(np.unique(grid)) <= 1:
            break # Exit main game loop

    # Game over: Find the sole remaining player ID
    final_winner_id_on_grid = np.unique(grid)[0]

    # The final ID *is* the initial ID we care about
    return final_winner_id_on_grid


# --- Main Execution (Unchanged structure, different N) ---

print(f"Starting {NUM_SIMULATIONS} simulations with new rules...")

win_counts = np.zeros(NUM_SQUARES, dtype=int)

# Use tqdm for a progress bar
for i in tqdm(range(NUM_SIMULATIONS), desc="Simulating Games"):
    winner_initial_id = simulate_game()
    # Basic check: ensure winner_initial_id is within expected range
    if 0 <= winner_initial_id < NUM_SQUARES:
        win_counts[winner_initial_id] += 1
    else:
        print(f"Warning: Invalid winner ID {winner_initial_id} in simulation {i+1}")


print("Simulations complete.")

# Calculate probabilities
# Avoid division by zero if NUM_SIMULATIONS was 0 for some reason
if NUM_SIMULATIONS > 0:
    win_probabilities = win_counts / NUM_SIMULATIONS
else:
    win_probabilities = np.zeros(NUM_SQUARES)


# Reshape probabilities into a 10x10 grid
probability_grid = win_probabilities.reshape((GRID_SIZE, GRID_SIZE))

# --- Visualization (Added timestamp to filename suggestion) ---
print("Generating heatmap...")

# Get current time for unique filename
# Removed this feature as the user just wants to see the map now
# current_timestamp = time.strftime("%Y%m%d-%H%M%S")
# filename = f"floor_heatmap_{NUM_SIMULATIONS}_sims_{current_timestamp}.png"

plt.figure(figsize=(12, 10))
# Format annotation to show fewer decimal places for 50k sims if needed, e.g. ".2%"
sns.heatmap(probability_grid, annot=True, fmt=".3%", cmap="viridis", linewidths=.5, cbar_kws={'label': 'Win Probability'})
plt.title(f'Win Probability Heatmap ({NUM_SIMULATIONS} Sims, Single-Square Priority)\n(Starting Position Advantage)')
plt.xlabel("Column")
plt.ylabel("Row")
# Ensure axes show 0-9 for intuitive grid mapping
plt.xticks(np.arange(GRID_SIZE) + 0.5, np.arange(GRID_SIZE))
plt.yticks(np.arange(GRID_SIZE) + 0.5, np.arange(GRID_SIZE), rotation=0)
plt.tight_layout()

# Optional: Save the figure automatically
# plt.savefig(filename)
# print(f"Heatmap saved as {filename}")

plt.show() # Display the heatmap


# --- Analysis (Unchanged) ---
print("\nAnalysis:")
print(f"Based on {NUM_SIMULATIONS} simulations:")
max_prob = np.max(probability_grid)
min_prob = np.min(probability_grid)
max_indices = np.unravel_index(np.argmax(probability_grid), probability_grid.shape)
min_indices = np.unravel_index(np.argmin(probability_grid), probability_grid.shape)

print(f"- Highest win probability: {max_prob:.3%} (found at starting square(s) like {max_indices})")
print(f"- Lowest win probability:  {min_prob:.3%} (found at starting square(s) like {min_indices})")

# Further Insights (Unchanged logic, potentially different results)
center_mask = np.zeros_like(probability_grid, dtype=bool)
center_mask[2:8, 2:8] = True # Inner 6x6
edge_mask = np.zeros_like(probability_grid, dtype=bool)
edge_mask[0, :] = True; edge_mask[-1, :] = True
edge_mask[:, 0] = True; edge_mask[:, -1] = True
corner_mask = np.zeros_like(probability_grid, dtype=bool)
corner_mask[(0,0,-1,-1),(0,-1,0,-1)] = True
edge_no_corner_mask = edge_mask & ~corner_mask

avg_center_prob = np.mean(probability_grid[center_mask]) if center_mask.any() else 0
avg_edge_prob = np.mean(probability_grid[edge_no_corner_mask]) if edge_no_corner_mask.any() else 0
avg_corner_prob = np.mean(probability_grid[corner_mask]) if corner_mask.any() else 0


print(f"- Average win probability for 'center' squares (approx): {avg_center_prob:.3%}")
print(f"- Average win probability for 'edge' (not corner) squares: {avg_edge_prob:.3%}")
print(f"- Average win probability for 'corner' squares: {avg_corner_prob:.3%}")

if avg_center_prob > avg_edge_prob and avg_center_prob > avg_corner_prob:
    print("- Central squares appear to be generally more advantageous.")
elif avg_corner_prob < avg_edge_prob and avg_corner_prob < avg_center_prob:
     print("- Corner squares appear to be generally less advantageous.")
