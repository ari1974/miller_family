import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # Optional: for progress bar
import time

# --- Constants ---
GRID_SIZE = 10
NUM_SQUARES = GRID_SIZE * GRID_SIZE
NUM_SIMULATIONS = 200_000 # Keeping the reduced count
CONTINUE_PROBABILITY = 0.5

# --- Helper Functions (get_neighbors is unchanged) ---

def get_neighbors(r, c, grid_shape):
    """Gets valid neighbor coordinates (up, down, left, right) for a square."""
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
            neighbors.append((nr, nc))
    return neighbors

# get_target_neighbors remains the same for now. Vectorizing it further adds
# complexity, and removing np.unique calls might be sufficient speedup.
def get_target_neighbors(grid, player_id):
    """Finds all valid neighboring squares owned by opponents."""
    player_squares = np.argwhere(grid == player_id)
    potential_targets = set()
    grid_shape = grid.shape

    for r, c in player_squares:
        # Inlined get_neighbors for potential minor speedup (reduces function call overhead)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
                 # Check owner directly
                 if grid[nr, nc] != player_id:
                    potential_targets.add((nr, nc))

    return list(potential_targets)

# --- Simulation Function (Optimized with incremental tracking) ---

def simulate_game_optimized():
    """Simulates a single game faster using incremental player/count tracking."""

    # 1. Initialize Grid
    grid = np.arange(NUM_SQUARES).reshape((GRID_SIZE, GRID_SIZE))

    # *** OPTIMIZATION: Track active players and counts incrementally ***
    active_players = set(range(NUM_SQUARES))
    # Initialize counts: each player starts with 1 square
    player_counts = {player_id: 1 for player_id in range(NUM_SQUARES)}

    # --- Main Game Loop ---
    while len(active_players) > 1:

        # *** OPTIMIZATION: Efficient Challenger Selection using player_counts ***
        single_square_players = []
        multi_square_players = []
        # Iterate through currently active players only
        for player_id in active_players:
            # player_counts.get(player_id, 0) handles potential edge cases if ID somehow not in counts
            count = player_counts.get(player_id, 0)
            if count == 1:
                single_square_players.append(player_id)
            elif count > 1: # Ensure player actually has squares
                multi_square_players.append(player_id)
            # Ignore players with count 0 if any lingered due to error

        if not single_square_players and not multi_square_players:
             # Should not happen if len(active_players) > 1
             print(f"Warning: No players found to select from! Active: {active_players}, Counts: {player_counts}")
             # Attempt to recover state or break
             current_active_in_grid = np.unique(grid)
             if len(current_active_in_grid) <=1: break
             active_players = set(current_active_in_grid)
             # This recovery is complex, safer to just break if state is inconsistent
             break


        # Prioritize single-square players
        if single_square_players:
            current_challenger_id = random.choice(single_square_players)
        else:
            current_challenger_id = random.choice(multi_square_players)
        # *** End of Optimized Challenger Selection ***

        active_player_id_in_turn = current_challenger_id

        # --- Inner Continuation Loop ---
        while True:
            target_squares = get_target_neighbors(grid, active_player_id_in_turn)

            if not target_squares:
                break # End turn

            target_r, target_c = random.choice(target_squares)
            defender_id = grid[target_r, target_c]

            # --- Battle ---
            # Ensure defender is still active (could have been eliminated in a rapid turn?)
            if defender_id not in active_players:
                 # Target square owner already eliminated, try finding another target?
                 # Or just end the turn? Ending turn is safer.
                 break # Target invalid, end turn.

            if random.random() < 0.5: # Attacker wins
                winner_id = active_player_id_in_turn
                loser_id = defender_id
            else: # Defender wins
                winner_id = defender_id
                loser_id = active_player_id_in_turn

            # *** OPTIMIZATION: Update counts and active set incrementally ***
            if loser_id in player_counts: # Check if loser hasn't been eliminated in same turn edge case
                loser_count = player_counts.pop(loser_id) # Remove loser's count entry
                active_players.remove(loser_id) # Remove loser from active set

                # Update winner's count (ensure winner is still in dict)
                if winner_id in player_counts:
                     player_counts[winner_id] += loser_count
                else:
                     # This implies winner was eliminated and then won? Should not happen.
                     # Or winner is the sole survivor.
                     # If winner is now the sole survivor, this update isn't strictly needed
                     # but helps maintain state consistency until loop exit check.
                     if len(active_players) == 1 and winner_id == list(active_players)[0]:
                          player_counts[winner_id] = NUM_SQUARES # Assign all squares count
                     else:
                          print(f"Warning: Winner {winner_id} not found in counts during update.")


                # --- Grid Update (this NumPy operation is already efficient) ---
                grid[grid == loser_id] = winner_id


                # --- Post-Battle State Check ---
                if len(active_players) <= 1:
                    break # Exit continuation loop (game over)

                # If the defender won, they become the active player for continuation check
                if winner_id == defender_id:
                    active_player_id_in_turn = winner_id # Update player for next potential challenge

                # Decide whether the *winner* continues
                if random.random() >= CONTINUE_PROBABILITY:
                    break # Stop challenging this turn
                # else: continue the inner 'while True' loop

            else:
                 # Loser was already eliminated (e.g. rare multi-challenge edge case?)
                 # End the current player's turn to avoid issues.
                 break # End turn


        # --- Check Game End after turn finishes ---
        if len(active_players) <= 1:
            break # Exit main game loop

    # --- Determine Winner ---
    if not active_players:
         # This indicates an error state, perhaps the last two eliminated each other?
         print("Warning: No active players left at the end!")
         # Find who is actually on the grid as a fallback
         final_winner_id_on_grid = np.unique(grid)[0] # Hope there's one left
         # Handle case where grid might be empty or inconsistent
         if final_winner_id_on_grid is None: return -1 # Return invalid ID
    else:
        final_winner_id_on_grid = list(active_players)[0]

    return final_winner_id_on_grid


# --- Main Execution (Uses the optimized function) ---

print(f"Starting {NUM_SIMULATIONS} OPTIMIZED simulations...")
start_time = time.time()

win_counts = np.zeros(NUM_SQUARES, dtype=int)

for i in tqdm(range(NUM_SIMULATIONS), desc="Simulating Games"):
    # *** Use the optimized function ***
    winner_initial_id = simulate_game_optimized()
    if 0 <= winner_initial_id < NUM_SQUARES:
        win_counts[winner_initial_id] += 1
    else:
         print(f"Warning: Invalid winner ID {winner_initial_id} received from simulation {i+1}")


print("Simulations complete.")
end_time = time.time()
print(f"Total simulation time: {end_time - start_time:.2f} seconds")


# Calculate probabilities (rest of the code is the same)
if NUM_SIMULATIONS > 0:
    win_probabilities = win_counts / NUM_SIMULATIONS
else:
    win_probabilities = np.zeros(NUM_SQUARES)

probability_grid = win_probabilities.reshape((GRID_SIZE, GRID_SIZE))

# --- Visualization ---
print("Generating heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(probability_grid, annot=True, fmt=".3%", cmap="viridis", linewidths=.5, cbar_kws={'label': 'Win Probability'})
plt.title(f'Win Probability Heatmap ({NUM_SIMULATIONS} Sims, Single-Square Priority, Optimized)\n(Starting Position Advantage)')
plt.xlabel("Column")
plt.ylabel("Row")
plt.xticks(np.arange(GRID_SIZE) + 0.5, np.arange(GRID_SIZE))
plt.yticks(np.arange(GRID_SIZE) + 0.5, np.arange(GRID_SIZE), rotation=0)
plt.tight_layout()
plt.show()

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

