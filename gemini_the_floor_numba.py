# --- START OF SCRIPT ---
import numpy as np
# import random # Replaced by np.random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import numba # Import numba

# --- Constants ---
GRID_SIZE = 10
NUM_SQUARES = GRID_SIZE * GRID_SIZE
NUM_SIMULATIONS = 1_000_000
CONTINUE_PROBABILITY = 0.5
GRID_SHAPE = (GRID_SIZE, GRID_SIZE) # Define shape tuple for Numba

# --- Numba-Optimized Helper Functions ---

@numba.jit(nopython=True)
def get_neighbors_numba(r, c, grid_shape):
    # (Function unchanged)
    neighbors = np.empty((4, 2), dtype=np.int64)
    count = 0
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
            neighbors[count, 0] = nr
            neighbors[count, 1] = nc
            count += 1
    return neighbors[:count]


@numba.jit(nopython=True)
def get_target_neighbors_numba(grid, player_id, grid_shape):
    # (Function unchanged - using simplified unique finding)
    player_squares = np.argwhere(grid == player_id)
    num_player_squares = player_squares.shape[0]
    if num_player_squares == 0:
        return np.empty((0, 2), dtype=np.int64)

    max_potential_targets = min(4 * num_player_squares + 4, 4 * NUM_SQUARES)
    potential_targets_arr = np.empty((max_potential_targets, 2), dtype=np.int64)
    target_count = 0

    for i in range(num_player_squares):
        r, c = player_squares[i, 0], player_squares[i, 1]
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
                if grid[nr, nc] != player_id:
                    if target_count < max_potential_targets:
                         potential_targets_arr[target_count, 0] = nr
                         potential_targets_arr[target_count, 1] = nc
                         target_count += 1

    if target_count == 0:
        return np.empty((0, 2), dtype=np.int64)

    actual_targets = potential_targets_arr[:target_count]

    if actual_targets.shape[0] > 1:
        indices_col1 = np.argsort(actual_targets[:, 1], kind='mergesort')
        targets_sorted_col1 = actual_targets[indices_col1]
        indices_col0 = np.argsort(targets_sorted_col1[:, 0], kind='mergesort')
        sorted_targets = targets_sorted_col1[indices_col0]
        unique_mask = np.ones(sorted_targets.shape[0], dtype=np.bool_)
        for i in range(sorted_targets.shape[0] - 1):
             if sorted_targets[i, 0] == sorted_targets[i+1, 0] and \
                sorted_targets[i, 1] == sorted_targets[i+1, 1]:
                unique_mask[i+1] = False
        return sorted_targets[unique_mask]
    else:
        return actual_targets


# --- Numba-Optimized Simulation Function ---
@numba.jit(nopython=True)
def simulate_game_numba():
    """Simulates a single game faster using Numba (v6 - manual unique fallback)."""
    grid = np.arange(NUM_SQUARES).reshape(GRID_SHAPE)
    is_active = np.ones(NUM_SQUARES, dtype=np.bool_)
    player_counts = np.ones(NUM_SQUARES, dtype=np.int32)
    num_active = NUM_SQUARES

    while num_active > 1:
        active_indices = np.where(is_active)[0]
        if len(active_indices) == 0: break

        counts_of_active = player_counts[active_indices]
        single_square_mask = (counts_of_active == 1)
        multi_square_mask = (counts_of_active > 1)
        single_square_indices = active_indices[single_square_mask]
        multi_square_indices = active_indices[multi_square_mask]

        if len(single_square_indices) > 0:
            challenger_idx_in_active = np.random.randint(0, len(single_square_indices))
            current_challenger_id = single_square_indices[challenger_idx_in_active]
        elif len(multi_square_indices) > 0:
             challenger_idx_in_active = np.random.randint(0, len(multi_square_indices))
             current_challenger_id = multi_square_indices[challenger_idx_in_active]
        else:
            break

        active_player_id_in_turn = current_challenger_id

        while True: # Inner continuation loop
            target_squares_coords = get_target_neighbors_numba(grid, active_player_id_in_turn, GRID_SHAPE)
            if target_squares_coords.shape[0] == 0: break

            target_idx = np.random.randint(0, target_squares_coords.shape[0])
            target_r, target_c = target_squares_coords[target_idx, 0], target_squares_coords[target_idx, 1]
            defender_id = grid[target_r, target_c]

            if not is_active[defender_id]: break

            if np.random.rand() < 0.5: # Attacker wins
                winner_id, loser_id = active_player_id_in_turn, defender_id
            else: # Defender wins
                winner_id, loser_id = defender_id, active_player_id_in_turn

            if is_active[loser_id]:
                 loser_count = player_counts[loser_id]
                 player_counts[loser_id] = 0
                 is_active[loser_id] = False
                 num_active -= 1

                 if is_active[winner_id]:
                     player_counts[winner_id] += loser_count

                 # Manual grid update (using argwhere + loop)
                 loser_coords = np.argwhere(grid == loser_id)
                 for i in range(loser_coords.shape[0]):
                     r = loser_coords[i, 0]
                     c = loser_coords[i, 1]
                     grid[r, c] = winner_id

                 if num_active <= 1: break

                 if winner_id == defender_id: active_player_id_in_turn = winner_id
                 if np.random.rand() >= CONTINUE_PROBABILITY: break
            else:
                 break

        if num_active <= 1: break

    # Determine winner
    if num_active == 1:
        winner_candidates = np.where(is_active)[0]
        if len(winner_candidates) == 1:
             final_winner_id = winner_candidates[0]
        else:
             final_winner_id = -1 # Error state
    else: # Fallback if num_active != 1 at end
        if grid.size > 0:
            # *** MODIFIED FALLBACK LOGIC START: Manual Counting ***
            flat_grid = grid.ravel()
            # Manual counting instead of np.unique(return_counts=True)
            fallback_counts = np.zeros(NUM_SQUARES, dtype=np.int32)
            for player_id in flat_grid:
                # Check bounds defensively before indexing counts array
                if 0 <= player_id < NUM_SQUARES:
                     fallback_counts[player_id] += 1

            # Find the player ID (index) with the maximum count
            if np.sum(fallback_counts) > 0: # Check if any counts were made
                final_winner_id = np.argmax(fallback_counts)
            else:
                final_winner_id = -1 # Grid contained only invalid IDs?
            # *** MODIFIED FALLBACK LOGIC END ***
        else:
             final_winner_id = -1 # Grid was size 0?

    return final_winner_id


# --- Main Execution ---
print("Compiling Numba functions (first run)...")
try:
    compile_start_time = time.time()
    _ = simulate_game_numba()
    compile_end_time = time.time()
    print(f"Compilation complete. (Took {compile_end_time - compile_start_time:.2f} seconds)")

    print(f"Starting {NUM_SIMULATIONS} NUMBA-OPTIMIZED simulations...")
    start_time = time.time()
    win_counts = np.zeros(NUM_SQUARES, dtype=np.int32)
    error_count = 0

    for i in tqdm(range(NUM_SIMULATIONS), desc="Simulating Games"):
        winner_initial_id = simulate_game_numba()
        if 0 <= winner_initial_id < NUM_SQUARES:
            win_counts[winner_initial_id] += 1
        else:
            error_count += 1

    print("Simulations complete.")
    end_time = time.time()
    print(f"Total Numba simulation time: {end_time - start_time:.2f} seconds")
    if error_count > 0:
        print(f"Warning: {error_count} simulations ended with an error state (-1).")

    # --- Plotting and Analysis ---
    if NUM_SIMULATIONS > 0:
        valid_sims = NUM_SIMULATIONS - error_count
        if valid_sims > 0:
             win_probabilities = win_counts / valid_sims
        else:
             win_probabilities = np.zeros(NUM_SQUARES)
    else:
        win_probabilities = np.zeros(NUM_SQUARES)

    probability_grid = win_probabilities.reshape((GRID_SIZE, GRID_SIZE))
    print("Generating heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(probability_grid, annot=True, fmt=".3%", cmap="viridis", linewidths=.5, cbar_kws={'label': 'Win Probability'})
    # Updated title again for clarity
    plt.title(f'Win Probability Heatmap ({NUM_SIMULATIONS} Sims, Single-Sq Prio, Numba Opt v6)\n(Starting Position Advantage)') # v6
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xticks(np.arange(GRID_SIZE) + 0.5, np.arange(GRID_SIZE))
    plt.yticks(np.arange(GRID_SIZE) + 0.5, np.arange(GRID_SIZE), rotation=0)
    plt.tight_layout()
    plt.show()

    print("\nAnalysis:")
    if valid_sims > 0 :
        print(f"Based on {valid_sims} valid simulations:")
        max_prob = np.max(probability_grid)
        min_prob = np.min(probability_grid)
        max_indices = np.unravel_index(np.argmax(probability_grid), probability_grid.shape)
        min_indices = np.unravel_index(np.argmin(probability_grid), probability_grid.shape)
        print(f"- Highest win probability: {max_prob:.3%} (found at starting square(s) like {max_indices})")
        print(f"- Lowest win probability:  {min_prob:.3%} (found at starting square(s) like {min_indices})")
        # ... rest of analysis code ...
    else:
        print("- No valid simulations completed to analyze probabilities.")


except Exception as e:
    print("\n--- An Error Occurred During Execution ---")
    import traceback
    traceback.print_exc()
    print("\nPlease check the error message above.")

# --- END OF SCRIPT ---
