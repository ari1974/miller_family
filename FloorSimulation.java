import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom; // Efficient random number generation

public class FloorSimulation {

    // --- Constants ---
    static final int GRID_SIZE = 10;
    static final int NUM_SQUARES = GRID_SIZE * GRID_SIZE;
    static final int NUM_SIMULATIONS = 50_000; // Same as the Numba version run
    static final double CONTINUE_PROBABILITY = 0.5;

    // --- Coordinate Helper Class ---
    // Used for storing coordinates and using Sets for uniqueness
    static class Coordinate {
        final int r, c;

        Coordinate(int r, int c) {
            this.r = r;
            this.c = c;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Coordinate that = (Coordinate) o;
            return r == that.r && c == that.c;
        }

        @Override
        public int hashCode() {
            // Simple hash code combining row and column
            return 31 * r + c;
            // Or use: return Objects.hash(r, c); // Requires import java.util.Objects;
        }

        @Override
        public String toString() {
            return "(" + r + "," + c + ")";
        }
    }

    // --- Main Simulation Driver ---
    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        int[] winCounts = new int[NUM_SQUARES];
        int errorCount = 0;

        System.out.println("Starting " + NUM_SIMULATIONS + " Java simulations...");

        // Simulation Loop
        for (int i = 0; i < NUM_SIMULATIONS; i++) {
            int winnerId = simulateGame();
            if (winnerId >= 0 && winnerId < NUM_SQUARES) {
                winCounts[winnerId]++;
            } else {
                errorCount++; // Count simulations returning the error code (-1)
            }

            // Optional: Print progress periodically
            if ((i + 1) % (NUM_SIMULATIONS / 10) == 0 && i > 0) {
                System.out.println("... " + (i + 1) + "/" + NUM_SIMULATIONS + " simulations done.");
            }
        }
        System.out.println("Simulations complete.");
        long endTime = System.currentTimeMillis();
        System.out.printf("Total simulation time: %.2f seconds%n", (endTime - startTime) / 1000.0);

        if (errorCount > 0) {
            System.out.println("Warning: " + errorCount + " simulations ended with an error state (-1).");
        }

        // --- Calculate and Print Probabilities ---
        int validSims = NUM_SIMULATIONS - errorCount;
        if (validSims > 0) {
            double[][] probabilityGrid = new double[GRID_SIZE][GRID_SIZE];
            System.out.println("\nWin Probabilities Grid (Starting Position):");
            System.out.println("--------------------------------------------");
            double maxProb = 0.0;
            double minProb = 1.0;
            Coordinate maxCoord = new Coordinate(-1, -1);
            Coordinate minCoord = new Coordinate(-1, -1);

            for(int i = 0; i < NUM_SQUARES; ++i){
                double prob = (double)winCounts[i] / validSims;
                int r = i / GRID_SIZE;
                int c = i % GRID_SIZE;
                probabilityGrid[r][c] = prob;

                // Track min/max
                if (prob > maxProb) { maxProb = prob; maxCoord = new Coordinate(r, c); }
                if (prob < minProb) { minProb = prob; minCoord = new Coordinate(r, c); }

                // Print grid format
                System.out.printf("| %.3f%% ", prob * 100);
                if ((i + 1) % GRID_SIZE == 0) {
                    System.out.println("|"); // End of row
                }
            }
            System.out.println("--------------------------------------------");
            System.out.printf("Highest win probability: %.3f%% (e.g., at %s)%n", maxProb * 100, maxCoord);
            System.out.printf("Lowest win probability:  %.3f%% (e.g., at %s)%n", minProb * 100, minCoord);

        } else {
             System.out.println("\nNo valid simulations completed to calculate probabilities.");
        }
    }

    // --- Core Game Simulation Logic ---
    static int simulateGame() {
        int[][] grid = new int[GRID_SIZE][GRID_SIZE];
        boolean[] isActive = new boolean[NUM_SQUARES];
        int[] playerCounts = new int[NUM_SQUARES];
        int numActive = NUM_SQUARES;

        // Initialization
        for (int r = 0; r < GRID_SIZE; r++) {
            for (int c = 0; c < GRID_SIZE; c++) {
                int id = r * GRID_SIZE + c;
                grid[r][c] = id;
                isActive[id] = true;
                playerCounts[id] = 1;
            }
        }

        // Use ThreadLocalRandom for better performance in concurrent scenarios (though not needed here)
        // and often preferred over new Random()
        Random rand = ThreadLocalRandom.current();

        // Reusable lists for player selection
        List<Integer> singleSquarePlayers = new ArrayList<>(NUM_SQUARES);
        List<Integer> multiSquarePlayers = new ArrayList<>(NUM_SQUARES);

        // --- Main Game Loop ---
        while (numActive > 1) {
            singleSquarePlayers.clear();
            multiSquarePlayers.clear();

            // --- Challenger Selection ---
            for (int id = 0; id < NUM_SQUARES; id++) {
                if (isActive[id]) {
                    if (playerCounts[id] == 1) {
                        singleSquarePlayers.add(id);
                    } else if (playerCounts[id] > 1) {
                        multiSquarePlayers.add(id);
                    }
                    // Ignore players with count 0
                }
            }

            int currentChallengerId;
            // Prioritize single-square players
            if (!singleSquarePlayers.isEmpty()) {
                currentChallengerId = singleSquarePlayers.get(rand.nextInt(singleSquarePlayers.size()));
            } else if (!multiSquarePlayers.isEmpty()) { // If no single-square, choose from multi-square
                currentChallengerId = multiSquarePlayers.get(rand.nextInt(multiSquarePlayers.size()));
            } else {
                 // Should not happen if numActive > 1 and state is consistent
                 System.err.println("Error: No active players found for selection!");
                 return -1; // Error code
            }

            int activePlayerIdInTurn = currentChallengerId;

            // --- Turn Loop (Challenge Continuation) ---
            while (true) {
                // Find valid opponent neighbors
                List<Coordinate> targets = getTargetNeighbors(grid, activePlayerIdInTurn);

                if (targets.isEmpty()) {
                    break; // End turn if no valid targets
                }

                // Choose a random target
                Coordinate targetCoord = targets.get(rand.nextInt(targets.size()));
                int defenderId = grid[targetCoord.r][targetCoord.c];

                // Defensive check: Is the defender still active?
                if (!isActive[defenderId]) {
                    // Target owner was eliminated very recently (e.g., multi-elimination turn?)
                    break; // End turn to be safe
                }

                // --- Battle ---
                int winnerId, loserId;
                if (rand.nextDouble() < 0.5) { // Attacker (activePlayerIdInTurn) wins
                    winnerId = activePlayerIdInTurn;
                    loserId = defenderId;
                } else { // Defender wins
                    winnerId = defenderId;
                    loserId = activePlayerIdInTurn;
                }

                // --- State Update (if loser was active) ---
                // Check isActive[loserId] defensively before proceeding
                if (isActive[loserId]) {
                     int loserCount = playerCounts[loserId];
                     playerCounts[loserId] = 0; // Update loser count
                     isActive[loserId] = false; // Mark loser inactive
                     numActive--;               // Decrement active count

                     // Update winner's count (check if winner still active)
                     if (isActive[winnerId]) { // If winner wasn't the loser
                         playerCounts[winnerId] += loserCount;
                     } else if (numActive == 0 && winnerId == loserId) {
                         // Special case: Last two players eliminate each other?
                         // This shouldn't happen with current rules, but handle defensively.
                         // Let winner determination handle this state. numActive will be 0.
                     } else if (numActive == 1 && winnerId == loserId) {
                         // Also unlikely: player eliminated themselves but somehow won?
                         // Handled below: if winnerId is the only one left, they win.
                     } else if (numActive == 1) {
                         // Winner is the sole survivor, set count correctly
                         // Check if the winner is indeed the last active one
                          boolean winnerIsLast = false;
                          for(int id=0; id < NUM_SQUARES; ++id) {
                              if (isActive[id] && id == winnerId) { winnerIsLast = true; break;}
                          }
                          if(winnerIsLast) playerCounts[winnerId] = NUM_SQUARES;
                     }
                     // else: Winner was already eliminated somehow? Error state.

                     // --- Grid Update (Manual Loop - like final Numba version) ---
                     for (int r = 0; r < GRID_SIZE; r++) {
                         for (int c = 0; c < GRID_SIZE; c++) {
                             if (grid[r][c] == loserId) {
                                 grid[r][c] = winnerId;
                             }
                         }
                     }

                     // --- Post-Battle Checks ---
                     // Check Game End
                     if (numActive <= 1) {
                         break; // Exit continuation loop (game over)
                     }

                     // Update who continues turn if defender won
                     if (winnerId == defenderId) {
                         activePlayerIdInTurn = winnerId;
                     }

                     // Continuation Check (Winner decides)
                     if (rand.nextDouble() >= CONTINUE_PROBABILITY) {
                         break; // End turn
                     }
                     // else: continue the inner 'while True' loop

                } else {
                    // Loser was already inactive (should be rare), end the turn
                    break;
                }
            } // End Turn Loop (while true)

            // Check Game End after turn finishes
            if (numActive <= 1) {
                break; // Exit main game loop
            }
        } // End Main Game Loop (while numActive > 1)


        // --- Winner Determination ---
        if (numActive == 1) {
            // Find the single remaining active player
            for (int id = 0; id < NUM_SQUARES; ++id) {
                if (isActive[id]) {
                    return id; // Return the winner's ID
                }
            }
            // Should have found one if numActive is 1
            System.err.println("Error: numActive is 1 but no active player found!");
            return -1; // Error code
        } else {
            // Fallback logic if game ended unexpectedly (numActive != 1)
            if (grid.length > 0 && grid[0].length > 0) { // Check grid exists
                int[] fallbackCounts = new int[NUM_SQUARES];
                int maxCount = -1;
                int potentialWinner = -1;
                // Manual count of final grid state
                for(int r = 0; r < GRID_SIZE; ++r) {
                    for(int c = 0; c < GRID_SIZE; ++c) {
                        int id = grid[r][c];
                        // Check bounds before indexing
                        if(id >= 0 && id < NUM_SQUARES) {
                            fallbackCounts[id]++;
                        }
                    }
                }
                // Find ID with max count
                for(int id = 0; id < NUM_SQUARES; ++id) {
                    if(fallbackCounts[id] > maxCount) {
                        maxCount = fallbackCounts[id];
                        potentialWinner = id;
                    }
                }
                if (potentialWinner != -1) {
                     // System.err.println("Warning: Fallback winner determination used. Final numActive=" + numActive);
                }
                 return potentialWinner; // Return ID with max count, or -1 if grid was empty/invalid
            } else {
                System.err.println("Error: Grid seems invalid in fallback.");
                return -1; // Grid error
            }
        }
    } // End simulateGame

    // --- Helper: Find Opponent Neighbors ---
    static List<Coordinate> getTargetNeighbors(int[][] grid, int playerId) {
        Set<Coordinate> potentialTargets = new HashSet<>(); // Use Set for uniqueness
        List<Coordinate> playerSquares = new ArrayList<>(); // Store player square coords

        // 1. Find all squares owned by the player
        for(int r = 0; r < GRID_SIZE; ++r){
            for(int c = 0; c < GRID_SIZE; ++c){
                if(grid[r][c] == playerId){
                    playerSquares.add(new Coordinate(r, c));
                }
            }
        }

        if (playerSquares.isEmpty()) return new ArrayList<>(); // Player already eliminated?

        // 2. Find neighbors for each player square
        int[] dr = {0, 0, 1, -1}; // Directions: Right, Left, Down, Up
        int[] dc = {1, -1, 0, 0};

        for (Coordinate square : playerSquares) {
            for (int i = 0; i < 4; i++) { // Check 4 neighbors
                int nr = square.r + dr[i];
                int nc = square.c + dc[i];

                // Check grid bounds
                if (nr >= 0 && nr < GRID_SIZE && nc >= 0 && nc < GRID_SIZE) {
                    // Check if the neighbor is owned by an opponent
                    if (grid[nr][nc] != playerId) {
                        potentialTargets.add(new Coordinate(nr, nc)); // Add to Set
                    }
                }
            }
        }
       return new ArrayList<>(potentialTargets); // Convert unique targets Set to List
   } // End getTargetNeighbors

} // End Class FloorSimulation
