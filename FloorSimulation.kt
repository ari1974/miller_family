import kotlin.math.max // For finding max simulation count
import kotlin.random.Random
import kotlin.system.exitProcess

// --- Constants ---
const val GRID_SIZE = 10
const val NUM_SQUARES = GRID_SIZE * GRID_SIZE
// REMOVED: NUM_SIMULATIONS constant
const val CONTINUE_PROBABILITY = 0.5

// --- Coordinate Data Class ---
data class Coordinate(val r: Int, val c: Int) {
    // Override toString for cleaner output in analysis print
    override fun toString(): String = "($r, $c)"
}

// --- Main Function (Accepts Comma-Delimited List) ---
fun main(args: Array<String>) {

    // --- Argument Parsing and Validation ---
    if (args.isEmpty()) {
        System.err.println("Error: Missing command-line argument.")
        System.err.println("Usage: java -jar FloorSimulation.jar <comma_separated_snapshot_counts>")
        System.err.println("Example: java -jar FloorSimulation.jar 1000,10000,50000")
        exitProcess(1)
    }

    // Parse comma-separated list, convert to int, filter out errors/non-positive, sort
    val snapshotCounts = args[0].split(',')
        .mapNotNull { it.trim().toIntOrNull() } // Convert to Int?, ignore nulls (parse errors)
        .filter { it > 0 }                     // Keep only positive counts
        .distinct()                            // Remove duplicates
        .sorted()                              // Ensure ascending order

    if (snapshotCounts.isEmpty()) {
        System.err.println("Error: Invalid or no positive integer snapshot counts provided.")
        System.err.println("Example: java -jar FloorSimulation.jar 1000,10000,50000")
        exitProcess(1)
    }

    // Determine the maximum number of simulations to run
    val maxSimulations = snapshotCounts.last() // The last element after sorting is the max
    // --- End Argument Parsing ---


    val startTime = System.currentTimeMillis()
    val winCounts = IntArray(NUM_SQUARES)
    var errorCount = 0
    var nextSnapshotIndex = 0 // Index for the snapshotCounts list

    println("Starting Kotlin simulations, running up to $maxSimulations total.")
    println("Will print snapshots at simulation counts: ${snapshotCounts.joinToString()}")

    // --- Simulation Loop ---
    for (i in 0 until maxSimulations) {
        val currentSimCount = i + 1
        val winnerId = simulateGame() // Simulate one game

        if (winnerId in 0 until NUM_SQUARES) {
            winCounts[winnerId]++
        } else {
            errorCount++
        }

        // --- Snapshot Check ---
        // Check if we've reached the next target snapshot count
        if (nextSnapshotIndex < snapshotCounts.size && currentSimCount == snapshotCounts[nextSnapshotIndex]) {
            val header = "--- Snapshot after $currentSimCount simulations ---"
            printProbabilityGrid(winCounts, currentSimCount, errorCount, header)
            nextSnapshotIndex++ // Move to the next snapshot target
        }
    } // End Simulation Loop

    println("\nFinished $maxSimulations simulations.")
    val endTime = System.currentTimeMillis()
    System.out.printf("Total simulation time: %.2f seconds%n", (endTime - startTime) / 1000.0)

    if (errorCount > 0) {
        println("Warning: $errorCount simulations ended with an error state (-1).")
    }

    // The final state (corresponding to maxSimulations) was already printed
    // when the loop hit the last snapshot count.
} // End main

// --- Helper Function to Print Probability Grid ---
fun printProbabilityGrid(winCounts: IntArray, totalSims: Int, errorCount: Int, header: String) {
    println("\n" + header) // Print the provided header

    val validSims = totalSims - errorCount
    if (validSims > 0) {
        val probabilityGrid = Array(GRID_SIZE) { DoubleArray(GRID_SIZE) }
        val separator = "-".repeat(GRID_SIZE * 9) // Adjust width based on formatting
        println(separator)
        var maxProb = 0.0
        var minProb = 1.0
        var maxCoord = Coordinate(-1, -1)
        var minCoord = Coordinate(-1, -1)

        for (i in 0 until NUM_SQUARES) {
            val prob = winCounts[i].toDouble() / validSims
            val r = i / GRID_SIZE
            val c = i % GRID_SIZE
            probabilityGrid[r][c] = prob

            if (prob > maxProb) { maxProb = prob; maxCoord = Coordinate(r, c) }
            if (prob < minProb) { minProb = prob; minCoord = Coordinate(r, c) }

            System.out.printf("| %.3f%% ", prob * 100)
            if ((i + 1) % GRID_SIZE == 0) {
                println("|")
            }
        }
        println(separator)
        System.out.printf("Highest win probability: %.3f%% (e.g., at %s)%n", maxProb * 100, maxCoord)
        System.out.printf("Lowest win probability:  %.3f%% (e.g., at %s)%n", minProb * 100, minCoord)

    } else {
        println("No valid simulations completed yet to calculate probabilities for this snapshot.")
    }
}


// --- Core Game Simulation Logic ---
// (simulateGame function remains unchanged)
fun simulateGame(): Int {
    val grid = Array(GRID_SIZE) { r -> IntArray(GRID_SIZE) { c -> r * GRID_SIZE + c } }
    val isActive = BooleanArray(NUM_SQUARES) { true }
    val playerCounts = IntArray(NUM_SQUARES) { 1 }
    var numActive = NUM_SQUARES
    val rand = Random.Default
    val singleSquarePlayers = mutableListOf<Int>()
    val multiSquarePlayers = mutableListOf<Int>()

    while (numActive > 1) {
        singleSquarePlayers.clear(); multiSquarePlayers.clear()
        for (id in 0 until NUM_SQUARES) {
            if (isActive[id]) {
                when (playerCounts[id]) {
                    1 -> singleSquarePlayers.add(id)
                    else -> if (playerCounts[id] > 1) multiSquarePlayers.add(id)
                }
            }
        }
        val currentChallengerId = when {
            singleSquarePlayers.isNotEmpty() -> singleSquarePlayers.random(rand)
            multiSquarePlayers.isNotEmpty() -> multiSquarePlayers.random(rand)
            else -> return -1
        }
        var activePlayerIdInTurn = currentChallengerId
        while (true) {
            val targets = getTargetNeighbors(grid, activePlayerIdInTurn)
            if (targets.isEmpty()) break
            val targetCoord = targets.random(rand)
            val defenderId = grid[targetCoord.r][targetCoord.c]
            if (!isActive[defenderId]) break
            val (winnerId, loserId) = if (rand.nextDouble() < 0.5) {
                 activePlayerIdInTurn to defenderId
            } else {
                 defenderId to activePlayerIdInTurn
            }
            if (isActive[loserId]) {
                val loserCount = playerCounts[loserId]
                playerCounts[loserId] = 0; isActive[loserId] = false; numActive--
                if (isActive[winnerId]) {
                    playerCounts[winnerId] += loserCount
                } else if (numActive == 1) {
                     var winnerIsLast = false
                     for(id in 0 until NUM_SQUARES) { if (isActive[id] && id == winnerId) { winnerIsLast = true; break; } }
                     if(winnerIsLast) playerCounts[winnerId] = NUM_SQUARES
                }
                for (r in 0 until GRID_SIZE) {
                    for (c in 0 until GRID_SIZE) {
                        if (grid[r][c] == loserId) grid[r][c] = winnerId
                    }
                }
                if (numActive <= 1) break
                if (winnerId == defenderId) activePlayerIdInTurn = winnerId
                if (rand.nextDouble() >= CONTINUE_PROBABILITY) break
            } else { break }
        }
        if (numActive <= 1) break
    }
    return when (numActive) {
        1 -> { (0 until NUM_SQUARES).firstOrNull { isActive[it] } ?: -1 }
        else -> {
            if (grid.isNotEmpty() && grid[0].isNotEmpty()) {
                val fallbackCounts = IntArray(NUM_SQUARES)
                for (r in 0 until GRID_SIZE) { for (c in 0 until GRID_SIZE) {
                        val id = grid[r][c]; if (id in 0 until NUM_SQUARES) fallbackCounts[id]++ } }
                fallbackCounts.indices.maxByOrNull { fallbackCounts[it] } ?: -1
            } else { -1 }
        }
    }
} // End simulateGame

// --- Helper: Find Opponent Neighbors ---
// (getTargetNeighbors function remains unchanged)
fun getTargetNeighbors(grid: Array<IntArray>, playerId: Int): List<Coordinate> {
    val potentialTargets = mutableSetOf<Coordinate>()
    val playerSquares = mutableListOf<Coordinate>()
    for (r in grid.indices) { for (c in grid[r].indices) {
            if (grid[r][c] == playerId) playerSquares.add(Coordinate(r, c)) } }
    if (playerSquares.isEmpty()) return emptyList()
    val dr = intArrayOf(0, 0, 1, -1); val dc = intArrayOf(1, -1, 0, 0)
    for (square in playerSquares) { for (i in 0..3) {
            val nr = square.r + dr[i]; val nc = square.c + dc[i]
            if (nr in 0 until GRID_SIZE && nc in 0 until GRID_SIZE) {
                if (grid[nr][nc] != playerId) potentialTargets.add(Coordinate(nr, nc)) } } }
    return potentialTargets.toList()
} // End getTargetNeighbors
