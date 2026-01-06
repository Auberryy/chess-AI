#include "../include/engine_perft.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

// ============================================================================
// Perft Implementation
// ============================================================================

uint64_t perft(Position& pos, int depth) {
    if (depth == 0) {
        return 1;
    }
    
    MoveList moves;
    generate_moves(pos, moves);
    
    if (depth == 1) {
        // Leaf node optimization - count legal moves without recursion
        uint64_t count = 0;
        for (int i = 0; i < moves.size(); ++i) {
            StateInfo state;
            if (pos.make_move(moves[i], state)) {
                count++;
                pos.undo_move(moves[i], state);
            }
        }
        return count;
    }
    
    uint64_t nodes = 0;
    
    for (int i = 0; i < moves.size(); ++i) {
        StateInfo state;
        if (pos.make_move(moves[i], state)) {
            nodes += perft(pos, depth - 1);
            pos.undo_move(moves[i], state);
        }
    }
    
    return nodes;
}

// ============================================================================
// Perft Divide (Detailed Breakdown)
// ============================================================================

void perft_divide(Position& pos, int depth) {
    MoveList moves;
    generate_moves(pos, moves);
    
    uint64_t total_nodes = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n";
    
    for (int i = 0; i < moves.size(); ++i) {
        StateInfo state;
        Move m = moves[i];
        
        if (!pos.make_move(m, state)) {
            continue; // Skip illegal moves
        }
        
        uint64_t nodes = (depth > 1) ? perft(pos, depth - 1) : 1;
        total_nodes += nodes;
        
        std::cout << move_to_string(m) << ": " << nodes << "\n";
        
        pos.undo_move(m, state);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double seconds = duration.count() / 1000.0;
    double mnps = (seconds > 0) ? (total_nodes / seconds / 1000000.0) : 0.0;
    
    std::cout << "\nTotal nodes: " << total_nodes << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(3) << seconds << "s\n";
    std::cout << "Speed: " << std::fixed << std::setprecision(2) << mnps << " Mnps\n";
}

// ============================================================================
// Standard Perft Test Suite
// ============================================================================

struct PerftTest {
    std::string fen;
    int depth;
    uint64_t expected_nodes;
};

void run_perft_suite() {
    // Standard perft test positions
    PerftTest tests[] = {
        // Starting position
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281},
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5, 4865609},
        
        // Kiwipete (complex middle game position)
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603},
        
        // Position 3 (tests en passant)
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624},
        
        // Position 4 (tests castling and promotions)
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1, 6},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2, 264},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333},
        
        // Position 5 (tests discovered check)
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1, 44},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2, 1486},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
        
        // Position 6 (tests promotions and discovered check)
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1, 46},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2, 2079},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89890},
    };
    
    int num_tests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    int failed = 0;
    
    std::cout << "\n========================================\n";
    std::cout << "Running Perft Test Suite\n";
    std::cout << "========================================\n";
    
    for (int i = 0; i < num_tests; ++i) {
        const PerftTest& test = tests[i];
        
        Position pos(test.fen);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t result = perft(pos, test.depth);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double seconds = duration.count() / 1000.0;
        
        bool success = (result == test.expected_nodes);
        
        if (success) {
            passed++;
            std::cout << "✓ ";
        } else {
            failed++;
            std::cout << "✗ ";
        }
        
        std::cout << "Depth " << test.depth << ": ";
        std::cout << result;
        
        if (!success) {
            std::cout << " (expected " << test.expected_nodes << ")";
        }
        
        std::cout << " [" << std::fixed << std::setprecision(3) << seconds << "s]\n";
        
        // Print position info only for failed tests
        if (!success) {
            std::cout << "  FEN: " << test.fen << "\n";
        }
    }
    
    std::cout << "\n========================================\n";
    std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "========================================\n\n";
}
