#include "../include/engine_bitboard.hpp"
#include <algorithm>

// ============================================================================
// Global Lookup Tables
// ============================================================================

Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];

// ============================================================================
// Helper Functions for Initialization
// ============================================================================

namespace {

// Generate knight attacks for a given square
Bitboard generate_knight_attacks(Square s) {
    Bitboard b = BB_ZERO;
    Bitboard bb = square_bb(s);
    
    // Knight moves: 2 squares in one direction, 1 square perpendicular
    b |= (bb & ~(FILE_G_BB | FILE_H_BB)) << 17;  // 2 up, 1 right
    b |= (bb & ~(FILE_A_BB | FILE_B_BB)) << 15;  // 2 up, 1 left
    b |= (bb & ~(FILE_A_BB | FILE_B_BB)) >> 17;  // 2 down, 1 left
    b |= (bb & ~(FILE_G_BB | FILE_H_BB)) >> 15;  // 2 down, 1 right
    b |= (bb & ~FILE_H_BB) << 10;                // 1 up, 2 right
    b |= (bb & ~FILE_A_BB) << 6;                 // 1 up, 2 left
    b |= (bb & ~FILE_A_BB) >> 10;                // 1 down, 2 left
    b |= (bb & ~FILE_H_BB) >> 6;                 // 1 down, 2 right
    
    return b;
}

// Generate king attacks for a given square
Bitboard generate_king_attacks(Square s) {
    Bitboard b = BB_ZERO;
    Bitboard bb = square_bb(s);
    
    b |= shift<NORTH>(bb);
    b |= shift<SOUTH>(bb);
    b |= shift<EAST>(bb);
    b |= shift<WEST>(bb);
    b |= shift<NORTH_EAST>(bb);
    b |= shift<NORTH_WEST>(bb);
    b |= shift<SOUTH_EAST>(bb);
    b |= shift<SOUTH_WEST>(bb);
    
    return b;
}

// Generate ray attacks in a given direction until hitting the edge
Bitboard generate_ray(Square s, Direction d) {
    Bitboard attacks = BB_ZERO;
    Square sq = s;
    
    while (true) {
        // Check if we'll go off the board
        if (d == NORTH && rank_of(sq) == RANK_8) break;
        if (d == SOUTH && rank_of(sq) == RANK_1) break;
        if (d == EAST && file_of(sq) == FILE_H) break;
        if (d == WEST && file_of(sq) == FILE_A) break;
        if (d == NORTH_EAST && (rank_of(sq) == RANK_8 || file_of(sq) == FILE_H)) break;
        if (d == NORTH_WEST && (rank_of(sq) == RANK_8 || file_of(sq) == FILE_A)) break;
        if (d == SOUTH_EAST && (rank_of(sq) == RANK_1 || file_of(sq) == FILE_H)) break;
        if (d == SOUTH_WEST && (rank_of(sq) == RANK_1 || file_of(sq) == FILE_A)) break;
        
        sq = sq + d;
        attacks |= square_bb(sq);
    }
    
    return attacks;
}

// Check if two squares are on the same line (rank, file, or diagonal)
bool is_aligned(Square s1, Square s2) {
    // Same rank or file
    if (rank_of(s1) == rank_of(s2) || file_of(s1) == file_of(s2))
        return true;
    
    // Same diagonal
    int file_diff = static_cast<int>(file_of(s1)) - static_cast<int>(file_of(s2));
    int rank_diff = static_cast<int>(rank_of(s1)) - static_cast<int>(rank_of(s2));
    
    return std::abs(file_diff) == std::abs(rank_diff);
}

// Get the direction from s1 to s2 (they must be aligned)
Direction get_direction(Square s1, Square s2) {
    int file_diff = static_cast<int>(file_of(s2)) - static_cast<int>(file_of(s1));
    int rank_diff = static_cast<int>(rank_of(s2)) - static_cast<int>(rank_of(s1));
    
    if (file_diff == 0 && rank_diff > 0) return NORTH;
    if (file_diff == 0 && rank_diff < 0) return SOUTH;
    if (rank_diff == 0 && file_diff > 0) return EAST;
    if (rank_diff == 0 && file_diff < 0) return WEST;
    
    int file_sign = file_diff > 0 ? 1 : -1;
    int rank_sign = rank_diff > 0 ? 1 : -1;
    
    if (file_sign > 0 && rank_sign > 0) return NORTH_EAST;
    if (file_sign < 0 && rank_sign > 0) return NORTH_WEST;
    if (file_sign > 0 && rank_sign < 0) return SOUTH_EAST;
    if (file_sign < 0 && rank_sign < 0) return SOUTH_WEST;
    
    return NORTH; // Should never reach here
}

} // anonymous namespace

// ============================================================================
// Public Initialization Function
// ============================================================================

void init_bitboards() {
    // Initialize pawn attacks
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        PawnAttacks[WHITE][s] = pawn_attacks_bb(s, WHITE);
        PawnAttacks[BLACK][s] = pawn_attacks_bb(s, BLACK);
    }
    
    // Initialize knight attacks
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        PseudoAttacks[KNIGHT][s] = generate_knight_attacks(s);
    }
    
    // Initialize king attacks
    for (Square s = SQ_A1; s <= SQ_H8; ++s) {
        PseudoAttacks[KING][s] = generate_king_attacks(s);
    }
    
    // Initialize between and line bitboards
    for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1) {
        for (Square s2 = SQ_A1; s2 <= SQ_H8; ++s2) {
            BetweenBB[s1][s2] = BB_ZERO;
            LineBB[s1][s2] = BB_ZERO;
            
            if (s1 == s2) continue;
            
            if (is_aligned(s1, s2)) {
                Direction d = get_direction(s1, s2);
                
                // Line includes both endpoints and everything beyond
                LineBB[s1][s2] = generate_ray(s1, d) | generate_ray(s1, static_cast<Direction>(-d));
                LineBB[s1][s2] |= square_bb(s1);
                
                // Between is only the squares strictly between s1 and s2
                Square sq = s1 + d;
                while (sq != s2) {
                    BetweenBB[s1][s2] |= square_bb(sq);
                    sq = sq + d;
                }
            }
        }
    }
}
