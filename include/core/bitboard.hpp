#ifndef BITBOARD_HPP
#define BITBOARD_HPP

#include "types.hpp"
#include <string>

// ============================================================================
// Bitboard Constants
// ============================================================================

constexpr Bitboard BB_ZERO = 0ULL;
constexpr Bitboard BB_ALL = ~0ULL;

// File bitboards
constexpr Bitboard FILE_A_BB = 0x0101010101010101ULL;
constexpr Bitboard FILE_B_BB = FILE_A_BB << 1;
constexpr Bitboard FILE_C_BB = FILE_A_BB << 2;
constexpr Bitboard FILE_D_BB = FILE_A_BB << 3;
constexpr Bitboard FILE_E_BB = FILE_A_BB << 4;
constexpr Bitboard FILE_F_BB = FILE_A_BB << 5;
constexpr Bitboard FILE_G_BB = FILE_A_BB << 6;
constexpr Bitboard FILE_H_BB = FILE_A_BB << 7;

// Rank bitboards
constexpr Bitboard RANK_1_BB = 0xFFULL;
constexpr Bitboard RANK_2_BB = RANK_1_BB << 8;
constexpr Bitboard RANK_3_BB = RANK_1_BB << 16;
constexpr Bitboard RANK_4_BB = RANK_1_BB << 24;
constexpr Bitboard RANK_5_BB = RANK_1_BB << 32;
constexpr Bitboard RANK_6_BB = RANK_1_BB << 40;
constexpr Bitboard RANK_7_BB = RANK_1_BB << 48;
constexpr Bitboard RANK_8_BB = RANK_1_BB << 56;

// Special bitboards
constexpr Bitboard LIGHT_SQUARES_BB = 0x55AA55AA55AA55AAULL;
constexpr Bitboard DARK_SQUARES_BB = 0xAA55AA55AA55AA55ULL;

// ============================================================================
// Compiler Intrinsics (cross-platform)
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)

// GCC/Clang intrinsics
inline int popcount(Bitboard b) {
    return __builtin_popcountll(b);
}

inline Square lsb(Bitboard b) {
    assert(b);
    return static_cast<Square>(__builtin_ctzll(b));
}

inline Square msb(Bitboard b) {
    assert(b);
    return static_cast<Square>(63 ^ __builtin_clzll(b));
}

#elif defined(_MSC_VER)

// MSVC intrinsics
#include <intrin.h>

inline int popcount(Bitboard b) {
    return static_cast<int>(__popcnt64(b));
}

inline Square lsb(Bitboard b) {
    assert(b);
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return static_cast<Square>(idx);
}

inline Square msb(Bitboard b) {
    assert(b);
    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return static_cast<Square>(idx);
}

#else

// Fallback implementations (slower)
inline int popcount(Bitboard b) {
    int count = 0;
    while (b) {
        count++;
        b &= b - 1; // Clear the least significant bit
    }
    return count;
}

inline Square lsb(Bitboard b) {
    assert(b);
    const int index64[64] = {
        0, 47,  1, 56, 48, 27,  2, 60,
       57, 49, 41, 37, 28, 16,  3, 61,
       54, 58, 35, 52, 50, 42, 21, 44,
       38, 32, 29, 23, 17, 11,  4, 62,
       46, 55, 26, 59, 40, 36, 15, 53,
       34, 51, 20, 43, 31, 22, 10, 45,
       25, 39, 14, 33, 19, 30,  9, 24,
       13, 18,  8, 12,  7,  6,  5, 63
    };
    const Bitboard debruijn64 = 0x03f79d71b4cb0a89ULL;
    return static_cast<Square>(index64[((b ^ (b - 1)) * debruijn64) >> 58]);
}

inline Square msb(Bitboard b) {
    assert(b);
    const int index64[64] = {
        0, 47,  1, 56, 48, 27,  2, 60,
       57, 49, 41, 37, 28, 16,  3, 61,
       54, 58, 35, 52, 50, 42, 21, 44,
       38, 32, 29, 23, 17, 11,  4, 62,
       46, 55, 26, 59, 40, 36, 15, 53,
       34, 51, 20, 43, 31, 22, 10, 45,
       25, 39, 14, 33, 19, 30,  9, 24,
       13, 18,  8, 12,  7,  6,  5, 63
    };
    const Bitboard debruijn64 = 0x03f79d71b4cb0a89ULL;
    b |= b >> 1;
    b |= b >> 2;
    b |= b >> 4;
    b |= b >> 8;
    b |= b >> 16;
    b |= b >> 32;
    return static_cast<Square>(index64[(b * debruijn64) >> 58]);
}

#endif

// ============================================================================
// Bitboard Manipulation
// ============================================================================

// Extract and clear the least significant bit
inline Square pop_lsb(Bitboard& b) {
    assert(b);
    const Square s = lsb(b);
    b &= b - 1;
    return s;
}

// Set a bit at a given square
inline constexpr Bitboard square_bb(Square s) {
    return 1ULL << s;
}

// Test if a bit is set
inline constexpr bool test_bit(Bitboard b, Square s) {
    return b & square_bb(s);
}

// Set a bit
inline constexpr Bitboard set_bit(Bitboard b, Square s) {
    return b | square_bb(s);
}

// Clear a bit
inline constexpr Bitboard clear_bit(Bitboard b, Square s) {
    return b & ~square_bb(s);
}

// Get file bitboard
inline constexpr Bitboard file_bb(File f) {
    return FILE_A_BB << f;
}

inline constexpr Bitboard file_bb(Square s) {
    return file_bb(file_of(s));
}

// Get rank bitboard
inline constexpr Bitboard rank_bb(Rank r) {
    return RANK_1_BB << (8 * r);
}

inline constexpr Bitboard rank_bb(Square s) {
    return rank_bb(rank_of(s));
}

// ============================================================================
// Bitboard Shifts
// ============================================================================

template<Direction D>
inline constexpr Bitboard shift(Bitboard b) {
    if constexpr (D == NORTH)      return b << 8;
    if constexpr (D == SOUTH)      return b >> 8;
    if constexpr (D == EAST)       return (b & ~FILE_H_BB) << 1;
    if constexpr (D == WEST)       return (b & ~FILE_A_BB) >> 1;
    if constexpr (D == NORTH_EAST) return (b & ~FILE_H_BB) << 9;
    if constexpr (D == NORTH_WEST) return (b & ~FILE_A_BB) << 7;
    if constexpr (D == SOUTH_EAST) return (b & ~FILE_H_BB) >> 7;
    if constexpr (D == SOUTH_WEST) return (b & ~FILE_A_BB) >> 9;
    return BB_ZERO;
}

// Pawn-specific shifts
inline constexpr Bitboard pawn_push_bb(Bitboard b, Color c) {
    return c == WHITE ? shift<NORTH>(b) : shift<SOUTH>(b);
}

inline constexpr Bitboard pawn_double_push_bb(Bitboard b, Color c) {
    Bitboard single = pawn_push_bb(b, c);
    return pawn_push_bb(single, c);
}

inline constexpr Bitboard pawn_attacks_bb(Square s, Color c) {
    Bitboard bb = square_bb(s);
    return c == WHITE ? shift<NORTH_WEST>(bb) | shift<NORTH_EAST>(bb)
                      : shift<SOUTH_WEST>(bb) | shift<SOUTH_EAST>(bb);
}

// ============================================================================
// Bitboard Utilities
// ============================================================================

// Check if more than one bit is set
inline constexpr bool more_than_one(Bitboard b) {
    return b & (b - 1);
}

// Check if exactly one bit is set
inline constexpr bool is_single_bit(Bitboard b) {
    return b && !more_than_one(b);
}

// Get the distance between two squares
inline constexpr int distance(Square a, Square b) {
    int file_dist = std::abs(file_of(a) - file_of(b));
    int rank_dist = std::abs(rank_of(a) - rank_of(b));
    return std::max(file_dist, rank_dist);
}

// Check if squares are aligned (same rank, file, or diagonal)
inline constexpr bool aligned(Square a, Square b, Square c) {
    return (file_of(a) == file_of(b) && file_of(b) == file_of(c)) ||
           (rank_of(a) == rank_of(b) && rank_of(b) == rank_of(c)) ||
           ((file_of(a) - file_of(b)) == (rank_of(a) - rank_of(b)) &&
            (file_of(b) - file_of(c)) == (rank_of(b) - rank_of(c)));
}

// ============================================================================
// Bitboard Visualization (Debug)
// ============================================================================

inline std::string bitboard_to_string(Bitboard b) {
    std::string s = "\n +---+---+---+---+---+---+---+---+\n";
    
    for (Rank r = RANK_8; r >= RANK_1; --r) {
        s += " |";
        for (File f = FILE_A; f <= FILE_H; ++f) {
            Square sq = make_square(f, r);
            s += test_bit(b, sq) ? " X |" : "   |";
        }
        s += " " + std::to_string(r + 1) + "\n";
        s += " +---+---+---+---+---+---+---+---+\n";
    }
    s += "   a   b   c   d   e   f   g   h\n";
    
    return s;
}

// ============================================================================
// Global Arrays (Initialized at startup)
// ============================================================================

// These will be initialized in bitboard.cpp
extern Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
extern Bitboard LineBB[SQUARE_NB][SQUARE_NB];
extern Bitboard PseudoAttacks[PIECE_TYPE_NB][SQUARE_NB];
extern Bitboard PawnAttacks[COLOR_NB][SQUARE_NB];

// Initialize all bitboard tables (call once at startup)
void init_bitboards();

// Precomputed attack tables (to be filled by init_bitboards)
inline Bitboard knight_attacks_bb(Square s) {
    return PseudoAttacks[KNIGHT][s];
}

inline Bitboard king_attacks_bb(Square s) {
    return PseudoAttacks[KING][s];
}

inline Bitboard pawn_attacks_bb(Square s, Color c) {
    return PawnAttacks[c][s];
}

// Returns squares between a and b (exclusive)
inline Bitboard between_bb(Square a, Square b) {
    return BetweenBB[a][b];
}

// Returns line through a and b (inclusive)
inline Bitboard line_bb(Square a, Square b) {
    return LineBB[a][b];
}

#endif // BITBOARD_HPP
