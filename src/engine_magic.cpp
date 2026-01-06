#include "../include/engine_magic.hpp"
#include <cstdlib>
#include <cstring>

// ============================================================================
// Global Magic Tables
// ============================================================================

Magic RookMagics[SQUARE_NB];
Magic BishopMagics[SQUARE_NB];

// Attack tables (allocated at runtime)
namespace {
    Bitboard RookAttackTable[0x19000];  // 102,400 entries (about 800 KB)
    Bitboard BishopAttackTable[0x1480]; // 5,248 entries (about 41 KB)
}

// ============================================================================
// Internal Helper Functions
// ============================================================================

namespace MagicInternal {

// Generate the relevant occupancy mask for a square
// (excludes edge squares since they don't block further attacks)
Bitboard generate_occupancy_mask(Square sq, bool is_rook) {
    Bitboard mask = BB_ZERO;
    int rank = rank_of(sq);
    int file = file_of(sq);
    
    if (is_rook) {
        // Rook: rank and file (excluding edges)
        for (int r = rank + 1; r < 7; r++)
            mask |= square_bb(make_square(static_cast<File>(file), static_cast<Rank>(r)));
        for (int r = rank - 1; r > 0; r--)
            mask |= square_bb(make_square(static_cast<File>(file), static_cast<Rank>(r)));
        for (int f = file + 1; f < 7; f++)
            mask |= square_bb(make_square(static_cast<File>(f), static_cast<Rank>(rank)));
        for (int f = file - 1; f > 0; f--)
            mask |= square_bb(make_square(static_cast<File>(f), static_cast<Rank>(rank)));
    } else {
        // Bishop: diagonals (excluding edges)
        for (int r = rank + 1, f = file + 1; r < 7 && f < 7; r++, f++)
            mask |= square_bb(make_square(static_cast<File>(f), static_cast<Rank>(r)));
        for (int r = rank + 1, f = file - 1; r < 7 && f > 0; r++, f--)
            mask |= square_bb(make_square(static_cast<File>(f), static_cast<Rank>(r)));
        for (int r = rank - 1, f = file + 1; r > 0 && f < 7; r--, f++)
            mask |= square_bb(make_square(static_cast<File>(f), static_cast<Rank>(r)));
        for (int r = rank - 1, f = file - 1; r > 0 && f > 0; r--, f--)
            mask |= square_bb(make_square(static_cast<File>(f), static_cast<Rank>(r)));
    }
    
    return mask;
}

// Generate actual attacks for a square given an occupancy bitboard
Bitboard sliding_attack(Square sq, Bitboard occupied, bool is_rook) {
    Bitboard attacks = BB_ZERO;
    int rank = rank_of(sq);
    int file = file_of(sq);
    
    if (is_rook) {
        // North
        for (int r = rank + 1; r <= 7; r++) {
            Square s = make_square(static_cast<File>(file), static_cast<Rank>(r));
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
        }
        // South
        for (int r = rank - 1; r >= 0; r--) {
            Square s = make_square(static_cast<File>(file), static_cast<Rank>(r));
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
        }
        // East
        for (int f = file + 1; f <= 7; f++) {
            Square s = make_square(static_cast<File>(f), static_cast<Rank>(rank));
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
        }
        // West
        for (int f = file - 1; f >= 0; f--) {
            Square s = make_square(static_cast<File>(f), static_cast<Rank>(rank));
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
        }
    } else {
        // North-East
        for (int r = rank + 1, f = file + 1; r <= 7 && f <= 7; r++, f++) {
            Square s = make_square(static_cast<File>(f), static_cast<Rank>(r));
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
        }
        // North-West
        for (int r = rank + 1, f = file - 1; r <= 7 && f >= 0; r++, f--) {
            Square s = make_square(static_cast<File>(f), static_cast<Rank>(r));
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
        }
        // South-East
        for (int r = rank - 1, f = file + 1; r >= 0 && f <= 7; r--, f++) {
            Square s = make_square(static_cast<File>(f), static_cast<Rank>(r));
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
        }
        // South-West
        for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; r--, f--) {
            Square s = make_square(static_cast<File>(f), static_cast<Rank>(r));
            attacks |= square_bb(s);
            if (occupied & square_bb(s)) break;
        }
    }
    
    return attacks;
}

// Generate a random 64-bit number with few bits set (for magic candidates)
Bitboard random_bitboard_sparse() {
    static uint64_t seed = 1070372;
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed & (seed >> 32) & (seed >> 16);
}

// Set up all occupancy variations for a given mask
void enumerate_occupancies(Bitboard mask, Bitboard occupancies[], int& count) {
    count = 0;
    int bits[16];
    int bit_count = 0;
    
    // Extract bit positions
    Bitboard m = mask;
    while (m) {
        bits[bit_count++] = pop_lsb(m);
    }
    
    // Generate all 2^n combinations
    int n = 1 << bit_count;
    for (int i = 0; i < n; i++) {
        Bitboard occ = BB_ZERO;
        for (int j = 0; j < bit_count; j++) {
            if (i & (1 << j))
                occ |= square_bb(static_cast<Square>(bits[j]));
        }
        occupancies[count++] = occ;
    }
}

// Try to find a magic number for a given square
bool try_magic(Square sq, Magic& magic, bool is_rook, Bitboard occupancies[], Bitboard attacks[], int count) {
    // Clear the attack table for this attempt
    std::memset(magic.attacks, 0, (1ULL << (64 - magic.shift)) * sizeof(Bitboard));
    
    // Try to map all occupancies to their attacks
    for (int i = 0; i < count; i++) {
        unsigned idx = magic.index(occupancies[i]);
        
        Bitboard attack = attacks[i];
        
        // Check for collision
        if (magic.attacks[idx] != BB_ZERO && magic.attacks[idx] != attack)
            return false; // Magic doesn't work
        
        magic.attacks[idx] = attack;
    }
    
    return true; // Magic works!
}

// Find a working magic number for a square
Bitboard find_magic(Square sq, bool is_rook, Bitboard* attack_table) {
    Bitboard mask = generate_occupancy_mask(sq, is_rook);
    int bit_count = popcount(mask);
    int shift = 64 - bit_count;
    
    // Generate all possible occupancy variations
    Bitboard occupancies[4096];
    Bitboard attacks[4096];
    int count;
    enumerate_occupancies(mask, occupancies, count);
    
    // Compute the attack for each occupancy
    for (int i = 0; i < count; i++) {
        attacks[i] = sliding_attack(sq, occupancies[i], is_rook);
    }
    
    // Set up magic structure
    Magic magic;
    magic.attacks = attack_table;
    magic.mask = mask;
    magic.shift = shift;
    
    // Try random magic numbers until we find one that works
    for (int attempt = 0; attempt < 100000000; attempt++) {
        magic.magic = random_bitboard_sparse();
        
        // Quick check: ensure the magic has enough bits
        if (popcount((mask * magic.magic) >> 56) < 6)
            continue;
        
        if (try_magic(sq, magic, is_rook, occupancies, attacks, count)) {
            return magic.magic;
        }
    }
    
    // Should never happen with good RNG
    return 0;
}

} // namespace MagicInternal

// ============================================================================
// Public Initialization Function
// ============================================================================

void init_magics() {
    using namespace MagicInternal;
    
    // Initialize rook magics
    Bitboard* rook_table = RookAttackTable;
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        Bitboard mask = generate_occupancy_mask(sq, true);
        int bit_count = popcount(mask);
        int table_size = 1 << bit_count;
        
        RookMagics[sq].attacks = rook_table;
        RookMagics[sq].mask = mask;
        RookMagics[sq].shift = 64 - bit_count;
        RookMagics[sq].magic = find_magic(sq, true, rook_table);
        
        rook_table += table_size;
    }
    
    // Initialize bishop magics
    Bitboard* bishop_table = BishopAttackTable;
    for (Square sq = SQ_A1; sq <= SQ_H8; ++sq) {
        Bitboard mask = generate_occupancy_mask(sq, false);
        int bit_count = popcount(mask);
        int table_size = 1 << bit_count;
        
        BishopMagics[sq].attacks = bishop_table;
        BishopMagics[sq].mask = mask;
        BishopMagics[sq].shift = 64 - bit_count;
        BishopMagics[sq].magic = find_magic(sq, false, bishop_table);
        
        bishop_table += table_size;
    }
}
