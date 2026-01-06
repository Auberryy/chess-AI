#include "../include/engine_position.hpp"
#include "../include/engine_perft.hpp"
#include "../include/engine_bitboard.hpp"
#include "../include/engine_magic.hpp"
#include <iostream>

int main() {
    // Initialize tables
    Zobrist::init();
    init_bitboards();
    init_magics();

    // Quick debug: show per-move breakdown for starting position at depth 1
    Position pos;
    pos.set_startpos();
    perft_divide(pos, 1);

    // Then run the full suite
    run_perft_suite();

    return 0;
}
