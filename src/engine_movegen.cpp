#include "../include/engine_movegen.hpp"

// ============================================================================
// Main Move Generation Entry Point
// ============================================================================

void generate_moves(const Position& pos, MoveList& move_list) {
    move_list.clear();
    
    generate_pawn_moves(pos, move_list);
    generate_knight_moves(pos, move_list);
    generate_bishop_moves(pos, move_list);
    generate_rook_moves(pos, move_list);
    generate_queen_moves(pos, move_list);
    generate_king_moves(pos, move_list);
    generate_castling_moves(pos, move_list);
}

// ============================================================================
// Pawn Move Generation
// ============================================================================

void generate_pawn_moves(const Position& pos, MoveList& move_list) {
    Color us = pos.side_to_move();
    Color them = ~us;
    
    Bitboard our_pawns = pos.pieces(us, PAWN);
    Bitboard occupied = pos.pieces();
    Bitboard enemy = pos.pieces(them);
    Bitboard empty = ~occupied;
    
    // Direction constants
    Direction up = (us == WHITE) ? NORTH : SOUTH;
    Rank promotion_rank = (us == WHITE) ? RANK_8 : RANK_1;
    Rank start_rank = (us == WHITE) ? RANK_2 : RANK_7;
    Rank double_push_rank = (us == WHITE) ? RANK_4 : RANK_5;
    
    // Single pushes (non-promoting)
    Bitboard pawns_not_on_7th = our_pawns & ~rank_bb(us == WHITE ? RANK_7 : RANK_2);
    Bitboard push_targets = shift(pawns_not_on_7th, up) & empty;
    
    while (push_targets) {
        Square to = pop_lsb(push_targets);
        Square from = to - up;
        move_list.add(make_move(from, to));
    }
    
    // Double pushes
    Bitboard pawns_on_start = our_pawns & rank_bb(start_rank);
    Bitboard single_push = shift(pawns_on_start, up) & empty;
    Bitboard double_push_targets = shift(single_push, up) & empty;
    
    while (double_push_targets) {
        Square to = pop_lsb(double_push_targets);
        Square from = to - up - up;
        move_list.add(make_move_flag(from, to, DOUBLE_PAWN_PUSH));
    }
    
    // Promotions (pushes)
    Bitboard pawns_on_7th = our_pawns & rank_bb(us == WHITE ? RANK_7 : RANK_2);
    Bitboard promo_push_targets = shift(pawns_on_7th, up) & empty;
    
    while (promo_push_targets) {
        Square to = pop_lsb(promo_push_targets);
        Square from = to - up;
        add_promotions(move_list, from, to);
    }
    
    // Captures (non-promoting)
    Bitboard left_attack_targets, right_attack_targets;
    
    if (us == WHITE) {
        left_attack_targets = shift<NORTH_WEST>(pawns_not_on_7th) & enemy;
        right_attack_targets = shift<NORTH_EAST>(pawns_not_on_7th) & enemy;
    } else {
        left_attack_targets = shift<SOUTH_WEST>(pawns_not_on_7th) & enemy;
        right_attack_targets = shift<SOUTH_EAST>(pawns_not_on_7th) & enemy;
    }
    
    while (left_attack_targets) {
        Square to = pop_lsb(left_attack_targets);
        Square from = to - (us == WHITE ? NORTH_WEST : SOUTH_WEST);
        move_list.add(make_move(from, to));
    }
    
    while (right_attack_targets) {
        Square to = pop_lsb(right_attack_targets);
        Square from = to - (us == WHITE ? NORTH_EAST : SOUTH_EAST);
        move_list.add(make_move(from, to));
    }
    
    // Promotion captures
    Bitboard promo_left_targets, promo_right_targets;
    
    if (us == WHITE) {
        promo_left_targets = shift<NORTH_WEST>(pawns_on_7th) & enemy;
        promo_right_targets = shift<NORTH_EAST>(pawns_on_7th) & enemy;
    } else {
        promo_left_targets = shift<SOUTH_WEST>(pawns_on_7th) & enemy;
        promo_right_targets = shift<SOUTH_EAST>(pawns_on_7th) & enemy;
    }
    
    while (promo_left_targets) {
        Square to = pop_lsb(promo_left_targets);
        Square from = to - (us == WHITE ? NORTH_WEST : SOUTH_WEST);
        add_promotions(move_list, from, to);
    }
    
    while (promo_right_targets) {
        Square to = pop_lsb(promo_right_targets);
        Square from = to - (us == WHITE ? NORTH_EAST : SOUTH_EAST);
        add_promotions(move_list, from, to);
    }
    
    // En Passant
    if (pos.en_passant_square() != SQ_NONE) {
        Square ep_square = pos.en_passant_square();
        Bitboard ep_pawns = pawn_attacks_bb_table(ep_square, them) & our_pawns;
        
        while (ep_pawns) {
            Square from = pop_lsb(ep_pawns);
            move_list.add(make_move_flag(from, ep_square, EN_PASSANT));
        }
    }
}

// ============================================================================
// Knight Move Generation
// ============================================================================

void generate_knight_moves(const Position& pos, MoveList& move_list) {
    Color us = pos.side_to_move();
    Bitboard our_knights = pos.pieces(us, KNIGHT);
    Bitboard our_pieces = pos.pieces(us);
    Bitboard enemy_pieces = pos.pieces(~us);
    
    while (our_knights) {
        Square from = pop_lsb(our_knights);
        Bitboard attacks = knight_attacks_bb(from);
        
        // Quiet moves
        Bitboard quiets = attacks & ~our_pieces & ~enemy_pieces;
        add_moves(move_list, from, quiets);
        
        // Captures
        Bitboard captures = attacks & enemy_pieces;
        add_moves(move_list, from, captures);
    }
}

// ============================================================================
// Bishop Move Generation
// ============================================================================

void generate_bishop_moves(const Position& pos, MoveList& move_list) {
    Color us = pos.side_to_move();
    Bitboard our_bishops = pos.pieces(us, BISHOP);
    Bitboard our_pieces = pos.pieces(us);
    Bitboard enemy_pieces = pos.pieces(~us);
    Bitboard occupied = pos.pieces();
    
    while (our_bishops) {
        Square from = pop_lsb(our_bishops);
        Bitboard attacks = get_bishop_attacks(from, occupied);
        
        // Quiet moves
        Bitboard quiets = attacks & ~our_pieces & ~enemy_pieces;
        add_moves(move_list, from, quiets);
        
        // Captures
        Bitboard captures = attacks & enemy_pieces;
        add_moves(move_list, from, captures);
    }
}

// ============================================================================
// Rook Move Generation
// ============================================================================

void generate_rook_moves(const Position& pos, MoveList& move_list) {
    Color us = pos.side_to_move();
    Bitboard our_rooks = pos.pieces(us, ROOK);
    Bitboard our_pieces = pos.pieces(us);
    Bitboard enemy_pieces = pos.pieces(~us);
    Bitboard occupied = pos.pieces();
    
    while (our_rooks) {
        Square from = pop_lsb(our_rooks);
        Bitboard attacks = get_rook_attacks(from, occupied);
        
        // Quiet moves
        Bitboard quiets = attacks & ~our_pieces & ~enemy_pieces;
        add_moves(move_list, from, quiets);
        
        // Captures
        Bitboard captures = attacks & enemy_pieces;
        add_moves(move_list, from, captures);
    }
}

// ============================================================================
// Queen Move Generation
// ============================================================================

void generate_queen_moves(const Position& pos, MoveList& move_list) {
    Color us = pos.side_to_move();
    Bitboard our_queens = pos.pieces(us, QUEEN);
    Bitboard our_pieces = pos.pieces(us);
    Bitboard enemy_pieces = pos.pieces(~us);
    Bitboard occupied = pos.pieces();
    
    while (our_queens) {
        Square from = pop_lsb(our_queens);
        Bitboard attacks = get_queen_attacks(from, occupied);
        
        // Quiet moves
        Bitboard quiets = attacks & ~our_pieces & ~enemy_pieces;
        add_moves(move_list, from, quiets);
        
        // Captures
        Bitboard captures = attacks & enemy_pieces;
        add_moves(move_list, from, captures);
    }
}

// ============================================================================
// King Move Generation
// ============================================================================

void generate_king_moves(const Position& pos, MoveList& move_list) {
    Color us = pos.side_to_move();
    Bitboard our_king = pos.pieces(us, KING);
    Bitboard our_pieces = pos.pieces(us);
    Bitboard enemy_pieces = pos.pieces(~us);
    
    if (our_king) {
        Square from = lsb(our_king);
        Bitboard attacks = king_attacks_bb(from);
        
        // Quiet moves
        Bitboard quiets = attacks & ~our_pieces & ~enemy_pieces;
        add_moves(move_list, from, quiets);
        
        // Captures
        Bitboard captures = attacks & enemy_pieces;
        add_moves(move_list, from, captures);
    }
}

// ============================================================================
// Castling Move Generation
// ============================================================================

void generate_castling_moves(const Position& pos, MoveList& move_list) {
    Color us = pos.side_to_move();
    Color them = ~us;
    Bitboard occupied = pos.pieces();
    
    if (us == WHITE) {
        // White Kingside Castling (O-O)
        if (pos.castling_rights() & WHITE_OO) {
            // Path must be clear (F1 and G1)
            if (!(occupied & square_bb(SQ_F1)) && !(occupied & square_bb(SQ_G1))) {
                // King must not be in check, pass through check, or land in check
                // Check E1 (starting), F1 (crossing), G1 (destination)
                if (!pos.is_square_attacked(SQ_E1, them) &&
                    !pos.is_square_attacked(SQ_F1, them) &&
                    !pos.is_square_attacked(SQ_G1, them)) {
                    move_list.add(make_move_flag(SQ_E1, SQ_G1, CASTLING));
                }
            }
        }
        
        // White Queenside Castling (O-O-O)
        if (pos.castling_rights() & WHITE_OOO) {
            // Path must be clear (D1, C1, and B1)
            if (!(occupied & square_bb(SQ_D1)) && 
                !(occupied & square_bb(SQ_C1)) && 
                !(occupied & square_bb(SQ_B1))) {
                // King must not be in check, pass through check, or land in check
                // Check E1 (starting), D1 (crossing), C1 (destination)
                if (!pos.is_square_attacked(SQ_E1, them) &&
                    !pos.is_square_attacked(SQ_D1, them) &&
                    !pos.is_square_attacked(SQ_C1, them)) {
                    move_list.add(make_move_flag(SQ_E1, SQ_C1, CASTLING));
                }
            }
        }
    } else { // BLACK
        // Black Kingside Castling (O-O)
        if (pos.castling_rights() & BLACK_OO) {
            // Path must be clear (F8 and G8)
            if (!(occupied & square_bb(SQ_F8)) && !(occupied & square_bb(SQ_G8))) {
                // King must not be in check, pass through check, or land in check
                // Check E8 (starting), F8 (crossing), G8 (destination)
                if (!pos.is_square_attacked(SQ_E8, them) &&
                    !pos.is_square_attacked(SQ_F8, them) &&
                    !pos.is_square_attacked(SQ_G8, them)) {
                    move_list.add(make_move_flag(SQ_E8, SQ_G8, CASTLING));
                }
            }
        }
        
        // Black Queenside Castling (O-O-O)
        if (pos.castling_rights() & BLACK_OOO) {
            // Path must be clear (D8, C8, and B8)
            if (!(occupied & square_bb(SQ_D8)) && 
                !(occupied & square_bb(SQ_C8)) && 
                !(occupied & square_bb(SQ_B8))) {
                // King must not be in check, pass through check, or land in check
                // Check E8 (starting), D8 (crossing), C8 (destination)
                if (!pos.is_square_attacked(SQ_E8, them) &&
                    !pos.is_square_attacked(SQ_D8, them) &&
                    !pos.is_square_attacked(SQ_C8, them)) {
                    move_list.add(make_move_flag(SQ_E8, SQ_C8, CASTLING));
                }
            }
        }
    }
}
