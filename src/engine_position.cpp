#include "../include/engine_position.hpp"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>

// ============================================================================
// Zobrist Hash Initialization
// ============================================================================

namespace Zobrist {
    uint64_t piece_keys[PIECE_NB][SQUARE_NB];
    uint64_t en_passant_keys[FILE_NB];
    uint64_t castling_keys[CASTLING_RIGHT_NB];
    uint64_t side_key;
    
    void init() {
        std::mt19937_64 rng(0x1234567890ABCDEFULL); // Deterministic seed for reproducibility
        std::uniform_int_distribution<uint64_t> dist;
        
        // Initialize piece-square keys
        for (int pc = NO_PIECE; pc < PIECE_NB; ++pc) {
            for (Square sq = SQ_A1; sq < SQUARE_NB; ++sq) {
                piece_keys[pc][sq] = dist(rng);
            }
        }
        
        // Initialize en passant keys
        for (int f = FILE_A; f < FILE_NB; ++f) {
            en_passant_keys[f] = dist(rng);
        }
        
        // Initialize castling keys
        for (int cr = 0; cr < CASTLING_RIGHT_NB; ++cr) {
            castling_keys[cr] = dist(rng);
        }
        
        // Initialize side to move key
        side_key = dist(rng);
    }
}

// ============================================================================
// Position Construction
// ============================================================================

Position::Position() {
    clear();
}

Position::Position(const std::string& fen) {
    set_fen(fen);
}

void Position::clear() {
    // Clear all bitboards
    for (int c = WHITE; c < COLOR_NB; ++c) {
        m_by_color[c] = BB_ZERO;
        for (int pt = NO_PIECE_TYPE; pt < PIECE_TYPE_NB; ++pt) {
            m_pieces[c][pt] = BB_ZERO;
        }
    }
    
    // Clear board
    for (Square sq = SQ_A1; sq < SQUARE_NB; ++sq) {
        m_board[sq] = NO_PIECE;
    }
    
    // Reset state
    m_side_to_move = WHITE;
    m_en_passant = SQ_NONE;
    m_castling_rights = NO_CASTLING;
    m_halfmove_clock = 0;
    m_fullmove_number = 1;
    m_hash = 0;
}

void Position::set_startpos() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

// ============================================================================
// Piece Manipulation
// ============================================================================

void Position::put_piece(Piece pc, Square sq) {
    assert(pc != NO_PIECE);
    assert(sq < SQUARE_NB);
    
    // Remove existing piece if any
    if (m_board[sq] != NO_PIECE) {
        remove_piece(sq);
    }
    
    Color c = color_of(pc);
    PieceType pt = type_of(pc);
    
    // Update bitboards
    m_pieces[c][pt] |= square_bb(sq);
    m_by_color[c] |= square_bb(sq);
    m_board[sq] = pc;
    
    // Update hash
    m_hash ^= Zobrist::piece_keys[pc][sq];
}

void Position::remove_piece(Square sq) {
    assert(sq < SQUARE_NB);
    
    Piece pc = m_board[sq];
    if (pc == NO_PIECE) return;
    
    Color c = color_of(pc);
    PieceType pt = type_of(pc);
    
    // Update bitboards
    m_pieces[c][pt] &= ~square_bb(sq);
    m_by_color[c] &= ~square_bb(sq);
    m_board[sq] = NO_PIECE;
    
    // Update hash
    m_hash ^= Zobrist::piece_keys[pc][sq];
}

// ============================================================================
// FEN Parsing
// ============================================================================

void Position::set_fen(const std::string& fen) {
    clear();

    std::istringstream ss(fen);
    std::string piece_placement;

    // 1) Piece placement
    ss >> piece_placement;
    int rank = 7; // start at rank 8
    int file = 0;

    for (size_t i = 0; i < piece_placement.size(); ++i) {
        char ch = piece_placement[i];
        if (ch == '/') {
            --rank;
            file = 0;
            continue;
        }

        if (ch >= '1' && ch <= '8') {
            file += ch - '0';
            continue;
        }

        Color c = (ch >= 'a') ? BLACK : WHITE;
        Piece pc = NO_PIECE;
        switch (std::tolower(ch)) {
            case 'p': pc = make_piece(c, PAWN); break;
            case 'n': pc = make_piece(c, KNIGHT); break;
            case 'b': pc = make_piece(c, BISHOP); break;
            case 'r': pc = make_piece(c, ROOK); break;
            case 'q': pc = make_piece(c, QUEEN); break;
            case 'k': pc = make_piece(c, KING); break;
        }

        if (pc != NO_PIECE) {
            Square sq = make_square(static_cast<File>(file), static_cast<Rank>(rank));
            put_piece(pc, sq);
        }
        ++file;
    }

    // 2) Side to move
    std::string token;
    ss >> token;
    m_side_to_move = (token == "w") ? WHITE : BLACK;
    if (m_side_to_move == BLACK) m_hash ^= Zobrist::side_key;

    // 3) Castling rights
    ss >> token;
    m_castling_rights = NO_CASTLING;
    if (token != "-") {
        for (char ch : token) {
            if (ch == 'K') m_castling_rights = static_cast<CastlingRight>(m_castling_rights | WHITE_OO);
            if (ch == 'Q') m_castling_rights = static_cast<CastlingRight>(m_castling_rights | WHITE_OOO);
            if (ch == 'k') m_castling_rights = static_cast<CastlingRight>(m_castling_rights | BLACK_OO);
            if (ch == 'q') m_castling_rights = static_cast<CastlingRight>(m_castling_rights | BLACK_OOO);
        }
    }
    m_hash ^= Zobrist::castling_keys[m_castling_rights];

    // 4) En passant
    ss >> token;
    if (token != "-") {
        File f = static_cast<File>(token[0] - 'a');
        Rank r = static_cast<Rank>(token[1] - '1');
        m_en_passant = make_square(f, r);
        m_hash ^= Zobrist::en_passant_keys[f];
    } else {
        m_en_passant = SQ_NONE;
    }

    // 5) Halfmove clock
    if (ss >> token) {
        m_halfmove_clock = std::stoi(token);
    }

    // 6) Fullmove number
    if (ss >> token) {
        m_fullmove_number = std::stoi(token);
    }
}

std::string Position::get_fen() const {
    std::ostringstream ss;
    
    // Piece placement
    for (int r = 7; r >= 0; --r) {
        int empty_count = 0;
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(static_cast<File>(f), static_cast<Rank>(r));
            Piece pc = piece_on(sq);
            
            if (pc == NO_PIECE) {
                empty_count++;
            } else {
                if (empty_count > 0) {
                    ss << empty_count;
                    empty_count = 0;
                }
                ss << PIECE_TO_CHAR[pc];
            }
        }
        if (empty_count > 0) ss << empty_count;
        if (r > 0) ss << '/';
    }
    
    // Side to move
    ss << ' ' << (m_side_to_move == WHITE ? 'w' : 'b');
    
    // Castling rights
    ss << ' ';
    if (m_castling_rights == NO_CASTLING) {
        ss << '-';
    } else {
        if (m_castling_rights & WHITE_OO) ss << 'K';
        if (m_castling_rights & WHITE_OOO) ss << 'Q';
        if (m_castling_rights & BLACK_OO) ss << 'k';
        if (m_castling_rights & BLACK_OOO) ss << 'q';
    }
    
    // En passant square
    ss << ' ';
    if (m_en_passant == SQ_NONE) {
        ss << '-';
    } else {
        ss << square_to_string(m_en_passant);
    }
    
    // Halfmove clock and fullmove number
    ss << ' ' << m_halfmove_clock << ' ' << m_fullmove_number;
    
    return ss.str();
}

// ============================================================================
// Attack Detection
// ============================================================================

Bitboard Position::get_slider_attacks(Square sq, PieceType pt, Bitboard occupied) const {
    switch (pt) {
        case BISHOP: return get_bishop_attacks(sq, occupied);
        case ROOK:   return get_rook_attacks(sq, occupied);
        case QUEEN:  return get_queen_attacks(sq, occupied);
        default:     return BB_ZERO;
    }
}

Bitboard Position::attackers_to(Square sq, Bitboard occupied) const {
    Bitboard attackers = BB_ZERO;
    
    // Pawn attackers
    attackers |= pawn_attacks_bb_table(sq, WHITE) & pieces(BLACK, PAWN);
    attackers |= pawn_attacks_bb_table(sq, BLACK) & pieces(WHITE, PAWN);
    
    // Knight attackers
    attackers |= knight_attacks_bb(sq) & pieces(KNIGHT);
    
    // Bishop/Queen attackers (diagonal)
    attackers |= get_bishop_attacks(sq, occupied) & pieces(BISHOP, QUEEN);
    
    // Rook/Queen attackers (orthogonal)
    attackers |= get_rook_attacks(sq, occupied) & pieces(ROOK, QUEEN);
    
    // King attackers
    attackers |= king_attacks_bb(sq) & pieces(KING);
    
    return attackers;
}

bool Position::is_square_attacked(Square sq, Color attacker) const {
    // Check pawn attacks
    if (pawn_attacks_bb_table(sq, ~attacker) & pieces(attacker, PAWN))
        return true;
    
    // Check knight attacks
    if (knight_attacks_bb(sq) & pieces(attacker, KNIGHT))
        return true;
    
    // Check king attacks
    if (king_attacks_bb(sq) & pieces(attacker, KING))
        return true;
    
    // Check bishop/queen attacks (diagonal)
    if (get_bishop_attacks(sq, pieces()) & pieces(attacker, BISHOP, QUEEN))
        return true;
    
    // Check rook/queen attacks (orthogonal)
    if (get_rook_attacks(sq, pieces()) & pieces(attacker, ROOK, QUEEN))
        return true;
    
    return false;
}

// ============================================================================
// State Setters
// ============================================================================

void Position::set_side_to_move(Color c) {
    if (m_side_to_move != c) {
        m_hash ^= Zobrist::side_key;
        m_side_to_move = c;
    }
}

void Position::set_en_passant(Square sq) {
    if (m_en_passant != SQ_NONE) {
        m_hash ^= Zobrist::en_passant_keys[file_of(m_en_passant)];
    }
    m_en_passant = sq;
    if (sq != SQ_NONE) {
        m_hash ^= Zobrist::en_passant_keys[file_of(sq)];
    }
}

void Position::set_castling_rights(CastlingRight cr) {
    m_hash ^= Zobrist::castling_keys[m_castling_rights];
    m_castling_rights = cr;
    m_hash ^= Zobrist::castling_keys[m_castling_rights];
}

// ============================================================================
// Board Visualization
// ============================================================================

std::string Position::to_string() const {
    std::ostringstream ss;
    
    ss << "\n +---+---+---+---+---+---+---+---+\n";
    
    for (int r = 7; r >= 0; --r) {
        ss << " |";
        for (int f = 0; f < 8; ++f) {
            Square sq = make_square(static_cast<File>(f), static_cast<Rank>(r));
            Piece pc = piece_on(sq);
            char ch = (pc == NO_PIECE) ? ' ' : PIECE_TO_CHAR[pc];
            ss << ' ' << ch << " |";
        }
        ss << " " << (r + 1) << "\n";
        ss << " +---+---+---+---+---+---+---+---+\n";
    }
    
    ss << "   a   b   c   d   e   f   g   h\n";
    ss << "\nFEN: " << get_fen() << "\n";
    ss << "Hash: 0x" << std::hex << m_hash << std::dec << "\n";
    
    return ss.str();
}

void Position::print() const {
    std::cout << to_string() << std::flush;
}

// ============================================================================
// Low-Level Piece Operations (No Hash Updates)
// ============================================================================

void Position::put_piece_no_hash(Piece pc, Square sq) {
    assert(pc != NO_PIECE);
    assert(sq < SQUARE_NB);
    assert(m_board[sq] == NO_PIECE);
    
    Color c = color_of(pc);
    PieceType pt = type_of(pc);
    
    m_pieces[c][pt] |= square_bb(sq);
    m_by_color[c] |= square_bb(sq);
    m_board[sq] = pc;
}

void Position::remove_piece_no_hash(Square sq) {
    assert(sq < SQUARE_NB);
    
    Piece pc = m_board[sq];
    if (pc == NO_PIECE) return;
    
    Color c = color_of(pc);
    PieceType pt = type_of(pc);
    
    m_pieces[c][pt] &= ~square_bb(sq);
    m_by_color[c] &= ~square_bb(sq);
    m_board[sq] = NO_PIECE;
}

void Position::move_piece(Square from, Square to) {
    assert(from < SQUARE_NB && to < SQUARE_NB);
    assert(m_board[from] != NO_PIECE);
    
    Piece pc = m_board[from];
    Color c = color_of(pc);
    PieceType pt = type_of(pc);
    
    Bitboard from_to_bb = square_bb(from) | square_bb(to);
    
    m_pieces[c][pt] ^= from_to_bb;
    m_by_color[c] ^= from_to_bb;
    m_board[to] = pc;
    m_board[from] = NO_PIECE;
    
    // Update hash
    m_hash ^= Zobrist::piece_keys[pc][from];
    m_hash ^= Zobrist::piece_keys[pc][to];
}

// ============================================================================
// Move Execution
// ============================================================================

bool Position::make_move(Move m, StateInfo& state) {
    // Save current state for undo
    state.castling = m_castling_rights;
    state.ep_square = m_en_passant;
    state.halfmove_clock = m_halfmove_clock;
    state.zobrist_key = m_hash;
    state.captured_piece = NO_PIECE;
    
    Square from = from_sq(m);
    Square to = to_sq(m);
    Piece moving_piece = piece_on(from);
    Piece captured = piece_on(to);
    uint8_t mt = move_type(m);
    
    Color us = m_side_to_move;
    Color them = ~us;
    
    assert(moving_piece != NO_PIECE);
    assert(color_of(moving_piece) == us);
    
    // Clear en passant hash (if set)
    if (m_en_passant != SQ_NONE) {
        m_hash ^= Zobrist::en_passant_keys[file_of(m_en_passant)];
        m_en_passant = SQ_NONE;
    }
    
    // Handle captures
    if (captured != NO_PIECE) {
        state.captured_piece = captured;
        remove_piece_no_hash(to);
        m_hash ^= Zobrist::piece_keys[captured][to];
        m_halfmove_clock = 0;
    } else if (type_of(moving_piece) == PAWN) {
        m_halfmove_clock = 0;
    } else {
        m_halfmove_clock++;
    }
    
    // Handle special moves
    switch (mt) {
        case EN_PASSANT: {
            // Capture the pawn behind the target square
            Direction down = (us == WHITE) ? SOUTH : NORTH;
            Square captured_pawn_sq = to + down;
            Piece captured_pawn = piece_on(captured_pawn_sq);
            
            state.captured_piece = captured_pawn;
            remove_piece_no_hash(captured_pawn_sq);
            m_hash ^= Zobrist::piece_keys[captured_pawn][captured_pawn_sq];
            
            // Move the pawn
            move_piece(from, to);
            break;
        }
        
        case CASTLING: {
            // Move the king
            move_piece(from, to);
            
            // Move the rook
            Square rook_from, rook_to;
            
            if (to > from) { // Kingside
                rook_from = (us == WHITE) ? SQ_H1 : SQ_H8;
                rook_to = (us == WHITE) ? SQ_F1 : SQ_F8;
            } else { // Queenside
                rook_from = (us == WHITE) ? SQ_A1 : SQ_A8;
                rook_to = (us == WHITE) ? SQ_D1 : SQ_D8;
            }
            
            move_piece(rook_from, rook_to);
            break;
        }
        
        case PROMOTION: {
            PieceType promo_type = promotion_type(m);
            
            // Remove the pawn
            remove_piece_no_hash(from);
            m_hash ^= Zobrist::piece_keys[moving_piece][from];
            
            // Add the promoted piece
            Piece promoted = make_piece(us, promo_type);
            put_piece_no_hash(promoted, to);
            m_hash ^= Zobrist::piece_keys[promoted][to];
            break;
        }
        
        case DOUBLE_PAWN_PUSH: {
            // Move the pawn
            move_piece(from, to);
            
            // Set en passant square
            Direction up = (us == WHITE) ? NORTH : SOUTH;
            m_en_passant = from + up;
            m_hash ^= Zobrist::en_passant_keys[file_of(m_en_passant)];
            break;
        }
        
        default: // NORMAL, CAPTURE
            move_piece(from, to);
            break;
    }
    
    // Update castling rights
    CastlingRight old_castling = m_castling_rights;
    
    // Remove castling rights if king moves
    if (type_of(moving_piece) == KING) {
        if (us == WHITE) {
            m_castling_rights &= ~(WHITE_OO | WHITE_OOO);
        } else {
            m_castling_rights &= ~(BLACK_OO | BLACK_OOO);
        }
    }
    
    // Remove castling rights if rook moves or is captured
    if (from == SQ_A1 || to == SQ_A1) m_castling_rights &= ~WHITE_OOO;
    if (from == SQ_H1 || to == SQ_H1) m_castling_rights &= ~WHITE_OO;
    if (from == SQ_A8 || to == SQ_A8) m_castling_rights &= ~BLACK_OOO;
    if (from == SQ_H8 || to == SQ_H8) m_castling_rights &= ~BLACK_OO;
    
    if (old_castling != m_castling_rights) {
        m_hash ^= Zobrist::castling_keys[old_castling];
        m_hash ^= Zobrist::castling_keys[m_castling_rights];
    }
    
    // Switch side to move
    m_side_to_move = them;
    m_hash ^= Zobrist::side_key;
    
    // Update fullmove number
    if (us == BLACK) {
        m_fullmove_number++;
    }
    
    // Check if move is legal (doesn't leave our king in check)
    Square our_king_sq = lsb(pieces(us, KING));
    if (is_square_attacked(our_king_sq, them)) {
        // Illegal move - undo it
        undo_move(m, state);
        return false;
    }
    
    return true;
}

// ============================================================================
// Move Undo
// ============================================================================

void Position::undo_move(Move m, const StateInfo& state) {
    Square from = from_sq(m);
    Square to = to_sq(m);
    uint8_t mt = move_type(m);
    
    Color them = m_side_to_move; // Current side (after move was made)
    Color us = ~them;             // Side that made the move
    
    // Switch side back
    m_side_to_move = us;
    
    // Restore state variables
    m_castling_rights = state.castling;
    m_en_passant = state.ep_square;
    m_halfmove_clock = state.halfmove_clock;
    m_hash = state.zobrist_key;
    
    // Update fullmove number
    if (us == BLACK) {
        m_fullmove_number--;
    }
    
    // Undo special moves
    switch (mt) {
        case EN_PASSANT: {
            // Move pawn back
            Piece pawn = piece_on(to);
            remove_piece_no_hash(to);
            put_piece_no_hash(pawn, from);
            
            // Restore captured pawn behind the target square
            Direction down = (us == WHITE) ? SOUTH : NORTH;
            Square captured_pawn_sq = to + down;
            put_piece_no_hash(state.captured_piece, captured_pawn_sq);
            break;
        }
        
        case CASTLING: {
            // Move king back
            Piece king = piece_on(to);
            remove_piece_no_hash(to);
            put_piece_no_hash(king, from);
            
            // Move rook back
            Square rook_from, rook_to;
            
            if (to > from) { // Kingside
                rook_from = (us == WHITE) ? SQ_H1 : SQ_H8;
                rook_to = (us == WHITE) ? SQ_F1 : SQ_F8;
            } else { // Queenside
                rook_from = (us == WHITE) ? SQ_A1 : SQ_A8;
                rook_to = (us == WHITE) ? SQ_D1 : SQ_D8;
            }
            
            Piece rook = piece_on(rook_to);
            remove_piece_no_hash(rook_to);
            put_piece_no_hash(rook, rook_from);
            break;
        }
        
        case PROMOTION: {
            // Remove promoted piece
            remove_piece_no_hash(to);
            
            // Restore pawn
            Piece pawn = make_piece(us, PAWN);
            put_piece_no_hash(pawn, from);
            
            // Restore captured piece (if any)
            if (state.captured_piece != NO_PIECE) {
                put_piece_no_hash(state.captured_piece, to);
            }
            break;
        }
        
        default: // NORMAL, CAPTURE, DOUBLE_PAWN_PUSH
            // Move piece back
            Piece piece = piece_on(to);
            remove_piece_no_hash(to);
            put_piece_no_hash(piece, from);
            
            // Restore captured piece
            if (state.captured_piece != NO_PIECE) {
                put_piece_no_hash(state.captured_piece, to);
            }
            break;
    }
}
