#include "core/board.hpp"
#include <random>
#include <iostream>
#include <algorithm>
#include <cstring>

namespace chess
{

    // Static member initialization
    std::array<std::array<Bitboard, 64>, 2> AttackTables::pawnAttackTable;
    std::array<Bitboard, 64> AttackTables::knightAttackTable;
    std::array<Bitboard, 64> AttackTables::kingAttackTable;
    std::array<std::array<Bitboard, 64>, 64> AttackTables::betweenTable;
    std::array<std::array<Bitboard, 64>, 64> AttackTables::lineTable;
    std::array<AttackTables::Magic, 64> AttackTables::bishopMagics;
    std::array<AttackTables::Magic, 64> AttackTables::rookMagics;
    std::vector<Bitboard> AttackTables::bishopAttackTable;
    std::vector<Bitboard> AttackTables::rookAttackTable;

    std::array<std::array<uint64_t, 64>, 12> Board::pieceKeys;
    std::array<uint64_t, 16> Board::castlingKeys;
    std::array<uint64_t, 8> Board::epFileKeys;
    uint64_t Board::sideKey;
    bool Board::zobristInitialized = false;

    // Magic numbers for bishop attacks (found through trial)
    constexpr uint64_t BishopMagics[64] = {
        0x0002020202020200ULL, 0x0002020202020000ULL, 0x0004010202000000ULL, 0x0004040080000000ULL,
        0x0001104000000000ULL, 0x0000821040000000ULL, 0x0000410410400000ULL, 0x0000104104104000ULL,
        0x0000040404040400ULL, 0x0000020202020200ULL, 0x0000040102020000ULL, 0x0000040400800000ULL,
        0x0000011040000000ULL, 0x0000008210400000ULL, 0x0000004104104000ULL, 0x0000002082082000ULL,
        0x0004000808080800ULL, 0x0002000404040400ULL, 0x0001000202020200ULL, 0x0000800802004000ULL,
        0x0000800400A00000ULL, 0x0000200100884000ULL, 0x0000400082082000ULL, 0x0000200041041000ULL,
        0x0002080010101000ULL, 0x0001040008080800ULL, 0x0000208004010400ULL, 0x0000404004010200ULL,
        0x0000840000802000ULL, 0x0000404002011000ULL, 0x0000808001041000ULL, 0x0000404000820800ULL,
        0x0001041000202000ULL, 0x0000820800101000ULL, 0x0000104400080800ULL, 0x0000020080080080ULL,
        0x0000404040040100ULL, 0x0000808100020100ULL, 0x0001010100020800ULL, 0x0000808080010400ULL,
        0x0000820820004000ULL, 0x0000410410002000ULL, 0x0000082088001000ULL, 0x0000002011000800ULL,
        0x0000080100400400ULL, 0x0001010101000200ULL, 0x0002020202000400ULL, 0x0001010101000200ULL,
        0x0000410410400000ULL, 0x0000208208200000ULL, 0x0000002084100000ULL, 0x0000000020880000ULL,
        0x0000001002020000ULL, 0x0000040408020000ULL, 0x0004040404040000ULL, 0x0002020202020000ULL,
        0x0000104104104000ULL, 0x0000002082082000ULL, 0x0000000020841000ULL, 0x0000000000208800ULL,
        0x0000000010020200ULL, 0x0000000404080200ULL, 0x0000040404040400ULL, 0x0002020202020200ULL};

    // Magic numbers for rook attacks
    constexpr uint64_t RookMagics[64] = {
        0x0080001020400080ULL, 0x0040001000200040ULL, 0x0080081000200080ULL, 0x0080040800100080ULL,
        0x0080020400080080ULL, 0x0080010200040080ULL, 0x0080008001000200ULL, 0x0080002040800100ULL,
        0x0000800020400080ULL, 0x0000400020005000ULL, 0x0000801000200080ULL, 0x0000800800100080ULL,
        0x0000800400080080ULL, 0x0000800200040080ULL, 0x0000800100020080ULL, 0x0000800040800100ULL,
        0x0000208000400080ULL, 0x0000404000201000ULL, 0x0000808010002000ULL, 0x0000808008001000ULL,
        0x0000808004000800ULL, 0x0000808002000400ULL, 0x0000010100020004ULL, 0x0000020000408104ULL,
        0x0000208080004000ULL, 0x0000200040005000ULL, 0x0000100080200080ULL, 0x0000080080100080ULL,
        0x0000040080080080ULL, 0x0000020080040080ULL, 0x0000010080800200ULL, 0x0000800080004100ULL,
        0x0000204000800080ULL, 0x0000200040401000ULL, 0x0000100080802000ULL, 0x0000080080801000ULL,
        0x0000040080800800ULL, 0x0000020080800400ULL, 0x0000020001010004ULL, 0x0000800040800100ULL,
        0x0000204000808000ULL, 0x0000200040008080ULL, 0x0000100020008080ULL, 0x0000080010008080ULL,
        0x0000040008008080ULL, 0x0000020004008080ULL, 0x0000010002008080ULL, 0x0000004081020004ULL,
        0x0000204000800080ULL, 0x0000200040008080ULL, 0x0000100020008080ULL, 0x0000080010008080ULL,
        0x0000040008008080ULL, 0x0000020004008080ULL, 0x0000800100020080ULL, 0x0000800041000080ULL,
        0x00FFFCDDFCED714AULL, 0x007FFCDDFCED714AULL, 0x003FFFCDFFD88096ULL, 0x0000040810002101ULL,
        0x0001000204080011ULL, 0x0001000204000801ULL, 0x0001000082000401ULL, 0x0001FFFAABFAD1A2ULL};

    void AttackTables::init()
    {
        initPawnAttacks();
        initKnightAttacks();
        initKingAttacks();
        initMagics();
        initBetweenAndLine();
    }

    void AttackTables::initPawnAttacks()
    {
        for (int sq = 0; sq < 64; sq++)
        {
            Bitboard sqBB = bb::squareBB(sq);

            // White pawn attacks (up-left and up-right)
            pawnAttackTable[0][sq] = ((sqBB & ~bb::FILE_A) << 7) | ((sqBB & ~bb::FILE_H) << 9);

            // Black pawn attacks (down-left and down-right)
            pawnAttackTable[1][sq] = ((sqBB & ~bb::FILE_H) >> 7) | ((sqBB & ~bb::FILE_A) >> 9);
        }
    }

    void AttackTables::initKnightAttacks()
    {
        for (int sq = 0; sq < 64; sq++)
        {
            Bitboard sqBB = bb::squareBB(sq);
            Bitboard attacks = 0;

            // All 8 knight moves
            if (sqBB & ~(bb::FILE_A | bb::FILE_B))
            {
                attacks |= (sqBB << 6) | (sqBB >> 10);
            }
            if (sqBB & ~bb::FILE_A)
            {
                attacks |= (sqBB << 15) | (sqBB >> 17);
            }
            if (sqBB & ~bb::FILE_H)
            {
                attacks |= (sqBB << 17) | (sqBB >> 15);
            }
            if (sqBB & ~(bb::FILE_G | bb::FILE_H))
            {
                attacks |= (sqBB << 10) | (sqBB >> 6);
            }

            knightAttackTable[sq] = attacks;
        }
    }

    void AttackTables::initKingAttacks()
    {
        for (int sq = 0; sq < 64; sq++)
        {
            Bitboard sqBB = bb::squareBB(sq);
            Bitboard attacks = 0;

            attacks |= (sqBB << 8) | (sqBB >> 8); // Up and down
            attacks |= ((sqBB & ~bb::FILE_A) << 7) | ((sqBB & ~bb::FILE_A) >> 1) | ((sqBB & ~bb::FILE_A) >> 9);
            attacks |= ((sqBB & ~bb::FILE_H) << 9) | ((sqBB & ~bb::FILE_H) << 1) | ((sqBB & ~bb::FILE_H) >> 7);

            kingAttackTable[sq] = attacks;
        }
    }

    Bitboard AttackTables::slidingAttack(Square sq, Bitboard occupied, const int deltas[4])
    {
        Bitboard attacks = 0;

        for (int i = 0; i < 4; i++)
        {
            int s = sq;
            while (true)
            {
                int prevFile = s & 7;
                int prevRank = s >> 3;
                s += deltas[i];

                if (s < 0 || s >= 64)
                    break;

                int newFile = s & 7;
                int newRank = s >> 3;

                // Check for wrap-around
                if (std::abs(newFile - prevFile) > 1 || std::abs(newRank - prevRank) > 1)
                    break;

                attacks |= bb::squareBB(s);

                if (occupied & bb::squareBB(s))
                    break;
            }
        }

        return attacks;
    }

    void AttackTables::initMagics()
    {
        constexpr int bishopDeltas[] = {-9, -7, 7, 9};
        constexpr int rookDeltas[] = {-8, -1, 1, 8};

        // Initialize bishop magics
        bishopAttackTable.resize(0x1480);
        Bitboard *bAttack = bishopAttackTable.data();

        for (int sq = 0; sq < 64; sq++)
        {
            // Calculate mask (edges excluded)
            Bitboard edges = ((bb::RANK_1 | bb::RANK_8) & ~(bb::squareBB(sq) | (bb::RANK_1 & bb::squareBB(sq)) |
                                                            (bb::RANK_8 & bb::squareBB(sq)))) |
                             ((bb::FILE_A | bb::FILE_H) & ~(bb::squareBB(sq) | (bb::FILE_A & bb::squareBB(sq)) |
                                                            (bb::FILE_H & bb::squareBB(sq))));

            Bitboard mask = slidingAttack(sq, 0, bishopDeltas) & ~edges;

            bishopMagics[sq].mask = mask;
            bishopMagics[sq].magic = BishopMagics[sq];
            bishopMagics[sq].shift = 64 - bb::popcount(mask);
            bishopMagics[sq].attacks = bAttack;

            // Enumerate all subsets of the mask
            Bitboard subset = 0;
            do
            {
                unsigned idx = bishopMagics[sq].index(subset);
                bAttack[idx] = slidingAttack(sq, subset, bishopDeltas);
                subset = (subset - mask) & mask;
            } while (subset);

            bAttack += 1ULL << bb::popcount(mask);
        }

        // Initialize rook magics
        rookAttackTable.resize(0x19000);
        Bitboard *rAttack = rookAttackTable.data();

        for (int sq = 0; sq < 64; sq++)
        {
            Bitboard edges = ((bb::RANK_1 | bb::RANK_8) & ~(bb::squareBB(sq & 56))) |
                             ((bb::FILE_A | bb::FILE_H) & ~(bb::squareBB(sq & 7)));

            // Remove edges except for current rank/file
            int file = sq & 7;
            int rank = sq >> 3;

            Bitboard mask = 0;
            for (int r = rank + 1; r < 7; r++)
                mask |= bb::squareBB(file + r * 8);
            for (int r = rank - 1; r > 0; r--)
                mask |= bb::squareBB(file + r * 8);
            for (int f = file + 1; f < 7; f++)
                mask |= bb::squareBB(f + rank * 8);
            for (int f = file - 1; f > 0; f--)
                mask |= bb::squareBB(f + rank * 8);

            rookMagics[sq].mask = mask;
            rookMagics[sq].magic = RookMagics[sq];
            rookMagics[sq].shift = 64 - bb::popcount(mask);
            rookMagics[sq].attacks = rAttack;

            Bitboard subset = 0;
            do
            {
                unsigned idx = rookMagics[sq].index(subset);
                rAttack[idx] = slidingAttack(sq, subset, rookDeltas);
                subset = (subset - mask) & mask;
            } while (subset);

            rAttack += 1ULL << bb::popcount(mask);
        }
    }

    void AttackTables::initBetweenAndLine()
    {
        for (int sq1 = 0; sq1 < 64; sq1++)
        {
            for (int sq2 = 0; sq2 < 64; sq2++)
            {
                betweenTable[sq1][sq2] = 0;
                lineTable[sq1][sq2] = 0;

                if (sq1 == sq2)
                    continue;

                // Check if squares are on same line (diagonal or straight)
                Bitboard sqBB1 = bb::squareBB(sq1);
                Bitboard sqBB2 = bb::squareBB(sq2);

                Bitboard bishopAttack1 = bishopAttacks(sq1, 0);
                Bitboard rookAttack1 = rookAttacks(sq1, 0);

                if (bishopAttack1 & sqBB2)
                {
                    lineTable[sq1][sq2] = (bishopAttack1 & bishopAttacks(sq2, 0)) | sqBB1 | sqBB2;
                    betweenTable[sq1][sq2] = bishopAttacks(sq1, sqBB2) & bishopAttacks(sq2, sqBB1);
                }
                else if (rookAttack1 & sqBB2)
                {
                    lineTable[sq1][sq2] = (rookAttack1 & rookAttacks(sq2, 0)) | sqBB1 | sqBB2;
                    betweenTable[sq1][sq2] = rookAttacks(sq1, sqBB2) & rookAttacks(sq2, sqBB1);
                }
            }
        }
    }

    Bitboard AttackTables::pawnAttacks(Color c, Square sq)
    {
        return pawnAttackTable[static_cast<int>(c)][sq];
    }

    Bitboard AttackTables::knightAttacks(Square sq)
    {
        return knightAttackTable[sq];
    }

    Bitboard AttackTables::kingAttacks(Square sq)
    {
        return kingAttackTable[sq];
    }

    Bitboard AttackTables::bishopAttacks(Square sq, Bitboard occupied)
    {
        return bishopMagics[sq].attacks[bishopMagics[sq].index(occupied)];
    }

    Bitboard AttackTables::rookAttacks(Square sq, Bitboard occupied)
    {
        return rookMagics[sq].attacks[rookMagics[sq].index(occupied)];
    }

    Bitboard AttackTables::queenAttacks(Square sq, Bitboard occupied)
    {
        return bishopAttacks(sq, occupied) | rookAttacks(sq, occupied);
    }

    Bitboard AttackTables::betweenBB(Square sq1, Square sq2)
    {
        return betweenTable[sq1][sq2];
    }

    Bitboard AttackTables::lineBB(Square sq1, Square sq2)
    {
        return lineTable[sq1][sq2];
    }

    // Board implementation
    void Board::initZobrist()
    {
        if (zobristInitialized)
            return;

        std::mt19937_64 rng(0x12345678);

        for (int p = 0; p < 12; p++)
        {
            for (int sq = 0; sq < 64; sq++)
            {
                pieceKeys[p][sq] = rng();
            }
        }

        for (int i = 0; i < 16; i++)
        {
            castlingKeys[i] = rng();
        }

        for (int i = 0; i < 8; i++)
        {
            epFileKeys[i] = rng();
        }

        sideKey = rng();
        zobristInitialized = true;
    }

    Board::Board() : state(&rootState), fullmove(1), side(Color::White)
    {
        initZobrist();
        AttackTables::init();
        setStartingPosition();
    }

    Board::Board(const std::string &fen) : state(&rootState), fullmove(1), side(Color::White)
    {
        initZobrist();
        AttackTables::init();
        setFromFEN(fen);
    }

    Board::Board(const Board& other)
        : pieces(other.pieces),
          byType(other.byType),
          byColor(other.byColor),
          side(other.side),
          fullmove(other.fullmove),
          rootState(other.rootState),
          state(&rootState),
          keyHistory(other.keyHistory)
    {
        // Copy the current state values, but point to our own rootState
        // This ensures the copied board is independent
        if (other.state != &other.rootState) {
            // The original has made moves - copy the current state values
            rootState = *other.state;
            rootState.previous = nullptr;  // We don't track history in copies
        }
    }

    Board& Board::operator=(const Board& other)
    {
        if (this != &other) {
            pieces = other.pieces;
            byType = other.byType;
            byColor = other.byColor;
            side = other.side;
            fullmove = other.fullmove;
            keyHistory = other.keyHistory;
            
            // Copy state and point to our own rootState
            rootState = other.rootState;
            state = &rootState;
            
            if (other.state != &other.rootState) {
                rootState = *other.state;
                rootState.previous = nullptr;
            }
        }
        return *this;
    }


    Board::Board(Board&& other) noexcept
        : pieces(std::move(other.pieces)),
          byType(std::move(other.byType)),
          byColor(std::move(other.byColor)),
          side(other.side),
          fullmove(other.fullmove),
          rootState(other.rootState),
          state(&rootState),
          keyHistory(std::move(other.keyHistory))
    {
        // Move constructor: take ownership but set state to our own rootState
        if (other.state != &other.rootState) {
            rootState = *other.state;
            rootState.previous = nullptr;
        }
    }

    Board& Board::operator=(Board&& other) noexcept
    {
        if (this != &other) {
            pieces = std::move(other.pieces);
            byType = std::move(other.byType);
            byColor = std::move(other.byColor);
            side = other.side;
            fullmove = other.fullmove;
            keyHistory = std::move(other.keyHistory);
            
            rootState = other.rootState;
            state = &rootState;
            
            if (other.state != &other.rootState) {
                rootState = *other.state;
                rootState.previous = nullptr;
            }
        }
        return *this;
    }

    void Board::setStartingPosition()
    {
        setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    void Board::putPiece(Piece p, Square sq)
    {
        pieces[sq] = p;
        Bitboard sqBB = bb::squareBB(sq);
        byType[static_cast<int>(typeOf(p))] |= sqBB;
        byColor[static_cast<int>(colorOf(p))] |= sqBB;
    }

    void Board::removePiece(Square sq)
    {
        Piece p = pieces[sq];
        if (p == Piece::None)
            return;

        Bitboard sqBB = bb::squareBB(sq);
        byType[static_cast<int>(typeOf(p))] &= ~sqBB;
        byColor[static_cast<int>(colorOf(p))] &= ~sqBB;
        pieces[sq] = Piece::None;
    }

    void Board::movePiece(Square from, Square to)
    {
        Piece p = pieces[from];
        Bitboard fromTo = bb::squareBB(from) | bb::squareBB(to);
        byType[static_cast<int>(typeOf(p))] ^= fromTo;
        byColor[static_cast<int>(colorOf(p))] ^= fromTo;
        pieces[to] = p;
        pieces[from] = Piece::None;
    }

    void Board::setFromFEN(const std::string &fen)
    {
        // Clear board
        pieces.fill(Piece::None);
        byType.fill(0);
        byColor.fill(0);
        keyHistory.clear();

        std::istringstream ss(fen);
        std::string board, sideStr, castling, ep;
        int halfmove = 0;

        ss >> board >> sideStr >> castling >> ep >> halfmove >> fullmove;

        // Parse piece placement
        int sq = 56; // Start at a8
        for (char c : board)
        {
            if (c == '/')
            {
                sq -= 16;
            }
            else if (c >= '1' && c <= '8')
            {
                sq += (c - '0');
            }
            else
            {
                Piece p;
                switch (c)
                {
                case 'P':
                    p = Piece::WhitePawn;
                    break;
                case 'N':
                    p = Piece::WhiteKnight;
                    break;
                case 'B':
                    p = Piece::WhiteBishop;
                    break;
                case 'R':
                    p = Piece::WhiteRook;
                    break;
                case 'Q':
                    p = Piece::WhiteQueen;
                    break;
                case 'K':
                    p = Piece::WhiteKing;
                    break;
                case 'p':
                    p = Piece::BlackPawn;
                    break;
                case 'n':
                    p = Piece::BlackKnight;
                    break;
                case 'b':
                    p = Piece::BlackBishop;
                    break;
                case 'r':
                    p = Piece::BlackRook;
                    break;
                case 'q':
                    p = Piece::BlackQueen;
                    break;
                case 'k':
                    p = Piece::BlackKing;
                    break;
                default:
                    continue;
                }
                putPiece(p, sq++);
            }
        }

        // Side to move
        side = (sideStr == "w") ? Color::White : Color::Black;

        // Castling rights
        rootState.castling = CastlingRights::None;
        for (char c : castling)
        {
            switch (c)
            {
            case 'K':
                rootState.castling |= CastlingRights::WhiteKingSide;
                break;
            case 'Q':
                rootState.castling |= CastlingRights::WhiteQueenSide;
                break;
            case 'k':
                rootState.castling |= CastlingRights::BlackKingSide;
                break;
            case 'q':
                rootState.castling |= CastlingRights::BlackQueenSide;
                break;
            }
        }

        // En passant
        if (ep != "-" && ep.length() >= 2)
        {
            rootState.epSquare = bb::makeSquare(ep[0] - 'a', ep[1] - '1');
        }
        else
        {
            rootState.epSquare = NO_SQUARE;
        }

        rootState.halfmoveClock = halfmove;
        rootState.pliesFromNull = 0;
        rootState.capturedPiece = Piece::None;
        rootState.previous = nullptr;

        state = &rootState;
        updateKey();
    }

    std::string Board::toFEN() const
    {
        std::string fen;

        // Piece placement
        for (int rank = 7; rank >= 0; rank--)
        {
            int empty = 0;
            for (int file = 0; file < 8; file++)
            {
                Square sq = bb::makeSquare(file, rank);
                Piece p = pieces[sq];

                if (p == Piece::None)
                {
                    empty++;
                }
                else
                {
                    if (empty > 0)
                    {
                        fen += static_cast<char>('0' + empty);
                        empty = 0;
                    }
                    const char pieceChars[] = "PNBRQKpnbrqk";
                    fen += pieceChars[static_cast<int>(p)];
                }
            }
            if (empty > 0)
            {
                fen += static_cast<char>('0' + empty);
            }
            if (rank > 0)
                fen += '/';
        }

        // Side to move
        fen += ' ';
        fen += (side == Color::White) ? 'w' : 'b';

        // Castling
        fen += ' ';
        if (state->castling == CastlingRights::None)
        {
            fen += '-';
        }
        else
        {
            if ((state->castling & CastlingRights::WhiteKingSide) != CastlingRights::None)
                fen += 'K';
            if ((state->castling & CastlingRights::WhiteQueenSide) != CastlingRights::None)
                fen += 'Q';
            if ((state->castling & CastlingRights::BlackKingSide) != CastlingRights::None)
                fen += 'k';
            if ((state->castling & CastlingRights::BlackQueenSide) != CastlingRights::None)
                fen += 'q';
        }

        // En passant
        fen += ' ';
        if (state->epSquare == NO_SQUARE)
        {
            fen += '-';
        }
        else
        {
            fen += static_cast<char>('a' + bb::fileOf(state->epSquare));
            fen += static_cast<char>('1' + bb::rankOf(state->epSquare));
        }

        // Halfmove clock and fullmove number
        fen += ' ' + std::to_string(state->halfmoveClock);
        fen += ' ' + std::to_string(fullmove);

        return fen;
    }

    void Board::updateKey()
    {
        state->key = 0;

        // Pieces
        Bitboard occ = occupied();
        while (occ)
        {
            Square sq = bb::popLsb(occ);
            Piece p = pieces[sq];
            state->key ^= pieceKeys[static_cast<int>(p)][sq];
        }

        // Castling
        state->key ^= castlingKeys[static_cast<int>(state->castling)];

        // En passant
        if (state->epSquare != NO_SQUARE)
        {
            state->key ^= epFileKeys[bb::fileOf(state->epSquare)];
        }

        // Side to move
        if (side == Color::Black)
        {
            state->key ^= sideKey;
        }
    }

    Bitboard Board::attackersTo(Square sq, Bitboard occupied) const
    {
        return (AttackTables::pawnAttacks(Color::Black, sq) & piecesBB(Color::White, PieceType::Pawn)) |
               (AttackTables::pawnAttacks(Color::White, sq) & piecesBB(Color::Black, PieceType::Pawn)) |
               (AttackTables::knightAttacks(sq) & piecesBB(PieceType::Knight)) |
               (AttackTables::kingAttacks(sq) & piecesBB(PieceType::King)) |
               (AttackTables::bishopAttacks(sq, occupied) & (piecesBB(PieceType::Bishop) | piecesBB(PieceType::Queen))) |
               (AttackTables::rookAttacks(sq, occupied) & (piecesBB(PieceType::Rook) | piecesBB(PieceType::Queen)));
    }

    Bitboard Board::checkers() const
    {
        return attackersTo(kingSquare(side)) & piecesBB(~side);
    }

    Bitboard Board::pinnedPieces(Color c) const
    {
        Bitboard pinned = 0;
        Square kingSq = kingSquare(c);
        Color them = ~c;

        // Potential pinners (sliders that could attack the king)
        Bitboard pinners = ((AttackTables::rookAttacks(kingSq, 0) & (piecesBB(them, PieceType::Rook) | piecesBB(them, PieceType::Queen))) |
                            (AttackTables::bishopAttacks(kingSq, 0) & (piecesBB(them, PieceType::Bishop) | piecesBB(them, PieceType::Queen))));

        while (pinners)
        {
            Square sq = bb::popLsb(pinners);
            Bitboard between = AttackTables::betweenBB(kingSq, sq) & occupied();

            if (bb::popcount(between) == 1)
            {
                pinned |= between & piecesBB(c);
            }
        }

        return pinned;
    }

    void Board::makeMove(Move m, StateInfo &newState)
    {
        // Copy state
        std::memcpy(&newState, state, sizeof(StateInfo));
        newState.previous = state;
        state = &newState;

        Color us = side;
        Color them = ~us;
        Square from = m.from();
        Square to = m.to();
        Piece movedPiece = pieces[from];
        Piece captured = pieces[to];

        // Save captured piece
        state->capturedPiece = captured;

        // Update halfmove clock
        state->halfmoveClock++;
        state->pliesFromNull++;

        // Handle en passant capture
        if (m.type() == MoveType::EnPassant)
        {
            Square epCaptureSq = bb::makeSquare(bb::fileOf(to), bb::rankOf(from));
            captured = pieces[epCaptureSq];
            removePiece(epCaptureSq);
            state->capturedPiece = captured;
        }

        // Clear en passant square
        if (state->epSquare != NO_SQUARE)
        {
            state->key ^= epFileKeys[bb::fileOf(state->epSquare)];
            state->epSquare = NO_SQUARE;
        }

        // Handle captures
        if (captured != Piece::None && m.type() != MoveType::EnPassant)
        {
            removePiece(to);
            state->halfmoveClock = 0;
        }

        // Move the piece
        movePiece(from, to);

        // Handle pawn moves
        if (typeOf(movedPiece) == PieceType::Pawn)
        {
            state->halfmoveClock = 0;

            // Double pawn push - set en passant square
            if (m.type() == MoveType::DoublePawnPush)
            {
                state->epSquare = bb::makeSquare(bb::fileOf(from), (bb::rankOf(from) + bb::rankOf(to)) / 2);
                state->key ^= epFileKeys[bb::fileOf(state->epSquare)];
            }

            // Promotion
            if (m.isPromotion())
            {
                removePiece(to);
                putPiece(makePiece(us, m.promotionPiece()), to);
            }
        }

        // Handle castling
        if (m.isCastle())
        {
            Square rookFrom, rookTo;
            if (m.type() == MoveType::KingSideCastle)
            {
                rookFrom = (us == Color::White) ? H1 : H8;
                rookTo = (us == Color::White) ? F1 : F8;
            }
            else
            {
                rookFrom = (us == Color::White) ? A1 : A8;
                rookTo = (us == Color::White) ? D1 : D8;
            }
            movePiece(rookFrom, rookTo);
        }

        // Update castling rights
        state->key ^= castlingKeys[static_cast<int>(state->castling)];
        updateCastlingRights(from, to);
        state->key ^= castlingKeys[static_cast<int>(state->castling)];

        // Update Zobrist key for moved piece
        state->key ^= pieceKeys[static_cast<int>(movedPiece)][from];
        if (m.isPromotion())
        {
            state->key ^= pieceKeys[static_cast<int>(makePiece(us, m.promotionPiece()))][to];
        }
        else
        {
            state->key ^= pieceKeys[static_cast<int>(movedPiece)][to];
        }

        if (captured != Piece::None)
        {
            if (m.type() == MoveType::EnPassant)
            {
                Square epSq = bb::makeSquare(bb::fileOf(to), bb::rankOf(from));
                state->key ^= pieceKeys[static_cast<int>(captured)][epSq];
            }
            else
            {
                state->key ^= pieceKeys[static_cast<int>(captured)][to];
            }
        }

        // Switch side
        side = them;
        state->key ^= sideKey;

        if (us == Color::Black)
        {
            fullmove++;
        }

        // Store key in history
        keyHistory.push_back(state->key);
    }

    void Board::unmakeMove(Move m)
    {
        Color them = side;
        Color us = ~them;
        Square from = m.from();
        Square to = m.to();

        // Remove from history
        keyHistory.pop_back();

        // Move piece back
        movePiece(to, from);

        // Handle promotion - restore pawn
        if (m.isPromotion())
        {
            removePiece(from);
            putPiece(makePiece(us, PieceType::Pawn), from);
        }

        // Restore captured piece
        if (state->capturedPiece != Piece::None)
        {
            if (m.type() == MoveType::EnPassant)
            {
                Square epCaptureSq = bb::makeSquare(bb::fileOf(to), bb::rankOf(from));
                putPiece(state->capturedPiece, epCaptureSq);
            }
            else
            {
                putPiece(state->capturedPiece, to);
            }
        }

        // Handle castling - move rook back
        if (m.isCastle())
        {
            Square rookFrom, rookTo;
            if (m.type() == MoveType::KingSideCastle)
            {
                rookFrom = (us == Color::White) ? H1 : H8;
                rookTo = (us == Color::White) ? F1 : F8;
            }
            else
            {
                rookFrom = (us == Color::White) ? A1 : A8;
                rookTo = (us == Color::White) ? D1 : D8;
            }
            movePiece(rookTo, rookFrom);
        }

        // Restore state
        state = state->previous;
        side = us;

        if (us == Color::Black)
        {
            fullmove--;
        }
    }

    void Board::updateCastlingRights(Square from, Square to)
    {
        // King moves remove all castling rights for that side
        if (from == E1)
        {
            state->castling &= ~CastlingRights::WhiteBoth;
        }
        else if (from == E8)
        {
            state->castling &= ~CastlingRights::BlackBoth;
        }

        // Rook moves or captures remove specific castling rights
        if (from == A1 || to == A1)
        {
            state->castling &= ~CastlingRights::WhiteQueenSide;
        }
        if (from == H1 || to == H1)
        {
            state->castling &= ~CastlingRights::WhiteKingSide;
        }
        if (from == A8 || to == A8)
        {
            state->castling &= ~CastlingRights::BlackQueenSide;
        }
        if (from == H8 || to == H8)
        {
            state->castling &= ~CastlingRights::BlackKingSide;
        }
    }

    void Board::makeNullMove(StateInfo &newState)
    {
        std::memcpy(&newState, state, sizeof(StateInfo));
        newState.previous = state;
        state = &newState;

        if (state->epSquare != NO_SQUARE)
        {
            state->key ^= epFileKeys[bb::fileOf(state->epSquare)];
            state->epSquare = NO_SQUARE;
        }

        state->key ^= sideKey;
        state->pliesFromNull = 0;
        side = ~side;

        keyHistory.push_back(state->key);
    }

    void Board::unmakeNullMove()
    {
        keyHistory.pop_back();
        state = state->previous;
        side = ~side;
    }

    bool Board::isDrawByRepetition(int count) const
    {
        int repetitions = 0;

        for (int i = static_cast<int>(keyHistory.size()) - 2;
             i >= 0 && i >= static_cast<int>(keyHistory.size()) - state->pliesFromNull;
             i -= 2)
        {
            if (keyHistory[i] == state->key)
            {
                repetitions++;
                if (repetitions >= count)
                    return true;
            }
        }

        return false;
    }

    bool Board::isInsufficientMaterial() const
    {
        // Pawns, rooks, or queens = sufficient
        if (piecesBB(PieceType::Pawn) || piecesBB(PieceType::Rook) || piecesBB(PieceType::Queen))
        {
            return false;
        }

        int numPieces = bb::popcount(occupied());

        // K vs K
        if (numPieces == 2)
            return true;

        // K+B vs K or K+N vs K
        if (numPieces == 3)
        {
            if (piecesBB(PieceType::Bishop) || piecesBB(PieceType::Knight))
            {
                return true;
            }
        }

        // K+B vs K+B with same colored bishops
        if (numPieces == 4 && bb::popcount(piecesBB(PieceType::Bishop)) == 2)
        {
            Bitboard bishops = piecesBB(PieceType::Bishop);
            if ((bishops & bb::LIGHT_SQUARES) == bishops || (bishops & bb::DARK_SQUARES) == bishops)
            {
                return true;
            }
        }

        return false;
    }

    bool Board::isLegal(Move m) const
    {
        Color us = side;
        Color them = ~us;
        Square from = m.from();
        Square to = m.to();
        Square kingSq = kingSquare(us);

        // King moves - check if destination is attacked
        // IMPORTANT: We must check attacks with the king removed from its current square!
        // Otherwise, sliding attacks will be blocked by the king itself.
        if (from == kingSq)
        {
            // Create occupancy without the king at 'from', but WITH king at 'to'
            Bitboard occ = occupied() & ~bb::squareBB(from);
            occ &= ~bb::squareBB(to);    // Remove any captured piece first
            occ |= bb::squareBB(to);     // Then add king to destination
            
            // Check if any enemy piece attacks 'to' with this occupancy
            Bitboard attackers = attackersTo(to, occ) & piecesBB(them);
            
            return !attackers;
        }

        // For ALL other moves: if we're in check, we must verify the move blocks/captures
        // This is done by simulating the move and checking if we're still in check
        // This handles all cases: pinned pieces, blocking checks, capturing checkers, etc.
        
        // Special fast path for en passant (complex due to two pieces being removed)
        if (m.type() == MoveType::EnPassant)
        {
            // Calculate where the captured pawn is (same rank as 'from', same file as 'to')
            Square capturedPawnSq = bb::makeSquare(bb::fileOf(to), bb::rankOf(from));
            
            // Simulate the position after en passant by updating occupancy
            Bitboard occ = occupied();
            occ &= ~bb::squareBB(from);           // Remove our pawn
            occ &= ~bb::squareBB(capturedPawnSq); // Remove captured pawn  
            occ |= bb::squareBB(to);              // Add our pawn to destination
            
            // Mask to exclude the captured pawn from piece bitboards
            Bitboard capturedMask = ~bb::squareBB(capturedPawnSq);
            
            // Check if king would be in check with this new occupancy
            Bitboard enemyRooksQueens = (piecesBB(them, PieceType::Rook) | 
                                         piecesBB(them, PieceType::Queen));
            Bitboard enemyBishopsQueens = (piecesBB(them, PieceType::Bishop) | 
                                           piecesBB(them, PieceType::Queen));
            
            if (AttackTables::rookAttacks(kingSq, occ) & enemyRooksQueens)
                return false;
            if (AttackTables::bishopAttacks(kingSq, occ) & enemyBishopsQueens)
                return false;
            if (AttackTables::knightAttacks(kingSq) & piecesBB(them, PieceType::Knight))
                return false;
            // Exclude captured pawn from pawn attack check
            if (AttackTables::pawnAttacks(us, kingSq) & (piecesBB(them, PieceType::Pawn) & capturedMask))
                return false;
            if (AttackTables::kingAttacks(kingSq) & piecesBB(them, PieceType::King))
                return false;
            
            return true;
        }

        // Castling - check squares between king and destination
        if (m.isCastle())
        {
            if (inCheck())
                return false;

            int direction = (m.type() == MoveType::KingSideCastle) ? 1 : -1;
            for (Square sq = kingSq + direction; sq != to; sq += direction)
            {
                if (isAttacked(sq, them))
                    return false;
            }
            return !isAttacked(to, them);
        }

        // For normal moves (non-king, non-castling, non-en-passant):
        // Simulate the move and check if king would be in check
        
        // Create occupancy after the move
        Bitboard occ = occupied();
        occ &= ~bb::squareBB(from);  // Remove piece from 'from'
        occ &= ~bb::squareBB(to);    // Remove any captured piece
        occ |= bb::squareBB(to);     // Add our piece to 'to'
        
        // Check if king would be attacked with this occupancy
        // For sliding pieces, use the updated occupancy AND exclude captured pieces
        Bitboard capturedMask = ~bb::squareBB(to);  // Mask to exclude captured piece
        Bitboard enemyRooksQueens = (piecesBB(them, PieceType::Rook) | 
                                     piecesBB(them, PieceType::Queen)) & capturedMask;
        Bitboard enemyBishopsQueens = (piecesBB(them, PieceType::Bishop) | 
                                       piecesBB(them, PieceType::Queen)) & capturedMask;
        
        if (AttackTables::rookAttacks(kingSq, occ) & enemyRooksQueens)
            return false;
        if (AttackTables::bishopAttacks(kingSq, occ) & enemyBishopsQueens)
            return false;
        
        // For non-sliding pieces, exclude any piece at the 'to' square (captured)
        Bitboard enemyKnights = piecesBB(them, PieceType::Knight) & capturedMask;
        Bitboard enemyPawns = piecesBB(them, PieceType::Pawn) & capturedMask;
        Bitboard enemyKing = piecesBB(them, PieceType::King) & capturedMask;
        
        if (AttackTables::knightAttacks(kingSq) & enemyKnights)
            return false;
        if (AttackTables::pawnAttacks(us, kingSq) & enemyPawns)
            return false;
        if (AttackTables::kingAttacks(kingSq) & enemyKing)
            return false;
        
        return true;
    }

    GameResult Board::result() const
    {
        // Check for checkmate/stalemate
        std::vector<Move> moves = MoveGen::generateLegal(*this);

        if (moves.empty())
        {
            if (inCheck())
            {
                return (side == Color::White) ? GameResult::BlackWins : GameResult::WhiteWins;
            }
            return GameResult::Draw; // Stalemate
        }

        // Check for draws
        if (isDrawByRepetition(2) || isDrawByFiftyMoves() || isInsufficientMaterial())
        {
            return GameResult::Draw;
        }

        return GameResult::Ongoing;
    }

    std::string Board::toString() const
    {
        std::string result;
        result += "  +---+---+---+---+---+---+---+---+\n";

        for (int rank = 7; rank >= 0; rank--)
        {
            result += std::to_string(rank + 1) + " |";
            for (int file = 0; file < 8; file++)
            {
                Square sq = bb::makeSquare(file, rank);
                Piece p = pieces[sq];

                char c = ' ';
                if (p != Piece::None)
                {
                    const char chars[] = "PNBRQKpnbrqk";
                    c = chars[static_cast<int>(p)];
                }
                result += ' ';
                result += c;
                result += " |";
            }
            result += '\n';
            result += "  +---+---+---+---+---+---+---+---+\n";
        }
        result += "    a   b   c   d   e   f   g   h\n";
        result += "\n";
        result += "FEN: " + toFEN() + "\n";
        result += "Key: " + std::to_string(state->key) + "\n";

        return result;
    }

    void Board::print() const
    {
        std::cout << toString();
    }

    // Move Generation
    std::vector<Move> MoveGen::generateLegal(const Board &board)
    {
        std::vector<Move> moves = generatePseudoLegal(board);

        // Filter illegal moves
        moves.erase(std::remove_if(moves.begin(), moves.end(),
                                   [&board](const Move &m)
                                   { return !board.isLegal(m); }),
                    moves.end());

        return moves;
    }

    std::vector<Move> MoveGen::generatePseudoLegal(const Board &board)
    {
        std::vector<Move> moves;
        moves.reserve(256);

        Bitboard targets = ~board.piecesBB(board.sideToMove());

        size_t before, after;
        
        before = moves.size();
        generatePawnMoves(board, moves, targets);
        after = moves.size();
        // std::cout << "Pawns added " << (after - before) << " moves" << std::endl;
        
        before = moves.size();
        generateKnightMoves(board, moves, targets);
        after = moves.size();
        // std::cout << "Knights added " << (after - before) << " moves" << std::endl;
        
        before = moves.size();
        generateBishopMoves(board, moves, targets);
        after = moves.size();
        // std::cout << "Bishops added " << (after - before) << " moves" << std::endl;
        
        before = moves.size();
        generateRookMoves(board, moves, targets);
        after = moves.size();
        // std::cout << "Rooks added " << (after - before) << " moves" << std::endl;
        
        before = moves.size();
        generateQueenMoves(board, moves, targets);
        after = moves.size();
        // std::cout << "Queens added " << (after - before) << " moves" << std::endl;
        
        before = moves.size();
        generateKingMoves(board, moves, targets);
        after = moves.size();
        // std::cout << "Kings added " << (after - before) << " moves" << std::endl;
        
        before = moves.size();
        generateCastlingMoves(board, moves);
        after = moves.size();
        // std::cout << "Castling added " << (after - before) << " moves" << std::endl;

        return moves;
    }

    void MoveGen::generatePawnMoves(const Board &board, std::vector<Move> &moves, Bitboard targets)
    {
        Color us = board.sideToMove();
        Bitboard pawns = board.piecesBB(us, PieceType::Pawn);
        Bitboard enemies = board.piecesBB(~us);
        Bitboard empty = board.empty();

        int direction = (us == Color::White) ? 8 : -8;
        Bitboard rank3 = (us == Color::White) ? bb::RANK_3 : bb::RANK_6;
        Bitboard rank7 = (us == Color::White) ? bb::RANK_7 : bb::RANK_2;

        // Non-promotion pawns
        Bitboard nonPromo = pawns & ~rank7;

        // Single pushes
        Bitboard push1 = (us == Color::White) ? (nonPromo << 8) : (nonPromo >> 8);
        push1 &= empty;

        // Double pushes
        Bitboard push2 = (us == Color::White) ? ((push1 & rank3) << 8) : ((push1 & rank3) >> 8);
        push2 &= empty;

        while (push1)
        {
            Square to = bb::popLsb(push1);
            moves.emplace_back(to - direction, to, MoveType::Normal);
        }

        while (push2)
        {
            Square to = bb::popLsb(push2);
            moves.emplace_back(to - 2 * direction, to, MoveType::DoublePawnPush);
        }

        // Captures
        Bitboard captureL = (us == Color::White) ? ((nonPromo & ~bb::FILE_A) << 7) : ((nonPromo & ~bb::FILE_H) >> 7);
        Bitboard captureR = (us == Color::White) ? ((nonPromo & ~bb::FILE_H) << 9) : ((nonPromo & ~bb::FILE_A) >> 9);
        captureL &= enemies;
        captureR &= enemies;

        // For white: left shift means to = from + delta, so from = to - delta
        // For black: right shift means to = from - delta, so from = to + delta = to - (-delta)
        int deltaL = (us == Color::White) ? 7 : -7;
        int deltaR = (us == Color::White) ? 9 : -9;

        while (captureL)
        {
            Square to = bb::popLsb(captureL);
            // std::cout << "  Pawn capture L (side=" << (us == Color::White ? "White" : "Black") << "): from=" << static_cast<int>(to - deltaL) << " to=" << static_cast<int>(to) << std::endl;
            moves.emplace_back(to - deltaL, to, MoveType::Capture);
        }

        while (captureR)
        {
            Square to = bb::popLsb(captureR);
            // std::cout << "  Pawn capture R (side=" << (us == Color::White ? "White" : "Black") << "): from=" << static_cast<int>(to - deltaR) << " to=" << static_cast<int>(to) << std::endl;
            moves.emplace_back(to - deltaR, to, MoveType::Capture);
        }

        // Promotions
        Bitboard promoPawns = pawns & rank7;
        if (promoPawns)
        {
            Bitboard promoPush = (us == Color::White) ? (promoPawns << 8) : (promoPawns >> 8);
            promoPush &= empty;

            while (promoPush)
            {
                Square to = bb::popLsb(promoPush);
                addPromotions(to - direction, to, moves, false);
            }

            Bitboard promoCapL = (us == Color::White) ? ((promoPawns & ~bb::FILE_A) << 7) : ((promoPawns & ~bb::FILE_H) >> 7);
            Bitboard promoCapR = (us == Color::White) ? ((promoPawns & ~bb::FILE_H) << 9) : ((promoPawns & ~bb::FILE_A) >> 9);
            promoCapL &= enemies;
            promoCapR &= enemies;

            while (promoCapL)
            {
                Square to = bb::popLsb(promoCapL);
                addPromotions(to - deltaL, to, moves, true);
            }

            while (promoCapR)
            {
                Square to = bb::popLsb(promoCapR);
                addPromotions(to - deltaR, to, moves, true);
            }
        }

        // En passant
        Square epSq = board.enPassantSquare();
        if (epSq != NO_SQUARE)
        {
            Bitboard epAttackers = AttackTables::pawnAttacks(~us, epSq) & pawns;
            while (epAttackers)
            {
                Square from = bb::popLsb(epAttackers);
                moves.emplace_back(from, epSq, MoveType::EnPassant);
            }
        }
    }

    void MoveGen::addPromotions(Square from, Square to, std::vector<Move> &moves, bool isCapture)
    {
        if (isCapture)
        {
            moves.emplace_back(from, to, MoveType::QueenPromotionCapture);
            moves.emplace_back(from, to, MoveType::RookPromotionCapture);
            moves.emplace_back(from, to, MoveType::BishopPromotionCapture);
            moves.emplace_back(from, to, MoveType::KnightPromotionCapture);
        }
        else
        {
            moves.emplace_back(from, to, MoveType::QueenPromotion);
            moves.emplace_back(from, to, MoveType::RookPromotion);
            moves.emplace_back(from, to, MoveType::BishopPromotion);
            moves.emplace_back(from, to, MoveType::KnightPromotion);
        }
    }

    void MoveGen::generateKnightMoves(const Board &board, std::vector<Move> &moves, Bitboard targets)
    {
        Color us = board.sideToMove();
        Bitboard knights = board.piecesBB(us, PieceType::Knight);
        Bitboard enemies = board.piecesBB(~us);

        while (knights)
        {
            Square from = bb::popLsb(knights);
            Bitboard attacks = AttackTables::knightAttacks(from) & targets;

            while (attacks)
            {
                Square to = bb::popLsb(attacks);
                MoveType type = (bb::squareBB(to) & enemies) ? MoveType::Capture : MoveType::Normal;
                moves.emplace_back(from, to, type);
            }
        }
    }

    void MoveGen::generateBishopMoves(const Board &board, std::vector<Move> &moves, Bitboard targets)
    {
        Color us = board.sideToMove();
        Bitboard bishops = board.piecesBB(us, PieceType::Bishop);
        Bitboard enemies = board.piecesBB(~us);
        Bitboard occupied = board.occupied();

        while (bishops)
        {
            Square from = bb::popLsb(bishops);
            Bitboard attacks = AttackTables::bishopAttacks(from, occupied) & targets;

            while (attacks)
            {
                Square to = bb::popLsb(attacks);
                MoveType type = (bb::squareBB(to) & enemies) ? MoveType::Capture : MoveType::Normal;
                moves.emplace_back(from, to, type);
            }
        }
    }

    void MoveGen::generateRookMoves(const Board &board, std::vector<Move> &moves, Bitboard targets)
    {
        Color us = board.sideToMove();
        Bitboard rooks = board.piecesBB(us, PieceType::Rook);
        Bitboard enemies = board.piecesBB(~us);
        Bitboard occupied = board.occupied();

        // std::cout << "generateRookMoves: rooks=0x" << std::hex << rooks << std::dec << std::endl;

        while (rooks)
        {
            Square from = bb::popLsb(rooks);
            Bitboard attacks = AttackTables::rookAttacks(from, occupied) & targets;

            while (attacks)
            {
                Square to = bb::popLsb(attacks);
                MoveType type = (bb::squareBB(to) & enemies) ? MoveType::Capture : MoveType::Normal;
                // std::cout << "  Adding rook move from=" << static_cast<int>(from) << " to=" << static_cast<int>(to) << " type=" << static_cast<int>(type) << std::endl;
                moves.emplace_back(from, to, type);
            }
        }
    }

    void MoveGen::generateQueenMoves(const Board &board, std::vector<Move> &moves, Bitboard targets)
    {
        Color us = board.sideToMove();
        Bitboard queens = board.piecesBB(us, PieceType::Queen);
        Bitboard enemies = board.piecesBB(~us);
        Bitboard occupied = board.occupied();

        while (queens)
        {
            Square from = bb::popLsb(queens);
            Bitboard attacks = AttackTables::queenAttacks(from, occupied) & targets;

            while (attacks)
            {
                Square to = bb::popLsb(attacks);
                MoveType type = (bb::squareBB(to) & enemies) ? MoveType::Capture : MoveType::Normal;
                moves.emplace_back(from, to, type);
            }
        }
    }

    void MoveGen::generateKingMoves(const Board &board, std::vector<Move> &moves, Bitboard targets)
    {
        Color us = board.sideToMove();
        Square from = board.kingSquare(us);
        Bitboard enemies = board.piecesBB(~us);
        Bitboard attacks = AttackTables::kingAttacks(from) & targets;

        // std::cout << "generateKingMoves: from=" << static_cast<int>(from) << " attacks=0x" << std::hex << attacks << std::dec << std::endl;

        while (attacks)
        {
            Square to = bb::popLsb(attacks);
            MoveType type = (bb::squareBB(to) & enemies) ? MoveType::Capture : MoveType::Normal;
            // std::cout << "  Adding king move to=" << static_cast<int>(to) << " type=" << static_cast<int>(type) << std::endl;
            moves.emplace_back(from, to, type);
        }
    }

    void MoveGen::generateCastlingMoves(const Board &board, std::vector<Move> &moves)
    {
        Color us = board.sideToMove();
        Color them = ~us;
        CastlingRights rights = board.castlingRights();
        Bitboard occupied = board.occupied();

        if (us == Color::White)
        {
            // King-side castling (O-O)
            if ((rights & CastlingRights::WhiteKingSide) != CastlingRights::None)
            {
                // Check path is clear (F1, G1)
                if (!(occupied & (bb::squareBB(F1) | bb::squareBB(G1))))
                {
                    // Check king doesn't pass through or land in check (E1, F1, G1)
                    if (!board.isAttacked(E1, them) &&
                        !board.isAttacked(F1, them) &&
                        !board.isAttacked(G1, them))
                    {
                        moves.emplace_back(E1, G1, MoveType::KingSideCastle);
                    }
                }
            }
            // Queen-side castling (O-O-O)
            if ((rights & CastlingRights::WhiteQueenSide) != CastlingRights::None)
            {
                // Check path is clear (B1, C1, D1)
                if (!(occupied & (bb::squareBB(B1) | bb::squareBB(C1) | bb::squareBB(D1))))
                {
                    // Check king doesn't pass through or land in check (E1, D1, C1)
                    if (!board.isAttacked(E1, them) &&
                        !board.isAttacked(D1, them) &&
                        !board.isAttacked(C1, them))
                    {
                        moves.emplace_back(E1, C1, MoveType::QueenSideCastle);
                    }
                }
            }
        }
        else
        {
            // King-side castling (O-O)
            if ((rights & CastlingRights::BlackKingSide) != CastlingRights::None)
            {
                // Check path is clear (F8, G8)
                if (!(occupied & (bb::squareBB(F8) | bb::squareBB(G8))))
                {
                    // Check king doesn't pass through or land in check (E8, F8, G8)
                    if (!board.isAttacked(E8, them) &&
                        !board.isAttacked(F8, them) &&
                        !board.isAttacked(G8, them))
                    {
                        moves.emplace_back(E8, G8, MoveType::KingSideCastle);
                    }
                }
            }
            // Queen-side castling (O-O-O)
            if ((rights & CastlingRights::BlackQueenSide) != CastlingRights::None)
            {
                // Check path is clear (B8, C8, D8)
                if (!(occupied & (bb::squareBB(B8) | bb::squareBB(C8) | bb::squareBB(D8))))
                {
                    // Check king doesn't pass through or land in check (E8, D8, C8)
                    if (!board.isAttacked(E8, them) &&
                        !board.isAttacked(D8, them) &&
                        !board.isAttacked(C8, them))
                    {
                        moves.emplace_back(E8, C8, MoveType::QueenSideCastle);
                    }
                }
            }
        }
    }

    size_t MoveGen::perft(Board &board, int depth)
    {
        if (depth == 0)
            return 1;

        std::vector<Move> moves = generateLegal(board);

        if (depth == 1)
            return moves.size();

        size_t nodes = 0;

        for (const Move &m : moves)
        {
            StateInfo st;  // Create fresh StateInfo for each move
            board.makeMove(m, st);
            nodes += perft(board, depth - 1);
            board.unmakeMove(m);
        }

        return nodes;
    }

} // namespace chess
