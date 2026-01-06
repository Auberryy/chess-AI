#pragma once

#include "types.hpp"
#include <array>
#include <string>
#include <vector>
#include <sstream>

namespace chess
{

    // Pre-computed attack tables
    class AttackTables
    {
    public:
        static void init();

        static Bitboard pawnAttacks(Color c, Square sq);
        static Bitboard knightAttacks(Square sq);
        static Bitboard kingAttacks(Square sq);
        static Bitboard bishopAttacks(Square sq, Bitboard occupied);
        static Bitboard rookAttacks(Square sq, Bitboard occupied);
        static Bitboard queenAttacks(Square sq, Bitboard occupied);

        static Bitboard betweenBB(Square sq1, Square sq2);
        static Bitboard lineBB(Square sq1, Square sq2);

    private:
        static std::array<std::array<Bitboard, 64>, 2> pawnAttackTable;
        static std::array<Bitboard, 64> knightAttackTable;
        static std::array<Bitboard, 64> kingAttackTable;
        static std::array<std::array<Bitboard, 64>, 64> betweenTable;
        static std::array<std::array<Bitboard, 64>, 64> lineTable;

        // Magic bitboards for sliding pieces
        struct Magic
        {
            Bitboard mask;
            Bitboard magic;
            Bitboard *attacks;
            int shift;

            unsigned index(Bitboard occupied) const
            {
                return static_cast<unsigned>(((occupied & mask) * magic) >> shift);
            }
        };

        static std::array<Magic, 64> bishopMagics;
        static std::array<Magic, 64> rookMagics;
        static std::vector<Bitboard> bishopAttackTable;
        static std::vector<Bitboard> rookAttackTable;

        static void initPawnAttacks();
        static void initKnightAttacks();
        static void initKingAttacks();
        static void initMagics();
        static void initBetweenAndLine();
        static Bitboard slidingAttack(Square sq, Bitboard occupied, const int deltas[4]);
    };

    // State info for unmake move
    struct StateInfo
    {
        CastlingRights castling;
        Square epSquare;
        int halfmoveClock;
        int pliesFromNull;
        uint64_t key; // Zobrist hash
        Piece capturedPiece;
        StateInfo *previous;
    };

    class Board
    {
    public:
        Board();
        Board(const std::string &fen);
        Board(const Board& other);  // Copy constructor
        Board& operator=(const Board& other);  // Copy assignment
        Board(Board&& other) noexcept;  // Move constructor
        Board& operator=(Board&& other) noexcept;  // Move assignment

        // Setup
        void setStartingPosition();
        void setFromFEN(const std::string &fen);
        [[nodiscard]] std::string toFEN() const;

        // Piece access
        [[nodiscard]] Piece pieceAt(Square sq) const { return pieces[sq]; }
        [[nodiscard]] Bitboard piecesBB(PieceType pt) const { return byType[static_cast<int>(pt)]; }
        [[nodiscard]] Bitboard piecesBB(Color c) const { return byColor[static_cast<int>(c)]; }
        [[nodiscard]] Bitboard piecesBB(Color c, PieceType pt) const
        {
            return byColor[static_cast<int>(c)] & byType[static_cast<int>(pt)];
        }
        [[nodiscard]] Bitboard occupied() const { return byColor[0] | byColor[1]; }
        [[nodiscard]] Bitboard empty() const { return ~occupied(); }

        // State
        [[nodiscard]] Color sideToMove() const { return side; }
        [[nodiscard]] CastlingRights castlingRights() const { return state->castling; }
        [[nodiscard]] Square enPassantSquare() const { return state->epSquare; }
        [[nodiscard]] int halfmoveClock() const { return state->halfmoveClock; }
        [[nodiscard]] int fullmoveNumber() const { return fullmove; }
        [[nodiscard]] uint64_t key() const { return state->key; }

        // King position
        [[nodiscard]] Square kingSquare(Color c) const
        {
            return bb::lsb(piecesBB(c, PieceType::King));
        }

        // Attacks
        [[nodiscard]] Bitboard attackersTo(Square sq, Bitboard occupied) const;
        [[nodiscard]] Bitboard attackersTo(Square sq) const { return attackersTo(sq, occupied()); }
        [[nodiscard]] bool isAttacked(Square sq, Color by) const
        {
            return attackersTo(sq) & piecesBB(by);
        }
        [[nodiscard]] bool inCheck() const
        {
            return isAttacked(kingSquare(side), ~side);
        }

        // Move validation
        [[nodiscard]] bool isLegal(Move m) const;
        [[nodiscard]] bool isPseudoLegal(Move m) const;
        [[nodiscard]] bool givesCheck(Move m) const;

        // Make/unmake moves
        void makeMove(Move m, StateInfo &newState);
        void unmakeMove(Move m);
        void makeNullMove(StateInfo &newState);
        void unmakeNullMove();

        // Game state
        [[nodiscard]] bool isDrawByRepetition(int count = 1) const;
        [[nodiscard]] bool isDrawByFiftyMoves() const { return state->halfmoveClock >= 100; }
        [[nodiscard]] bool isInsufficientMaterial() const;
        [[nodiscard]] GameResult result() const;

        // Move generation helpers
        [[nodiscard]] Bitboard checkers() const;
        [[nodiscard]] Bitboard pinnedPieces(Color c) const;
        [[nodiscard]] Bitboard blockersForKing(Color c) const;

        // Display
        [[nodiscard]] std::string toString() const;
        void print() const;

    private:
        // Piece placement
        std::array<Piece, 64> pieces;
        std::array<Bitboard, 6> byType;
        std::array<Bitboard, 2> byColor;

        // State
        Color side;
        int fullmove;
        StateInfo rootState;
        StateInfo *state;

        // History for repetition detection
        std::vector<uint64_t> keyHistory;

        // Internal methods
        void putPiece(Piece p, Square sq);
        void removePiece(Square sq);
        void movePiece(Square from, Square to);
        void updateCastlingRights(Square from, Square to);
        void updateKey();

        // Zobrist hashing
        static std::array<std::array<uint64_t, 64>, 12> pieceKeys;
        static std::array<uint64_t, 16> castlingKeys;
        static std::array<uint64_t, 8> epFileKeys;
        static uint64_t sideKey;
        static bool zobristInitialized;
        static void initZobrist();
    };

    // Move generation
    class MoveGen
    {
    public:
        // Generate all legal moves
        static std::vector<Move> generateLegal(const Board &board);

        // Generate pseudo-legal moves (may leave king in check)
        static std::vector<Move> generatePseudoLegal(const Board &board);

        // Generate specific move types
        static std::vector<Move> generateCaptures(const Board &board);
        static std::vector<Move> generateQuiet(const Board &board);
        static std::vector<Move> generateEvasions(const Board &board);

        // Count legal moves (for perft testing)
        static size_t perft(Board &board, int depth);

    private:
        static void generatePawnMoves(const Board &board, std::vector<Move> &moves, Bitboard targets);
        static void generateKnightMoves(const Board &board, std::vector<Move> &moves, Bitboard targets);
        static void generateBishopMoves(const Board &board, std::vector<Move> &moves, Bitboard targets);
        static void generateRookMoves(const Board &board, std::vector<Move> &moves, Bitboard targets);
        static void generateQueenMoves(const Board &board, std::vector<Move> &moves, Bitboard targets);
        static void generateKingMoves(const Board &board, std::vector<Move> &moves, Bitboard targets);
        static void generateCastlingMoves(const Board &board, std::vector<Move> &moves);

        static void addPawnMoves(Square from, Bitboard targets, std::vector<Move> &moves,
                                 Color us, bool isCapture);
        static void addPromotions(Square from, Square to, std::vector<Move> &moves, bool isCapture);
    };

} // namespace chess
