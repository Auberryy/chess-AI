#include "training/data_generator.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace chess {
namespace training {

// ============================================================================
// Phase 1: Improved Random Self-Play Data Generator
// ============================================================================

GameRecord RandomDataGenerator::playRandomGame() {
    AttackTables::init();
    Board board;
    GameRecord record;
    StateInfo states[400];
    int ply = 0;
    
    while (ply < maxMoves_) {
        // Generate legal moves
        std::vector<Move> moves = MoveGen::generateLegal(board);
        
        if (moves.empty()) {
            // Checkmate or stalemate
            if (board.inCheck()) {
                // Checkmate - opponent wins
                record.result = (board.sideToMove() == Color::White) ? -1 : 1;
            } else {
                // Stalemate
                record.result = 0;
            }
            break;
        }
        
        // Create position sample
        PositionSample sample;
        sample.fen = board.toFEN();
        
        // Select move based on strategy
        Move move = selectMove(board, moves);
        
        // Calculate policy based on move scores
        sample.policy.resize(moves.size());
        float totalScore = 0;
        for (size_t i = 0; i < moves.size(); i++) {
            float score = std::exp(evaluateMove(board, moves[i]) / temperature_);
            sample.policy[i] = score;
            totalScore += score;
        }
        // Normalize
        for (size_t i = 0; i < moves.size(); i++) {
            sample.policy[i] /= totalScore;
        }
        
        board.makeMove(move, states[ply]);
        record.positions.push_back(sample);
        ply++;
        
        // Check for draw
        if (board.isDrawByRepetition() || board.isDrawByFiftyMoves()) {
            record.result = 0;
            break;
        }
    }
    
    if (ply >= maxMoves_) {
        record.result = 0;  // Draw by move limit
    }
    
    // Assign values based on result (from each side's perspective)
    for (size_t i = 0; i < record.positions.size(); i++) {
        bool whiteToMove = (i % 2 == 0);
        if (record.result == 1) {
            record.positions[i].value = whiteToMove ? 1.0f : -1.0f;
        } else if (record.result == -1) {
            record.positions[i].value = whiteToMove ? -1.0f : 1.0f;
        } else {
            record.positions[i].value = 0.0f;
        }
        record.positions[i].result = record.result;
    }
    
    record.numMoves = ply;
    return record;
}

Move RandomDataGenerator::selectMove(const Board& board, const std::vector<Move>& moves) {
    switch (strategy_) {
        case ExplorationStrategy::PureRandom:
            return selectRandomMove(moves);
        
        case ExplorationStrategy::WeightedCaptures:
        case ExplorationStrategy::WeightedTactical:
        case ExplorationStrategy::Softmax:
            return selectWeightedMove(board, moves);
        
        case ExplorationStrategy::SemiRandom:
            // 50% random, 50% weighted
            if (std::uniform_real_distribution<>(0, 1)(rng_) < 0.5f) {
                return selectRandomMove(moves);
            }
            return selectWeightedMove(board, moves);
        
        default:
            return selectRandomMove(moves);
    }
}

Move RandomDataGenerator::selectWeightedMove(const Board& board, const std::vector<Move>& moves) {
    std::vector<float> scores(moves.size());
    float totalScore = 0;
    
    for (size_t i = 0; i < moves.size(); i++) {
        float score = evaluateMove(board, moves[i]);
        scores[i] = std::exp(score / temperature_);
        totalScore += scores[i];
    }
    
    // Sample from distribution
    float r = std::uniform_real_distribution<float>(0, totalScore)(rng_);
    float cumulative = 0;
    
    for (size_t i = 0; i < moves.size(); i++) {
        cumulative += scores[i];
        if (r <= cumulative) {
            return moves[i];
        }
    }
    
    return moves.back();
}

float RandomDataGenerator::evaluateMove(const Board& board, const Move& move) {
    float score = 1.0f;  // Base score
    
    // Piece values for MVV-LVA
    constexpr float pieceValues[] = {0, 1, 3, 3, 5, 9, 100}; // None, Pawn, Knight, Bishop, Rook, Queen, King
    
    // Capture bonus (MVV-LVA style)
    Piece captured = board.pieceAt(move.to());
    if (captured != Piece::None) {
        PieceType capturedType = typeOf(captured);
        Piece attacker = board.pieceAt(move.from());
        PieceType attackerType = typeOf(attacker);
        
        float victimValue = pieceValues[static_cast<int>(capturedType)];
        float attackerValue = pieceValues[static_cast<int>(attackerType)];
        
        // MVV-LVA: prefer capturing high value with low value piece
        score += captureBonus_ * victimValue - 0.5f * attackerValue;
    }
    
    // Promotion bonus
    if (move.isPromotion()) {
        switch (move.promotionPiece()) {
            case PieceType::Queen: score += promotionBonus_; break;
            case PieceType::Rook: score += promotionBonus_ * 0.6f; break;
            case PieceType::Bishop: score += promotionBonus_ * 0.4f; break;
            case PieceType::Knight: score += promotionBonus_ * 0.4f; break;
            default: break;
        }
    }
    
    // Center control bonus for pawns and minor pieces
    Square to = move.to();
    int toFile = static_cast<int>(to) % 8;
    int toRank = static_cast<int>(to) / 8;
    
    // Distance from center (d4=27, d5=35, e4=28, e5=36)
    float centerDist = std::abs(toFile - 3.5f) + std::abs(toRank - 3.5f);
    score += centerBonus_ * (4.0f - centerDist) / 4.0f;
    
    return score;
}

Move RandomDataGenerator::selectRandomMove(const std::vector<Move>& moves) {
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    return moves[dist(rng_)];
}

std::vector<GameRecord> RandomDataGenerator::generateGames(int numGames) {
    std::vector<GameRecord> games;
    games.reserve(numGames);
    
    int decisiveGames = 0;
    std::cout << "Generating " << numGames << " games (strategy: ";
    switch (strategy_) {
        case ExplorationStrategy::PureRandom: std::cout << "PureRandom"; break;
        case ExplorationStrategy::WeightedCaptures: std::cout << "WeightedCaptures"; break;
        case ExplorationStrategy::WeightedTactical: std::cout << "WeightedTactical"; break;
        case ExplorationStrategy::SemiRandom: std::cout << "SemiRandom"; break;
        case ExplorationStrategy::Softmax: std::cout << "Softmax"; break;
    }
    std::cout << ")...\n";
    
    for (int i = 0; i < numGames; i++) {
        games.push_back(playRandomGame());
        if (games.back().result != 0) decisiveGames++;
        
        if ((i + 1) % 10 == 0) {
            std::cout << "  Generated " << (i + 1) << "/" << numGames 
                      << " (" << decisiveGames << " decisive)\r" << std::flush;
        }
    }
    std::cout << "\nDone! Total positions: " << 
        std::accumulate(games.begin(), games.end(), static_cast<size_t>(0),
            [](size_t sum, const GameRecord& g) { return sum + g.positions.size(); })
        << " (Decisive games: " << (100.0f * decisiveGames / numGames) << "%)\n";
    
    return games;
}

void RandomDataGenerator::saveToFile(const std::vector<GameRecord>& games, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    
    // Write header
    file << "fen,value,result\n";
    
    // Write all positions
    for (const auto& game : games) {
        for (const auto& pos : game.positions) {
            file << pos.fen << "," << pos.value << "," << pos.result << "\n";
        }
    }
    
    file.close();
    std::cout << "Saved to " << filename << "\n";
}

// ============================================================================
// Phase 2: Stockfish Data Generator
// ============================================================================

PositionSample StockfishDataGenerator::analyzePosition(const Board& board) {
    PositionSample sample;
    sample.fen = board.toFEN();  // Use toFEN() for proper FEN format
    
    // Query Stockfish
    auto [eval, bestMove] = queryStockfish(sample.fen);
    
    // Convert centipawn eval to [-1, 1]
    sample.value = std::tanh(eval / 400.0f);
    
    // For now, put high probability on best move
    std::vector<Move> moves = MoveGen::generateLegal(board);
    sample.policy.resize(moves.size(), 0.1f / moves.size());
    
    // Find best move and give it high probability
    for (size_t i = 0; i < moves.size(); i++) {
        if (moves[i].toUCI() == bestMove) {
            sample.policy[i] = 0.9f;
            break;
        }
    }
    
    return sample;
}

std::vector<PositionSample> StockfishDataGenerator::generateDataset(int numPositions) {
    std::vector<PositionSample> samples;
    samples.reserve(numPositions);
    
    std::cout << "Generating dataset with Stockfish...\n";
    
    for (int i = 0; i < numPositions; i++) {
        Board board = generateRandomPosition();
        samples.push_back(analyzePosition(board));
        
        if ((i + 1) % 100 == 0) {
            std::cout << "  Analyzed " << (i + 1) << "/" << numPositions << " positions\r" << std::flush;
        }
    }
    std::cout << "\nDone!\n";
    
    return samples;
}

void StockfishDataGenerator::saveDataset(const std::vector<PositionSample>& samples, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }
    
    file << "fen,value\n";
    for (const auto& sample : samples) {
        file << sample.fen << "," << sample.value << "\n";
    }
    
    file.close();
    std::cout << "Saved dataset to " << filename << "\n";
}

std::pair<float, std::string> StockfishDataGenerator::queryStockfish(const std::string& fen) {
    // TODO: Implement actual Stockfish communication via UCI protocol
    // For now, return placeholder
    return {0.0f, "e2e4"};
}

Board StockfishDataGenerator::generateRandomPosition() {
    // Start from initial position and make random moves
    AttackTables::init();
    Board board;
    StateInfo states[40];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> moveDist(5, 15);
    
    int numMoves = moveDist(gen);
    for (int i = 0; i < numMoves; i++) {
        std::vector<Move> moves = MoveGen::generateLegal(board);
        if (moves.empty()) break;
        
        std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
        board.makeMove(moves[dist(gen)], states[i]);
    }
    
    return board;
}

// ============================================================================
// Data Loader
// ============================================================================

std::vector<PositionSample> DataLoader::loadData() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename_ << "\n";
        return {};
    }
    
    data_.clear();
    std::string line;
    std::getline(file, line);  // Skip header
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string fen, valueStr, resultStr;
        
        std::getline(ss, fen, ',');
        std::getline(ss, valueStr, ',');
        std::getline(ss, resultStr, ',');
        
        PositionSample sample;
        sample.fen = fen;
        sample.value = std::stof(valueStr);
        if (!resultStr.empty()) {
            sample.result = std::stoi(resultStr);
        }
        
        data_.push_back(sample);
    }
    
    file.close();
    std::cout << "Loaded " << data_.size() << " positions from " << filename_ << "\n";
    return data_;
}

std::vector<PositionSample> DataLoader::getBatch() {
    if (data_.empty()) return {};
    
    std::vector<PositionSample> batch;
    batch.reserve(batchSize_);
    
    std::uniform_int_distribution<size_t> dist(0, data_.size() - 1);
    for (int i = 0; i < batchSize_; i++) {
        batch.push_back(data_[dist(rng_)]);
    }
    
    return batch;
}

void DataLoader::shuffle() {
    std::shuffle(data_.begin(), data_.end(), rng_);
}

} // namespace training
} // namespace chess
