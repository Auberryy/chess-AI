#include "nn/network.hpp"
#include "core/board.hpp"
#include <iostream>
#include <algorithm>
#include <random>

// Use chess:: types explicitly to avoid conflicts with forward declarations in nn namespace
using chess::Board;
using chess::Move;
using chess::MoveGen;
using chess::Piece;
using chess::PieceType;
using chess::MoveType;
using chess::Square;
using chess::Color;
using chess::CastlingRights;
using chess::GameResult;

namespace chess {
namespace nn {

// ============================================================================
// BoardEncoder
// ============================================================================

torch::Tensor BoardEncoder::encode(const chess::Board& board, int repetitions) {
    torch::Tensor input = torch::zeros({INPUT_CHANNELS, 8, 8});
    auto acc = input.accessor<float, 3>();
    
    // Encode piece positions (planes 0-11: 6 piece types x 2 colors)
    for (int sq = 0; sq < 64; sq++) {
        int rank = sq / 8;
        int file = sq % 8;
        
        chess::Piece piece = board.pieceAt(static_cast<chess::Square>(sq));
        if (piece != chess::Piece::None) {
            int pt = static_cast<int>(piece) % 6;  // Piece type (0-5)
            int c = static_cast<int>(piece) / 6;   // Color (0 or 1)
            int planeIdx = pt + c * 6;
            acc[planeIdx][rank][file] = 1.0f;
        }
    }
    
    // Side to move (plane 12)
    float stm = board.sideToMove() == Color::White ? 1.0f : 0.0f;
    for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
            acc[12][r][f] = stm;
        }
    }
    
    // Castling rights (planes 13-16)
    auto cr = board.castlingRights();
    for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
            acc[13][r][f] = (static_cast<int>(cr) & static_cast<int>(CastlingRights::WhiteKingSide)) ? 1.0f : 0.0f;
            acc[14][r][f] = (static_cast<int>(cr) & static_cast<int>(CastlingRights::WhiteQueenSide)) ? 1.0f : 0.0f;
            acc[15][r][f] = (static_cast<int>(cr) & static_cast<int>(CastlingRights::BlackKingSide)) ? 1.0f : 0.0f;
            acc[16][r][f] = (static_cast<int>(cr) & static_cast<int>(CastlingRights::BlackQueenSide)) ? 1.0f : 0.0f;
        }
    }
    
    // En passant (plane 17)
    Square ep = board.enPassantSquare();
    if (ep != NO_SQUARE) {
        int rank = ep / 8;
        int file = ep % 8;
        acc[17][rank][file] = 1.0f;
    }
    
    // Repetition count (planes 18-19)
    for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
            acc[18][r][f] = (repetitions >= 1) ? 1.0f : 0.0f;
            acc[19][r][f] = (repetitions >= 2) ? 1.0f : 0.0f;
        }
    }
    
    // Halfmove clock normalized (plane 20)
    float halfmove = std::min(board.halfmoveClock() / 100.0f, 1.0f);
    for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
            acc[20][r][f] = halfmove;
        }
    }
    
    // Fullmove number normalized (plane 21)
    float fullmove = std::min(board.fullmoveNumber() / 200.0f, 1.0f);
    for (int r = 0; r < 8; r++) {
        for (int f = 0; f < 8; f++) {
            acc[21][r][f] = fullmove;
        }
    }
    
    return input;
}

torch::Tensor BoardEncoder::encodeBatch(const std::vector<chess::Board>& boards) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(boards.size());
    
    for (const auto& board : boards) {
        tensors.push_back(encode(board));
    }
    
    return torch::stack(tensors);
}

int BoardEncoder::moveToIndex(const chess::Move& move) {
    // Policy encoding: 4672 possible moves
    // From square (64) x Move type (73)
    // Move types: 56 queen moves (7 distances x 8 directions) + 8 knight moves + 9 underpromotions
    
    if (move == NULL_MOVE) return -1;
    
    Square from = move.from();
    Square to = move.to();
    
    int fromRank = from / 8;
    int fromFile = from % 8;
    int toRank = to / 8;
    int toFile = to % 8;
    
    int dr = toRank - fromRank;
    int df = toFile - fromFile;
    
    int moveType = 0;
    
    // Check if it's a knight move
    if ((std::abs(dr) == 2 && std::abs(df) == 1) || (std::abs(dr) == 1 && std::abs(df) == 2)) {
        // Knight move encoding
        // 8 possible knight moves
        static const int knightDelta[8][2] = {
            {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
            {1, -2}, {1, 2}, {2, -1}, {2, 1}
        };
        for (int i = 0; i < 8; i++) {
            if (dr == knightDelta[i][0] && df == knightDelta[i][1]) {
                moveType = 56 + i;  // After queen moves
                break;
            }
        }
    } else {
        // Queen move encoding: direction (8) x distance (7)
        int direction = -1;
        int distance = std::max(std::abs(dr), std::abs(df));
        
        // Normalize direction
        int ddr = (dr > 0) - (dr < 0);
        int ddf = (df > 0) - (df < 0);
        
        // Direction encoding: N, NE, E, SE, S, SW, W, NW
        static const int dirDelta[8][2] = {
            {1, 0}, {1, 1}, {0, 1}, {-1, 1},
            {-1, 0}, {-1, -1}, {0, -1}, {1, -1}
        };
        for (int i = 0; i < 8; i++) {
            if (ddr == dirDelta[i][0] && ddf == dirDelta[i][1]) {
                direction = i;
                break;
            }
        }
        
        if (direction >= 0 && distance >= 1 && distance <= 7) {
            moveType = direction * 7 + (distance - 1);
        }
    }
    
    // Handle underpromotions (knight, bishop, rook)
    if (move.isPromotion()) {
        PieceType promo = move.promotionPiece();
        if (promo == PieceType::Knight || promo == PieceType::Bishop || promo == PieceType::Rook) {
            int promoIdx = (promo == PieceType::Knight) ? 0 : (promo == PieceType::Bishop) ? 1 : 2;
            // 3 underpromotion types x 3 directions (left, straight, right)
            int direction = (df == 0) ? 1 : (df > 0) ? 2 : 0;
            moveType = 64 + promoIdx * 3 + direction;
        }
    }
    
    return from * 73 + moveType;
}

chess::Move BoardEncoder::indexToMove(int index, const chess::Board& board) {
    if (index < 0 || index >= POLICY_OUTPUT_SIZE) {
        return chess::NULL_MOVE;
    }
    
    int from = index / 73;
    int moveType = index % 73;
    
    int fromRank = from / 8;
    int fromFile = from % 8;
    
    int toRank, toFile;
    PieceType promo = PieceType::None;
    
    if (moveType < 56) {
        // Queen move
        int direction = moveType / 7;
        int distance = (moveType % 7) + 1;
        
        static const int dirDelta[8][2] = {
            {1, 0}, {1, 1}, {0, 1}, {-1, 1},
            {-1, 0}, {-1, -1}, {0, -1}, {1, -1}
        };
        
        toRank = fromRank + dirDelta[direction][0] * distance;
        toFile = fromFile + dirDelta[direction][1] * distance;
    } else if (moveType < 64) {
        // Knight move
        static const int knightDelta[8][2] = {
            {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
            {1, -2}, {1, 2}, {2, -1}, {2, 1}
        };
        int idx = moveType - 56;
        toRank = fromRank + knightDelta[idx][0];
        toFile = fromFile + knightDelta[idx][1];
    } else {
        // Underpromotion
        int promoIdx = (moveType - 64) / 3;
        int direction = (moveType - 64) % 3;
        
        promo = (promoIdx == 0) ? PieceType::Knight : (promoIdx == 1) ? PieceType::Bishop : PieceType::Rook;
        
        toRank = (board.sideToMove() == Color::White) ? 7 : 0;
        toFile = fromFile + (direction - 1);  // -1, 0, or +1
    }
    
    if (toRank < 0 || toRank > 7 || toFile < 0 || toFile > 7) {
        return chess::NULL_MOVE;
    }

    chess::Square fromSq = static_cast<chess::Square>(fromRank * 8 + fromFile);
    chess::Square toSq = static_cast<chess::Square>(toRank * 8 + toFile);
    
    // Check for promotion
    chess::Piece piece = board.pieceAt(fromSq);
    int pt = static_cast<int>(piece) % 6;
    if (pt == static_cast<int>(chess::PieceType::Pawn) && (toRank == 0 || toRank == 7)) {
        if (promo == chess::PieceType::None) {
            // Default to queen promotion
            return chess::Move(fromSq, toSq, chess::MoveType::QueenPromotion);
        }
        // Convert promo to MoveType
        chess::MoveType mt = (promo == chess::PieceType::Knight) ? chess::MoveType::KnightPromotion :
                      (promo == chess::PieceType::Bishop) ? chess::MoveType::BishopPromotion :
                      chess::MoveType::RookPromotion;
        return chess::Move(fromSq, toSq, mt);
    }
    
    return chess::Move(fromSq, toSq);
}

torch::Tensor BoardEncoder::getLegalMoveMask(const chess::Board& board) {
    torch::Tensor mask = torch::zeros({POLICY_OUTPUT_SIZE});
    auto acc = mask.accessor<float, 1>();
    
    auto moves = chess::MoveGen::generateLegal(board);
    for (const chess::Move& move : moves) {
        int idx = moveToIndex(move);
        if (idx >= 0 && idx < POLICY_OUTPUT_SIZE) {
            acc[idx] = 1.0f;
        }
    }
    
    return mask;
}

// ============================================================================
// NeuralNetwork
// ============================================================================

NeuralNetwork::NeuralNetwork(const std::string& modelPath, int gpuId, 
                             float gpuMemoryFraction, bool useAmp)
    : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, gpuId),
      useAmp_(useAmp),
      gpuMemoryFraction_(gpuMemoryFraction) {
    
    // Create model
    model_ = ChessNet(NUM_FILTERS, NUM_RESIDUAL_BLOCKS);
    model_->to(device_);
    
    // Load weights if provided
    if (!modelPath.empty()) {
        load(modelPath);
    }
    
    // Create optimizer
    optimizer_ = std::make_unique<torch::optim::Adam>(
        model_->parameters(),
        torch::optim::AdamOptions(0.002).weight_decay(1e-4)
    );
    
    std::cout << "NeuralNetwork initialized on " 
              << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
    
    // Count parameters
    int64_t numParams = 0;
    for (const auto& p : model_->parameters()) {
        numParams += p.numel();
    }
    std::cout << "  Parameters: " << numParams << std::endl;
}

NeuralNetwork::~NeuralNetwork() = default;

std::pair<std::vector<float>, float> NeuralNetwork::evaluate(const chess::Board& board) {
    torch::NoGradGuard noGrad;
    model_->eval();
    
    // Encode board
    torch::Tensor input = BoardEncoder::encode(board).unsqueeze(0).to(device_);
    
    // Forward pass
    auto [policyLogits, value] = model_->forward(input);
    
    // Get legal move mask
    torch::Tensor mask = BoardEncoder::getLegalMoveMask(board).to(device_);
    
    // Apply mask (set illegal moves to -infinity)
    policyLogits = policyLogits.squeeze(0);
    policyLogits = policyLogits.masked_fill(mask == 0, -1e9f);
    
    // Softmax
    torch::Tensor policy = torch::softmax(policyLogits, 0);
    
    // Convert to vectors
    policy = policy.cpu();
    value = value.cpu();
    
    std::vector<float> policyVec(policy.data_ptr<float>(), 
                                  policy.data_ptr<float>() + policy.numel());
    float valueScalar = value.item<float>();
    
    return {policyVec, valueScalar};
}

std::pair<std::vector<std::vector<float>>, std::vector<float>> 
NeuralNetwork::evaluateBatch(const std::vector<chess::Board>& boards) {
    if (boards.empty()) {
        return {{}, {}};
    }
    
    torch::NoGradGuard noGrad;
    model_->eval();
    
    // Encode batch
    torch::Tensor input = BoardEncoder::encodeBatch(boards).to(device_);
    
    // Forward pass
    auto [policyLogits, values] = model_->forward(input);
    
    // Convert to vectors
    policyLogits = policyLogits.cpu();
    values = values.cpu();
    
    std::vector<std::vector<float>> policies;
    std::vector<float> valueVec;
    
    for (size_t i = 0; i < boards.size(); i++) {
        torch::Tensor mask = BoardEncoder::getLegalMoveMask(boards[i]);
        torch::Tensor policyRow = policyLogits[i].masked_fill(mask == 0, -1e9f);
        torch::Tensor policy = torch::softmax(policyRow, 0);
        
        policies.emplace_back(policy.data_ptr<float>(), 
                              policy.data_ptr<float>() + policy.numel());
        valueVec.push_back(values[i].item<float>());
    }
    
    return {policies, valueVec};
}

float NeuralNetwork::trainBatch(torch::Tensor inputBatch, torch::Tensor policyBatch,
                                torch::Tensor valueBatch, float learningRate) {
    model_->train();
    
    // Update learning rate
    for (auto& group : optimizer_->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(learningRate);
    }
    
    // Move to device
    inputBatch = inputBatch.to(device_);
    policyBatch = policyBatch.to(device_);
    valueBatch = valueBatch.to(device_).view({-1, 1});
    
    // Forward pass
    auto [policyLogits, valuePred] = model_->forward(inputBatch);
    
    // Policy loss (cross entropy)
    torch::Tensor policyLoss = -torch::sum(
        policyBatch * torch::log_softmax(policyLogits, 1)
    ) / inputBatch.size(0);
    
    // Value loss (MSE)
    torch::Tensor valueLoss = torch::mse_loss(valuePred, valueBatch);
    
    // Total loss
    torch::Tensor loss = policyLoss + valueLoss;
    
    // Backward pass
    optimizer_->zero_grad();
    loss.backward();
    
    // Gradient clipping
    torch::nn::utils::clip_grad_norm_(model_->parameters(), 1.0);
    
    // Update weights
    optimizer_->step();
    
    return loss.item<float>();
}

void NeuralNetwork::save(const std::string& path) {
    torch::save(model_, path);
    std::cout << "Model saved to " << path << std::endl;
}

void NeuralNetwork::load(const std::string& path) {
    try {
        torch::load(model_, path);
        model_->to(device_);
        std::cout << "Model loaded from " << path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
    }
}

size_t NeuralNetwork::getGpuMemoryUsed() const {
    if (!device_.is_cuda()) return 0;
    // This would require CUDA runtime API calls
    return 0;
}

size_t NeuralNetwork::getGpuMemoryTotal() const {
    if (!device_.is_cuda()) return 0;
    // This would require CUDA runtime API calls
    return 0;
}

} // namespace nn
} // namespace chess
