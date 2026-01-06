#include "nn/network.hpp"
#include "core/board.hpp"
#include <torch/cuda.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <iostream>
#include <algorithm>

namespace chess
{
    namespace nn
    {

        // Move encoding tables
        // Each square can have up to 73 move types:
        // - 56 queen-like moves (7 directions * 7 distances + diagonal 7)
        // - 8 knight moves
        // - 9 underpromotions (3 pieces * 3 directions)

        namespace
        {
            // Direction offsets for queen moves
            constexpr int QUEEN_DIRECTIONS[8][2] = {
                {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};

            // Knight move offsets
            constexpr int KNIGHT_OFFSETS[8][2] = {
                {2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}};

            int encodeQueenMove(int fromFile, int fromRank, int toFile, int toRank)
            {
                int df = toFile - fromFile;
                int dr = toRank - fromRank;

                // Normalize direction
                int dirIdx = -1;
                int distance = std::max(std::abs(df), std::abs(dr));

                int signF = (df > 0) ? 1 : ((df < 0) ? -1 : 0);
                int signR = (dr > 0) ? 1 : ((dr < 0) ? -1 : 0);

                for (int i = 0; i < 8; i++)
                {
                    if (QUEEN_DIRECTIONS[i][0] == signF && QUEEN_DIRECTIONS[i][1] == signR)
                    {
                        dirIdx = i;
                        break;
                    }
                }

                if (dirIdx == -1 || distance == 0)
                    return -1;

                return dirIdx * 7 + (distance - 1); // 0-55
            }

            int encodeKnightMove(int fromFile, int fromRank, int toFile, int toRank)
            {
                int df = toFile - fromFile;
                int dr = toRank - fromRank;

                for (int i = 0; i < 8; i++)
                {
                    if (KNIGHT_OFFSETS[i][0] == df && KNIGHT_OFFSETS[i][1] == dr)
                    {
                        return 56 + i; // 56-63
                    }
                }
                return -1;
            }

            int encodeUnderpromotion(int fromFile, int toFile, PieceType promo)
            {
                // 3 directions: left capture, forward, right capture
                int direction = (toFile - fromFile) + 1; // -1, 0, 1 -> 0, 1, 2

                int pieceIdx;
                switch (promo)
                {
                case PieceType::Knight:
                    pieceIdx = 0;
                    break;
                case PieceType::Bishop:
                    pieceIdx = 1;
                    break;
                case PieceType::Rook:
                    pieceIdx = 2;
                    break;
                default:
                    return -1;
                }

                return 64 + direction * 3 + pieceIdx; // 64-72
            }
        }

        torch::Tensor BoardEncoder::encode(const Board &board, int repetitions)
        {
            auto tensor = torch::zeros({INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE});
            auto accessor = tensor.accessor<float, 3>();

            Color us = board.sideToMove();
            Color them = ~us;

            // Piece planes (from current player's perspective)
            for (int sq = 0; sq < 64; sq++)
            {
                int file = bb::fileOf(sq);
                int rank = bb::rankOf(sq);

                // Flip board for black
                int r = (us == Color::White) ? rank : (7 - rank);
                int f = (us == Color::White) ? file : (7 - file);

                Piece p = board.pieceAt(sq);
                if (p == Piece::None)
                    continue;

                Color pieceColor = colorOf(p);
                PieceType pieceType = typeOf(p);

                int plane;
                if (pieceColor == us)
                {
                    plane = static_cast<int>(pieceType);
                }
                else
                {
                    plane = 6 + static_cast<int>(pieceType);
                }

                accessor[plane][r][f] = 1.0f;
            }

            // Repetition planes
            if (repetitions >= 1)
            {
                tensor.slice(0, 12, 13).fill_(1.0f);
            }
            if (repetitions >= 2)
            {
                tensor.slice(0, 13, 14).fill_(1.0f);
            }

            // Side to move (always 1 since we flip for black)
            tensor.slice(0, 14, 15).fill_(1.0f);

            // Castling rights
            CastlingRights rights = board.castlingRights();
            CastlingRights ourKS = (us == Color::White) ? CastlingRights::WhiteKingSide : CastlingRights::BlackKingSide;
            CastlingRights ourQS = (us == Color::White) ? CastlingRights::WhiteQueenSide : CastlingRights::BlackQueenSide;
            CastlingRights theirKS = (us == Color::White) ? CastlingRights::BlackKingSide : CastlingRights::WhiteKingSide;
            CastlingRights theirQS = (us == Color::White) ? CastlingRights::BlackQueenSide : CastlingRights::WhiteQueenSide;

            if ((rights & ourKS) != CastlingRights::None)
                tensor.slice(0, 15, 16).fill_(1.0f);
            if ((rights & ourQS) != CastlingRights::None)
                tensor.slice(0, 16, 17).fill_(1.0f);
            if ((rights & theirKS) != CastlingRights::None)
                tensor.slice(0, 17, 18).fill_(1.0f);
            if ((rights & theirQS) != CastlingRights::None)
                tensor.slice(0, 18, 19).fill_(1.0f);

            // En passant
            Square epSq = board.enPassantSquare();
            if (epSq != NO_SQUARE)
            {
                int file = bb::fileOf(epSq);
                int rank = bb::rankOf(epSq);
                int r = (us == Color::White) ? rank : (7 - rank);
                int f = (us == Color::White) ? file : (7 - file);
                accessor[19][r][f] = 1.0f;
            }

            // Halfmove clock (normalized to [0, 1])
            float halfmoveNorm = std::min(board.halfmoveClock(), 100) / 100.0f;
            tensor.slice(0, 20, 21).fill_(halfmoveNorm);

            // Fullmove number (normalized)
            float fullmoveNorm = std::min(board.fullmoveNumber(), 500) / 500.0f;
            tensor.slice(0, 21, 22).fill_(fullmoveNorm);

            return tensor;
        }

        torch::Tensor BoardEncoder::encodeBatch(const std::vector<Board> &boards)
        {
            std::vector<torch::Tensor> tensors;
            tensors.reserve(boards.size());

            for (const auto &board : boards)
            {
                tensors.push_back(encode(board));
            }

            return torch::stack(tensors);
        }

        int BoardEncoder::moveToIndex(const Move &move)
        {
            Square from = move.from();
            Square to = move.to();
            int fromFile = bb::fileOf(from);
            int fromRank = bb::rankOf(from);
            int toFile = bb::fileOf(to);
            int toRank = bb::rankOf(to);

            int moveType;

            // Check for underpromotion
            if (move.isPromotion())
            {
                PieceType promo = move.promotionPiece();
                if (promo != PieceType::Queen)
                {
                    moveType = encodeUnderpromotion(fromFile, toFile, promo);
                }
                else
                {
                    // Queen promotion encoded as queen move
                    moveType = encodeQueenMove(fromFile, fromRank, toFile, toRank);
                }
            }
            else
            {
                // Try queen move first
                moveType = encodeQueenMove(fromFile, fromRank, toFile, toRank);
                if (moveType == -1)
                {
                    // Must be knight move
                    moveType = encodeKnightMove(fromFile, fromRank, toFile, toRank);
                }
            }

            if (moveType == -1)
                return -1;

            return from * 73 + moveType;
        }

        Move BoardEncoder::indexToMove(int index, const Board &board)
        {
            int fromSq = index / 73;
            int moveType = index % 73;

            int fromFile = bb::fileOf(fromSq);
            int fromRank = bb::rankOf(fromSq);

            int toFile, toRank;
            MoveType mt = MoveType::Normal;

            if (moveType < 56)
            {
                // Queen move
                int dir = moveType / 7;
                int dist = (moveType % 7) + 1;

                toFile = fromFile + QUEEN_DIRECTIONS[dir][0] * dist;
                toRank = fromRank + QUEEN_DIRECTIONS[dir][1] * dist;
            }
            else if (moveType < 64)
            {
                // Knight move
                int knightIdx = moveType - 56;
                toFile = fromFile + KNIGHT_OFFSETS[knightIdx][0];
                toRank = fromRank + KNIGHT_OFFSETS[knightIdx][1];
            }
            else
            {
                // Underpromotion
                int promoIdx = moveType - 64;
                int direction = promoIdx / 3 - 1;
                int pieceIdx = promoIdx % 3;

                toFile = fromFile + direction;
                toRank = (board.sideToMove() == Color::White) ? 7 : 0;

                switch (pieceIdx)
                {
                case 0:
                    mt = MoveType::KnightPromotion;
                    break;
                case 1:
                    mt = MoveType::BishopPromotion;
                    break;
                case 2:
                    mt = MoveType::RookPromotion;
                    break;
                }
            }

            if (toFile < 0 || toFile > 7 || toRank < 0 || toRank > 7)
            {
                return NULL_MOVE;
            }

            Square to = bb::makeSquare(toFile, toRank);
            return Move(fromSq, to, mt);
        }

        torch::Tensor BoardEncoder::getLegalMoveMask(const Board &board)
        {
            auto mask = torch::zeros({POLICY_OUTPUT_SIZE});
            auto accessor = mask.accessor<float, 1>();

            auto moves = MoveGen::generateLegal(board);

            for (const auto &move : moves)
            {
                int idx = moveToIndex(move);
                if (idx >= 0 && idx < POLICY_OUTPUT_SIZE)
                {
                    accessor[idx] = 1.0f;
                }
            }

            return mask;
        }

        // NeuralNetwork implementation
        NeuralNetwork::NeuralNetwork(const std::string &modelPath, int gpuId, float gpuMemoryFraction, bool useAmp)
            : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
              gpuMemoryFraction_(gpuMemoryFraction),
              useAmp_(useAmp)
        {

            if (torch::cuda::is_available())
            {
                // Set GPU device
                device_ = torch::Device(torch::kCUDA, gpuId);

                // Configure memory fraction
                // Note: LibTorch doesn't have direct memory fraction API like TensorFlow
                // We'll set environment variable before initialization
                std::string memFraction = std::to_string(gpuMemoryFraction);

                std::cout << "Using GPU: " << torch::cuda::device_count() << " available" << std::endl;
                std::cout << "GPU Memory Fraction: " << gpuMemoryFraction * 100 << "%" << std::endl;

                // Enable TF32 for Ampere GPUs (RTX 3090) - ~8x matmul speedup, no code changes needed
                torch::globalContext().setAllowTF32CuBLAS(true);
                torch::globalContext().setAllowTF32CuDNN(true);
                std::cout << "TF32 enabled for cuBLAS and cuDNN (Ampere optimization)" << std::endl;

                // Enable cuDNN benchmarking for best performance
                torch::globalContext().setBenchmarkCuDNN(true);
                std::cout << "cuDNN auto-tuner enabled" << std::endl;

                // AMP (FP16 mixed precision) - ~2x speedup + 2x memory savings
                if (useAmp_)
                {
                    std::cout << "AMP (Automatic Mixed Precision) enabled - using FP16 for compute" << std::endl;
                }
            }
            else
            {
                std::cout << "CUDA not available, using CPU" << std::endl;
                useAmp_ = false; // Disable AMP on CPU
            }

            // Create model
            model_ = ChessNet(NUM_FILTERS, NUM_RESIDUAL_BLOCKS);
            model_->to(device_);

            // Load weights if path provided
            if (!modelPath.empty())
            {
                load(modelPath);
            }

            // Initialize optimizer
            optimizer_ = std::make_unique<torch::optim::Adam>(
                model_->parameters(), torch::optim::AdamOptions(0.001));
        }

        NeuralNetwork::~NeuralNetwork() = default;

        std::pair<std::vector<float>, float> NeuralNetwork::evaluate(const Board &board)
        {
            torch::NoGradGuard no_grad;
            model_->eval();

            auto input = BoardEncoder::encode(board).unsqueeze(0).to(device_);
            auto [policy, value] = model_->forward(input);

            // Apply softmax to policy
            auto legalMask = BoardEncoder::getLegalMoveMask(board).to(device_);
            policy = policy.squeeze(0);

            // Mask illegal moves
            policy = torch::where(legalMask > 0, policy, torch::full_like(policy, -1e9));
            policy = torch::softmax(policy, 0);

            // Convert to vector
            auto policyData = policy.to(torch::kCPU);
            std::vector<float> policyVec(policyData.data_ptr<float>(),
                                         policyData.data_ptr<float>() + policyData.numel());

            float valueScalar = value.item<float>();

            return {policyVec, valueScalar};
        }

        std::pair<std::vector<std::vector<float>>, std::vector<float>>
        NeuralNetwork::evaluateBatch(const std::vector<Board> &boards)
        {
            torch::NoGradGuard no_grad;
            model_->eval();

            auto input = BoardEncoder::encodeBatch(boards).to(device_);
            auto [policies, values] = model_->forward(input);

            // Process each board's policy
            std::vector<std::vector<float>> policyVecs;
            std::vector<float> valueVecs;

            auto policiesCpu = policies.to(torch::kCPU);
            auto valuesCpu = values.squeeze().to(torch::kCPU);

            for (size_t i = 0; i < boards.size(); i++)
            {
                auto legalMask = BoardEncoder::getLegalMoveMask(boards[i]);
                auto policy = policiesCpu[i];

                policy = torch::where(legalMask > 0, policy, torch::full_like(policy, -1e9));
                policy = torch::softmax(policy, 0);

                policyVecs.emplace_back(policy.data_ptr<float>(),
                                        policy.data_ptr<float>() + policy.numel());

                valueVecs.push_back(valuesCpu[i].item<float>());
            }

            return {policyVecs, valueVecs};
        }

        void NeuralNetwork::train(
            const std::vector<std::tuple<Board, std::vector<float>, float>> &samples,
            int batchSize, float learningRate)
        {

            model_->train();

            // Update learning rate
            for (auto &group : optimizer_->param_groups())
            {
                static_cast<torch::optim::AdamOptions &>(group.options()).lr(learningRate);
            }

            // Shuffle indices
            std::vector<size_t> indices(samples.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());

            float totalLoss = 0;
            int numBatches = 0;

            for (size_t i = 0; i < samples.size(); i += batchSize)
            {
                size_t batchEnd = std::min(i + batchSize, samples.size());

                // Prepare batch
                std::vector<torch::Tensor> inputs;
                std::vector<torch::Tensor> targetPolicies;
                std::vector<float> targetValues;

                for (size_t j = i; j < batchEnd; j++)
                {
                    const auto &[board, policy, value] = samples[indices[j]];
                    inputs.push_back(BoardEncoder::encode(board));
                    targetPolicies.push_back(torch::tensor(policy));
                    targetValues.push_back(value);
                }

                auto inputBatch = torch::stack(inputs).to(device_);
                auto policyTarget = torch::stack(targetPolicies).to(device_);
                auto valueTarget = torch::tensor(targetValues).unsqueeze(1).to(device_);

                // Forward pass with optional AMP
                optimizer_->zero_grad();

                torch::Tensor policy, value, loss;

                if (useAmp_ && device_.is_cuda())
                {
                    // AMP autocast - use FP16 for compute-intensive ops
                    auto autocast_guard = torch::cuda::amp::autocast(true);
                    std::tie(policy, value) = model_->forward(inputBatch);

                    // Loss calculation
                    auto policyLoss = -torch::sum(policyTarget * torch::log_softmax(policy, 1)) /
                                      static_cast<float>(batchEnd - i);
                    auto valueLoss = torch::mse_loss(value, valueTarget);
                    loss = policyLoss + valueLoss;
                }
                else
                {
                    // Regular FP32 training
                    std::tie(policy, value) = model_->forward(inputBatch);

                    // Loss calculation
                    auto policyLoss = -torch::sum(policyTarget * torch::log_softmax(policy, 1)) /
                                      static_cast<float>(batchEnd - i);
                    auto valueLoss = torch::mse_loss(value, valueTarget);
                    loss = policyLoss + valueLoss;
                }

                // Backward pass
                loss.backward();
                optimizer_->step();

                totalLoss += loss.item<float>();
                numBatches++;
            }

            std::cout << "Training loss: " << totalLoss / numBatches << std::endl;
        }

        float NeuralNetwork::trainBatch(torch::Tensor inputBatch, torch::Tensor policyBatch,
                                        torch::Tensor valueBatch, float learningRate)
        {
            model_->train();

            // Update learning rate
            for (auto &group : optimizer_->param_groups())
            {
                static_cast<torch::optim::AdamOptions &>(group.options()).lr(learningRate);
            }

            // Move to device
            inputBatch = inputBatch.to(device_);
            policyBatch = policyBatch.to(device_);
            valueBatch = valueBatch.to(device_);

            // Forward pass with optional AMP
            optimizer_->zero_grad();

            torch::Tensor policy, value, loss;

            if (useAmp_ && device_.is_cuda())
            {
                // AMP autocast - use FP16 for compute-intensive ops
                auto autocast_guard = torch::cuda::amp::autocast(true);
                std::tie(policy, value) = model_->forward(inputBatch);

                // Loss calculation
                auto policyLoss = -torch::sum(policyBatch * torch::log_softmax(policy, 1)) / inputBatch.size(0);
                auto valueLoss = torch::mse_loss(value, valueBatch);
                loss = policyLoss + valueLoss;
            }
            else
            {
                // Regular FP32 training
                std::tie(policy, value) = model_->forward(inputBatch);

                // Loss calculation
                auto policyLoss = -torch::sum(policyBatch * torch::log_softmax(policy, 1)) / inputBatch.size(0);
                auto valueLoss = torch::mse_loss(value, valueBatch);
                loss = policyLoss + valueLoss;
            }

            // Backward pass
            loss.backward();
            optimizer_->step();

            return loss.item<float>();
        }

        void NeuralNetwork::save(const std::string &path)
        {
            torch::save(model_, path);
        }

        void NeuralNetwork::load(const std::string &path)
        {
            torch::load(model_, path);
            model_->to(device_);
        }

        size_t NeuralNetwork::getGpuMemoryUsed() const
        {
            if (!torch::cuda::is_available())
                return 0;

            // Get allocated memory
            return c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[0].current;
        }

        size_t NeuralNetwork::getGpuMemoryTotal() const
        {
            if (!torch::cuda::is_available())
                return 0;

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            return prop.totalGlobalMem;
        }

    } // namespace nn
} // namespace chess
