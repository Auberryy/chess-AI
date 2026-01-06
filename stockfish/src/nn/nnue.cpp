#include "nn/nnue.hpp"
#include <iostream>
#include <algorithm>

namespace chess
{
    namespace nn
    {

        // NNUEEvaluator implementation
        NNUEEvaluator::NNUEEvaluator(const std::string &modelPath)
            : device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
        {

            model_ = NNUENet();
            model_->to(device_);

            if (!modelPath.empty())
            {
                load(modelPath);
            }

            // Initialize accumulators
            accumulators_[0] = torch::zeros({NNUE_HALF_DIMENSIONS}, device_);
            accumulators_[1] = torch::zeros({NNUE_HALF_DIMENSIONS}, device_);
        }

        std::pair<std::vector<int>, std::vector<int>> NNUEEvaluator::extractFeatures(const Board &board)
        {
            std::vector<int> whiteFeatures, blackFeatures;

            Square whiteKing = board.kingSquare(Color::White);
            Square blackKing = board.kingSquare(Color::Black);

            // Iterate through all pieces (except kings)
            Bitboard pieces = board.occupied() & ~board.piecesBB(PieceType::King);

            while (pieces)
            {
                Square sq = bb::popLsb(pieces);
                Piece p = board.pieceAt(sq);

                // White's perspective
                int whitePieceIdx = nnuePieceIndex(p, Color::White);
                if (whitePieceIdx >= 0)
                {
                    whiteFeatures.push_back(nnueFeatureIndex(whiteKing, sq, whitePieceIdx));
                }

                // Black's perspective (mirror the board)
                Square mirroredSq = static_cast<Square>(sq ^ 56); // Flip rank
                Square mirroredKing = static_cast<Square>(blackKing ^ 56);
                int blackPieceIdx = nnuePieceIndex(p, Color::Black);
                if (blackPieceIdx >= 0)
                {
                    blackFeatures.push_back(nnueFeatureIndex(mirroredKing, mirroredSq, blackPieceIdx));
                }
            }

            return {whiteFeatures, blackFeatures};
        }

        int NNUEEvaluator::evaluate(const Board &board)
        {
            torch::NoGradGuard no_grad;
            model_->eval();

            auto [whiteFeatures, blackFeatures] = extractFeatures(board);

            auto output = model_->forward(whiteFeatures, blackFeatures, board.sideToMove());

            // Convert to centipawns
            int eval = static_cast<int>(output.item<float>());

            return eval;
        }

        void NNUEEvaluator::reset(const Board &board)
        {
            auto [whiteFeatures, blackFeatures] = extractFeatures(board);

            accumulators_[0] = model_->getAccumulator(whiteFeatures);
            accumulators_[1] = model_->getAccumulator(blackFeatures);
            accumulatorStack_.clear();
        }

        void NNUEEvaluator::makeMove(const Board &board, Move move)
        {
            // Save current accumulators
            accumulatorStack_.push_back(accumulators_);

            // For now, just recompute (incremental updates can be optimized later)
            reset(board);
        }

        void NNUEEvaluator::unmakeMove()
        {
            if (!accumulatorStack_.empty())
            {
                accumulators_ = accumulatorStack_.back();
                accumulatorStack_.pop_back();
            }
        }

        void NNUEEvaluator::train(const std::vector<std::tuple<Board, int>> &samples,
                                  int batchSize, float learningRate)
        {
            model_->train();

            torch::optim::Adam optimizer(model_->parameters(),
                                         torch::optim::AdamOptions(learningRate));

            // Shuffle indices
            std::vector<size_t> indices(samples.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());

            float totalLoss = 0;
            int numBatches = 0;

            for (size_t i = 0; i < samples.size(); i += batchSize)
            {
                size_t batchEnd = std::min(i + batchSize, samples.size());
                int actualBatchSize = batchEnd - i;

                // Prepare batch tensors
                auto whiteFeatureTensor = torch::zeros({actualBatchSize, NNUE_FEATURES}, device_);
                auto blackFeatureTensor = torch::zeros({actualBatchSize, NNUE_FEATURES}, device_);
                auto stmTensor = torch::zeros({actualBatchSize}, torch::kLong).to(device_);
                auto targetTensor = torch::zeros({actualBatchSize, 1}, device_);

                for (size_t j = 0; j < actualBatchSize; j++)
                {
                    const auto &[board, eval] = samples[indices[i + j]];
                    auto [whiteFeatures, blackFeatures] = extractFeatures(board);

                    // Set sparse features
                    for (int idx : whiteFeatures)
                    {
                        if (idx >= 0 && idx < NNUE_FEATURES)
                        {
                            whiteFeatureTensor[j][idx] = 1.0f;
                        }
                    }
                    for (int idx : blackFeatures)
                    {
                        if (idx >= 0 && idx < NNUE_FEATURES)
                        {
                            blackFeatureTensor[j][idx] = 1.0f;
                        }
                    }

                    stmTensor[j] = (board.sideToMove() == Color::White) ? 0 : 1;
                    targetTensor[j][0] = static_cast<float>(eval);
                }

                // Forward pass
                optimizer.zero_grad();
                auto output = model_->forwardBatch(whiteFeatureTensor, blackFeatureTensor, stmTensor);

                // MSE loss
                auto loss = torch::mse_loss(output, targetTensor);

                // Backward pass
                loss.backward();
                optimizer.step();

                totalLoss += loss.item<float>();
                numBatches++;
            }

            std::cout << "NNUE Training loss: " << totalLoss / numBatches << std::endl;
        }

        void NNUEEvaluator::save(const std::string &path)
        {
            torch::save(model_, path);
        }

        void NNUEEvaluator::load(const std::string &path)
        {
            torch::load(model_, path);
            model_->to(device_);
        }

        // NNUEEncoder implementation
        std::pair<std::vector<int>, std::vector<int>> NNUEEncoder::encode(const Board &board)
        {
            std::vector<int> whiteFeatures, blackFeatures;

            Square whiteKing = board.kingSquare(Color::White);
            Square blackKing = board.kingSquare(Color::Black);

            Bitboard pieces = board.occupied() & ~board.piecesBB(PieceType::King);

            while (pieces)
            {
                Square sq = bb::popLsb(pieces);
                Piece p = board.pieceAt(sq);

                int whitePieceIdx = nnuePieceIndex(p, Color::White);
                if (whitePieceIdx >= 0)
                {
                    whiteFeatures.push_back(nnueFeatureIndex(whiteKing, sq, whitePieceIdx));
                }

                Square mirroredSq = static_cast<Square>(sq ^ 56);
                Square mirroredKing = static_cast<Square>(blackKing ^ 56);
                int blackPieceIdx = nnuePieceIndex(p, Color::Black);
                if (blackPieceIdx >= 0)
                {
                    blackFeatures.push_back(nnueFeatureIndex(mirroredKing, mirroredSq, blackPieceIdx));
                }
            }

            return {whiteFeatures, blackFeatures};
        }

        torch::Tensor NNUEEncoder::createSparseTensor(const std::vector<int> &features, int numFeatures)
        {
            auto tensor = torch::zeros({numFeatures});
            for (int idx : features)
            {
                if (idx >= 0 && idx < numFeatures)
                {
                    tensor[idx] = 1.0f;
                }
            }
            return tensor;
        }

    } // namespace nn
} // namespace chess
