#include "mcts/mcts.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

namespace chess
{
    namespace mcts
    {

        // MCTSNode implementation
        MCTSNode::MCTSNode(MCTSNode *parent, Move move, float prior)
            : parent(parent), move(move)
        {
            stats.priorProbability = prior;
        }

        MCTSNode::~MCTSNode() = default;

        float MCTSNode::getUCBScore(float cpuct, float parentVisits) const
        {
            float q = stats.getQ();
            float u = cpuct * stats.priorProbability * std::sqrt(parentVisits) / (1.0f + stats.visitCount.load());
            return q + u;
        }

        MCTSNode *MCTSNode::selectChild(float cpuct)
        {
            float parentVisits = static_cast<float>(stats.visitCount.load());

            MCTSNode *best = nullptr;
            float bestScore = -std::numeric_limits<float>::infinity();

            for (auto &child : children)
            {
                float score = child->getUCBScore(cpuct, parentVisits);

                // Subtract virtual loss for parallel search
                score -= child->stats.virtualLoss.load() / (1.0f + child->stats.visitCount.load());

                if (score > bestScore)
                {
                    bestScore = score;
                    best = child.get();
                }
            }

            return best;
        }

        void MCTSNode::expand(const std::vector<Move> &moves, const std::vector<float> &priors)
        {
            children.reserve(moves.size());

            for (size_t i = 0; i < moves.size(); i++)
            {
                children.push_back(std::make_unique<MCTSNode>(this, moves[i], priors[i]));
            }

            isExpanded = true;
        }

        void MCTSNode::backpropagate(float value, float virtualLoss)
        {
            MCTSNode *node = this;
            float v = value;

            while (node != nullptr)
            {
                node->stats.visitCount++;
                node->stats.totalValue = node->stats.totalValue.load() + v;
                node->stats.virtualLoss = node->stats.virtualLoss.load() - virtualLoss;

                // Negate value for opponent
                v = -v;
                node = node->parent;
            }
        }

        void MCTSNode::addVirtualLoss(float loss)
        {
            MCTSNode *node = this;
            while (node != nullptr)
            {
                node->stats.virtualLoss = node->stats.virtualLoss.load() + loss;
                node = node->parent;
            }
        }

        void MCTSNode::removeVirtualLoss(float loss)
        {
            MCTSNode *node = this;
            while (node != nullptr)
            {
                node->stats.virtualLoss = node->stats.virtualLoss.load() - loss;
                node = node->parent;
            }
        }

        std::vector<std::pair<Move, int>> MCTSNode::getVisitCounts() const
        {
            std::vector<std::pair<Move, int>> counts;
            counts.reserve(children.size());

            for (const auto &child : children)
            {
                counts.emplace_back(child->move, child->stats.visitCount.load());
            }

            return counts;
        }

        Move MCTSNode::getBestMove() const
        {
            const MCTSNode *best = nullptr;
            int bestVisits = -1;

            for (const auto &child : children)
            {
                int visits = child->stats.visitCount.load();
                if (visits > bestVisits)
                {
                    bestVisits = visits;
                    best = child.get();
                }
            }

            return best ? best->move : NULL_MOVE;
        }

        Move MCTSNode::selectMoveWithTemperature(float temperature, std::mt19937 &rng) const
        {
            if (children.empty())
                return NULL_MOVE;

            std::vector<float> probs;
            probs.reserve(children.size());

            if (temperature < 0.01f)
            {
                // Temperature ~= 0: pick max visit count
                return getBestMove();
            }

            // Calculate probabilities with temperature
            float sum = 0;
            for (const auto &child : children)
            {
                float visits = static_cast<float>(child->stats.visitCount.load());
                float prob = std::pow(visits, 1.0f / temperature);
                probs.push_back(prob);
                sum += prob;
            }

            // Normalize
            for (float &p : probs)
            {
                p /= sum;
            }

            // Sample
            std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
            size_t idx = dist(rng);

            return children[idx]->move;
        }

        // MCTSEngine implementation
        MCTSEngine::MCTSEngine(std::shared_ptr<nn::NeuralNetwork> network, MCTSConfig config)
            : network_(std::move(network)), config_(config), rng_(std::random_device{}())
        {
        }

        MCTSEngine::~MCTSEngine()
        {
            searching_ = false;
            for (auto &thread : searchThreads_)
            {
                if (thread.joinable())
                {
                    thread.join();
                }
            }
        }

        Move MCTSEngine::search(Board &board, int numSimulations)
        {
            if (numSimulations < 0)
            {
                numSimulations = config_.numSimulations;
            }

            // Initialize or reuse root
            if (!root_ || board.key() != rootBoard_.key())
            {
                root_ = std::make_unique<MCTSNode>();
                rootBoard_ = board;
            }

            searchStart_ = std::chrono::steady_clock::now();
            totalSimulations_ = 0;

            // Add noise at root for exploration
            if (config_.addNoise && root_->isExpanded)
            {
                addDirichletNoise();
            }

            // Run search
            if (config_.numThreads > 1)
            {
                parallelSearch(numSimulations);
            }
            else
            {
                for (int i = 0; i < numSimulations; i++)
                {
                    Board boardCopy = board;
                    runSimulation(boardCopy);
                    totalSimulations_++;
                }
            }

            // Select move with temperature
            float temperature = (rootBoard_.fullmoveNumber() <= config_.temperatureThreshold)
                                    ? config_.temperatureInitial
                                    : config_.temperatureFinal;

            return root_->selectMoveWithTemperature(temperature, rng_);
        }

        std::vector<std::pair<Move, float>> MCTSEngine::getPolicy(float temperature) const
        {
            if (!root_ || root_->children.empty())
            {
                return {};
            }

            auto visitCounts = root_->getVisitCounts();

            // Calculate probabilities
            std::vector<std::pair<Move, float>> policy;
            policy.reserve(visitCounts.size());

            if (temperature < 0.01f)
            {
                // Deterministic
                int maxVisits = 0;
                for (const auto &[m, v] : visitCounts)
                {
                    maxVisits = std::max(maxVisits, v);
                }
                for (const auto &[m, v] : visitCounts)
                {
                    policy.emplace_back(m, v == maxVisits ? 1.0f : 0.0f);
                }
            }
            else
            {
                // With temperature
                float sum = 0;
                for (const auto &[m, v] : visitCounts)
                {
                    sum += std::pow(static_cast<float>(v), 1.0f / temperature);
                }
                for (const auto &[m, v] : visitCounts)
                {
                    float prob = std::pow(static_cast<float>(v), 1.0f / temperature) / sum;
                    policy.emplace_back(m, prob);
                }
            }

            return policy;
        }

        float MCTSEngine::getRootValue() const
        {
            return root_ ? root_->stats.getQ() : 0.0f;
        }

        void MCTSEngine::newGame()
        {
            root_.reset();
        }

        void MCTSEngine::advanceTree(Move move)
        {
            if (!root_)
                return;

            // Find child with matching move
            for (auto &child : root_->children)
            {
                if (child->move == move)
                {
                    // Detach child and make it the new root
                    child->parent = nullptr;
                    root_ = std::move(child);
                    return;
                }
            }

            // Move not found, reset tree
            root_.reset();
        }

        float MCTSEngine::getNodesPerSecond() const
        {
            auto elapsed = std::chrono::steady_clock::now() - searchStart_;
            float seconds = std::chrono::duration<float>(elapsed).count();
            return seconds > 0 ? totalSimulations_ / seconds : 0;
        }

        void MCTSEngine::runSimulation(Board board)
        {
            // Selection
            MCTSNode *node = select(board);

            // Check terminal
            if (node->isTerminal)
            {
                float value = 0;
                if (node->terminalResult == GameResult::WhiteWins)
                {
                    value = (board.sideToMove() == Color::White) ? -1.0f : 1.0f;
                }
                else if (node->terminalResult == GameResult::BlackWins)
                {
                    value = (board.sideToMove() == Color::Black) ? -1.0f : 1.0f;
                }
                backup(node, value);
                return;
            }

            // Evaluation and expansion
            float value = evaluate(board, node);

            // Backup
            backup(node, value);
        }

        MCTSNode *MCTSEngine::select(Board &board)
        {
            MCTSNode *node = root_.get();

            while (node->isExpanded && !node->isTerminal)
            {
                node->addVirtualLoss(config_.virtualLoss);
                node = node->selectChild(config_.cpuct);

                if (node == nullptr)
                    break;

                StateInfo st;
                board.makeMove(node->move, st);
            }

            return node;
        }

        float MCTSEngine::evaluate(Board &board, MCTSNode *node)
        {
            // Check for game end
            GameResult result = board.result();
            if (result != GameResult::Ongoing)
            {
                node->isTerminal = true;
                node->terminalResult = result;

                if (result == GameResult::Draw)
                    return 0;
                if ((result == GameResult::WhiteWins && board.sideToMove() == Color::Black) ||
                    (result == GameResult::BlackWins && board.sideToMove() == Color::White))
                {
                    return 1.0f; // Current player won (we just finished their move)
                }
                return -1.0f;
            }

            // Neural network evaluation
            auto [policy, value] = network_->evaluate(board);

            // Get legal moves and their policies
            auto legalMoves = MoveGen::generateLegal(board);

            if (legalMoves.empty())
            {
                node->isTerminal = true;
                node->terminalResult = board.inCheck() ? (board.sideToMove() == Color::White ? GameResult::BlackWins : GameResult::WhiteWins) : GameResult::Draw;
                return 0;
            }

            // Extract priors for legal moves
            std::vector<float> priors;
            priors.reserve(legalMoves.size());

            float sumPriors = 0;
            for (const auto &move : legalMoves)
            {
                int idx = nn::BoardEncoder::moveToIndex(move);
                float prior = (idx >= 0 && idx < static_cast<int>(policy.size())) ? policy[idx] : 0.0f;
                priors.push_back(prior);
                sumPriors += prior;
            }

            // Normalize priors
            if (sumPriors > 0)
            {
                for (float &p : priors)
                {
                    p /= sumPriors;
                }
            }
            else
            {
                // Uniform distribution if no legal moves have prior
                float uniform = 1.0f / legalMoves.size();
                std::fill(priors.begin(), priors.end(), uniform);
            }

            // Expand node
            {
                std::lock_guard<std::mutex> lock(expansionMutex_);
                if (!node->isExpanded)
                {
                    node->expand(legalMoves, priors);
                }
            }

            return value;
        }

        void MCTSEngine::backup(MCTSNode *node, float value)
        {
            node->backpropagate(value, config_.virtualLoss);
        }

        void MCTSEngine::addDirichletNoise()
        {
            if (!root_ || root_->children.empty())
                return;

            // Generate Dirichlet noise
            std::gamma_distribution<float> gamma(config_.dirichletAlpha, 1.0f);
            std::vector<float> noise;
            noise.reserve(root_->children.size());

            float sum = 0;
            for (size_t i = 0; i < root_->children.size(); i++)
            {
                float n = gamma(rng_);
                noise.push_back(n);
                sum += n;
            }

            // Normalize and add to priors
            float epsilon = config_.dirichletEpsilon;
            for (size_t i = 0; i < root_->children.size(); i++)
            {
                float &prior = root_->children[i]->stats.priorProbability;
                prior = (1 - epsilon) * prior + epsilon * (noise[i] / sum);
            }
        }

        void MCTSEngine::parallelSearch(int numSimulations)
        {
            std::atomic<int> remaining(numSimulations);

            for (int i = 0; i < config_.numThreads; i++)
            {
                searchThreads_.emplace_back(&MCTSEngine::searchWorker, this, rootBoard_, std::ref(remaining));
            }

            for (auto &thread : searchThreads_)
            {
                thread.join();
            }
            searchThreads_.clear();
        }

        void MCTSEngine::searchWorker(Board board, std::atomic<int> &remaining)
        {
            while (remaining-- > 0)
            {
                Board boardCopy = board;
                runSimulation(boardCopy);
                totalSimulations_++;
            }
        }

        // BatchedMCTS implementation
        BatchedMCTS::BatchedMCTS(std::shared_ptr<nn::NeuralNetwork> network, int numGames, MCTSConfig config)
            : network_(std::move(network)), config_(config), activeGames_(numGames), rng_(std::random_device{}())
        {

            games_.resize(numGames);
            for (auto &game : games_)
            {
                game.board.setStartingPosition();
                game.root = std::make_unique<MCTSNode>();
            }
        }

        void BatchedMCTS::step(int simulationsPerStep)
        {
            // Run simulations for all active games
            runBatchedSimulations(simulationsPerStep);

            // Make moves for games that have completed their simulations
            for (auto &game : games_)
            {
                if (game.finished)
                    continue;

                // Get policy
                std::vector<float> policy(nn::POLICY_OUTPUT_SIZE, 0.0f);
                auto visitCounts = game.root->getVisitCounts();
                int totalVisits = 0;
                for (const auto &[m, v] : visitCounts)
                {
                    totalVisits += v;
                }
                if (totalVisits > 0)
                {
                    for (const auto &[m, v] : visitCounts)
                    {
                        int idx = nn::BoardEncoder::moveToIndex(m);
                        if (idx >= 0)
                        {
                            policy[idx] = static_cast<float>(v) / totalVisits;
                        }
                    }
                }

                // Store position and policy
                game.history.push_back(game.board);
                game.policies.push_back(policy);

                // Select move
                float temperature = (game.moveCount < config_.temperatureThreshold)
                                        ? config_.temperatureInitial
                                        : config_.temperatureFinal;
                Move move = game.root->selectMoveWithTemperature(temperature, rng_);

                // Make move
                StateInfo st;
                game.board.makeMove(move, st);
                game.moveCount++;

                // Reset tree for next move (simple approach - could reuse subtree)
                game.root = std::make_unique<MCTSNode>();

                // Check game end
                GameResult result = game.board.result();
                if (result != chess::GameResult::Ongoing)
                {
                    game.finished = true;
                    activeGames_--;
                    totalGamesCompleted_++;

                    // Calculate values based on result
                    std::vector<float> values;
                    for (size_t i = 0; i < game.history.size(); i++)
                    {
                        float value;
                        if (result == chess::GameResult::Draw)
                        {
                            value = 0;
                        }
                        else if (result == chess::GameResult::WhiteWins)
                        {
                            value = (game.history[i].sideToMove() == Color::White) ? 1.0f : -1.0f;
                        }
                        else
                        {
                            value = (game.history[i].sideToMove() == Color::Black) ? 1.0f : -1.0f;
                        }
                        values.push_back(value);
                    }

                    // Store finished game
                    finishedGames_.push_back({std::move(game.history),
                                              std::move(game.policies),
                                              std::move(values),
                                              result});
                }
            }
        }

        void BatchedMCTS::runBatchedSimulations(int numSimulations)
        {
            // Simplified batched simulation - could be optimized further
            for (int sim = 0; sim < numSimulations; sim++)
            {
                // Collect boards that need evaluation
                std::vector<size_t> gameIndices;
                std::vector<Board> boards;
                std::vector<MCTSNode *> nodes;

                for (size_t i = 0; i < games_.size(); i++)
                {
                    if (games_[i].finished)
                        continue;

                    // Select path
                    Board board = games_[i].board;
                    MCTSNode *node = games_[i].root.get();

                    while (node->isExpanded && !node->isTerminal && !node->children.empty())
                    {
                        node = node->selectChild(config_.cpuct);
                        if (node == nullptr)
                            break;
                        StateInfo st;
                        board.makeMove(node->move, st);
                    }

                    if (node && !node->isTerminal)
                    {
                        gameIndices.push_back(i);
                        boards.push_back(board);
                        nodes.push_back(node);
                    }
                }

                if (boards.empty())
                    continue;

                // Batch evaluation
                auto [policies, values] = network_->evaluateBatch(boards);

                // Expand and backup
                for (size_t i = 0; i < gameIndices.size(); i++)
                {
                    MCTSNode *node = nodes[i];
                    Board &board = boards[i];

                    // Check terminal
                    GameResult result = board.result();
                    if (result != chess::GameResult::Ongoing)
                    {
                        node->isTerminal = true;
                        node->terminalResult = result;

                        float value = 0;
                        if (result == chess::GameResult::WhiteWins)
                        {
                            value = (board.sideToMove() == Color::Black) ? 1.0f : -1.0f;
                        }
                        else if (result == chess::GameResult::BlackWins)
                        {
                            value = (board.sideToMove() == Color::White) ? 1.0f : -1.0f;
                        }
                        node->backpropagate(value, 0);
                        continue;
                    }

                    // Expand
                    auto legalMoves = MoveGen::generateLegal(board);
                    if (legalMoves.empty())
                        continue;

                    std::vector<float> priors;
                    float sumPriors = 0;
                    for (const auto &move : legalMoves)
                    {
                        int idx = nn::BoardEncoder::moveToIndex(move);
                        float prior = (idx >= 0 && idx < static_cast<int>(policies[i].size())) ? policies[i][idx] : 0.0f;
                        priors.push_back(prior);
                        sumPriors += prior;
                    }

                    if (sumPriors > 0)
                    {
                        for (float &p : priors)
                            p /= sumPriors;
                    }
                    else
                    {
                        std::fill(priors.begin(), priors.end(), 1.0f / legalMoves.size());
                    }

                    if (!node->isExpanded)
                    {
                        node->expand(legalMoves, priors);
                    }

                    node->backpropagate(values[i], 0);
                }
            }
        }

        std::vector<BatchedMCTS::GameResult> BatchedMCTS::getFinishedGames()
        {
            return std::move(finishedGames_);
        }

        void BatchedMCTS::resetFinishedGames()
        {
            for (auto &game : games_)
            {
                if (game.finished)
                {
                    game.board.setStartingPosition();
                    game.root = std::make_unique<MCTSNode>();
                    game.history.clear();
                    game.policies.clear();
                    game.finished = false;
                    game.moveCount = 0;
                    activeGames_++;
                }
            }
            finishedGames_.clear();
        }

    } // namespace mcts
} // namespace chess
