#include "mcts/mcts.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace chess {
namespace mcts {

// ============================================================================
// MCTSNode
// ============================================================================

MCTSNode::MCTSNode(MCTSNode* parent, Move move, float prior)
    : parent(parent), move(move) {
    stats.priorProbability = prior;
}

MCTSNode::~MCTSNode() = default;

float MCTSNode::getUCBScore(float cpuct, float parentVisits) const {
    float Q = stats.getQ();
    float U = cpuct * stats.priorProbability * std::sqrt(parentVisits) / 
              (1.0f + stats.visitCount.load());
    return Q + U;
}

MCTSNode* MCTSNode::selectChild(float cpuct) {
    MCTSNode* best = nullptr;
    float bestScore = -std::numeric_limits<float>::infinity();
    float parentVisits = static_cast<float>(stats.visitCount.load());
    
    for (auto& child : children) {
        float score = child->getUCBScore(cpuct, parentVisits);
        if (score > bestScore) {
            bestScore = score;
            best = child.get();
        }
    }
    
    return best;
}

void MCTSNode::expand(const std::vector<Move>& moves, const std::vector<float>& priors) {
    children.reserve(moves.size());
    for (size_t i = 0; i < moves.size(); i++) {
        float prior = (i < priors.size()) ? priors[i] : 1.0f / moves.size();
        children.push_back(std::make_unique<MCTSNode>(this, moves[i], prior));
    }
    isExpanded = true;
}

void MCTSNode::backpropagate(float value, float virtualLoss) {
    MCTSNode* node = this;
    float v = value;
    
    while (node != nullptr) {
        node->stats.visitCount.fetch_add(1);
        node->stats.addValue(v);
        node->stats.removeVirtualLoss(virtualLoss);
        
        v = -v;  // Flip value for parent (opponent's perspective)
        node = node->parent;
    }
}

void MCTSNode::addVirtualLoss(float loss) {
    stats.addVirtualLoss(loss);
}

void MCTSNode::removeVirtualLoss(float loss) {
    stats.removeVirtualLoss(loss);
}

std::vector<std::pair<Move, int>> MCTSNode::getVisitCounts() const {
    std::vector<std::pair<Move, int>> counts;
    counts.reserve(children.size());
    
    for (const auto& child : children) {
        counts.emplace_back(child->move, child->stats.visitCount.load());
    }
    
    return counts;
}

Move MCTSNode::getBestMove() const {
    MCTSNode* best = nullptr;
    int bestVisits = -1;
    
    for (const auto& child : children) {
        int visits = child->stats.visitCount.load();
        if (visits > bestVisits) {
            bestVisits = visits;
            best = child.get();
        }
    }
    
    return best ? best->move : NULL_MOVE;
}

Move MCTSNode::selectMoveWithTemperature(float temperature, std::mt19937& rng) const {
    if (children.empty()) return NULL_MOVE;
    
    if (temperature < 0.01f) {
        return getBestMove();
    }
    
    std::vector<float> probabilities;
    probabilities.reserve(children.size());
    
    float sum = 0.0f;
    for (const auto& child : children) {
        float visits = static_cast<float>(child->stats.visitCount.load());
        float prob = std::pow(visits, 1.0f / temperature);
        probabilities.push_back(prob);
        sum += prob;
    }
    
    // Normalize (with guard for division by zero)
    if (sum > 1e-8f) {
        for (float& p : probabilities) {
            p /= sum;
        }
    } else {
        // Uniform distribution if all zero
        float uniform = 1.0f / children.size();
        for (float& p : probabilities) {
            p = uniform;
        }
    }
    
    // Sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumsum = 0.0f;
    
    for (size_t i = 0; i < children.size(); i++) {
        cumsum += probabilities[i];
        if (r <= cumsum) {
            return children[i]->move;
        }
    }
    
    return children.back()->move;
}

// ============================================================================
// MCTSEngine
// ============================================================================

MCTSEngine::MCTSEngine(std::shared_ptr<nn::NeuralNetwork> network, MCTSConfig config)
    : network_(std::move(network)), config_(std::move(config)), 
      rng_(std::random_device{}()) {
    newGame();
}

MCTSEngine::~MCTSEngine() {
    searching_ = false;
}

Move MCTSEngine::search(Board& board, int numSimulations) {
    if (numSimulations < 0) {
        numSimulations = config_.numSimulations;
    }
    
    
    searchStart_ = std::chrono::steady_clock::now();
    totalSimulations_ = 0;
    searching_ = true;
    
    // Reset root if board doesn't match
    rootBoard_ = board;
    root_ = std::make_unique<MCTSNode>(nullptr, NULL_MOVE, 1.0f);
    
    
    // Add Dirichlet noise at root
    if (config_.addNoise) {
        addDirichletNoise();
    }
    
    
    // Run simulations
    for (int i = 0; i < numSimulations && searching_; i++) {
                    Board simBoard = rootBoard_;
                    runSimulation(simBoard);
            totalSimulations_++;
    }
    
    
    searching_ = false;
    
    // Select move
    float temperature = (rootBoard_.fullmoveNumber() < config_.temperatureThreshold) 
        ? config_.temperatureInitial 
        : config_.temperatureFinal;
    
    return root_->selectMoveWithTemperature(temperature, rng_);
}

std::vector<std::pair<Move, float>> MCTSEngine::getPolicy(float temperature) const {
    std::vector<std::pair<Move, float>> policy;
    
    auto visitCounts = root_->getVisitCounts();
    if (visitCounts.empty()) return policy;
    
    // Calculate probabilities
    float sum = 0.0f;
    for (const auto& [move, visits] : visitCounts) {
        sum += std::pow(static_cast<float>(visits), 1.0f / temperature);
    }
    
    for (const auto& [move, visits] : visitCounts) {
        float prob = std::pow(static_cast<float>(visits), 1.0f / temperature) / sum;
        policy.emplace_back(move, prob);
    }
    
    return policy;
}

float MCTSEngine::getRootValue() const {
    return root_ ? root_->stats.getQ() : 0.0f;
}

void MCTSEngine::newGame() {
    root_ = std::make_unique<MCTSNode>(nullptr, NULL_MOVE, 1.0f);
    rootBoard_.setStartingPosition();
    rootStates_.clear();
}

void MCTSEngine::advanceTree(Move move) {
    if (!root_) {
        newGame();
        return;
    }
    
    // Find child with matching move
    for (auto& child : root_->children) {
        if (child->move == move) {
            // Detach child and make it the new root
            child->parent = nullptr;
            root_ = std::move(child);
            rootStates_.emplace_back();
            rootBoard_.makeMove(move, rootStates_.back());
            return;
        }
    }
    
    // Move not found in tree, create new root
    rootStates_.emplace_back();
    rootBoard_.makeMove(move, rootStates_.back());
    root_ = std::make_unique<MCTSNode>(nullptr, NULL_MOVE, 1.0f);
}

float MCTSEngine::getNodesPerSecond() const {
    auto elapsed = std::chrono::steady_clock::now() - searchStart_;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    if (ms == 0) return 0.0f;
    return static_cast<float>(totalSimulations_) * 1000.0f / ms;
}

void MCTSEngine::runSimulation(Board board) {
    std::vector<StateInfo> states;  // Keep states alive for duration of simulation
    states.reserve(100);  // Pre-allocate to avoid reallocation
    MCTSNode* node = select(board, states);
    
    if (node == nullptr) return;
    
    float value = evaluate(board, node);
    node->backpropagate(value, config_.virtualLoss);
}

MCTSNode* MCTSEngine::select(Board& board, std::vector<StateInfo>& states) {
    MCTSNode* node = root_.get();
    // States vector passed from caller to keep them alive
    
    while (node->isExpanded && !node->isTerminal) {
        node->addVirtualLoss(config_.virtualLoss);
        
        node = node->selectChild(config_.cpuct);
        if (node == nullptr) break;
        
        states.emplace_back();
        board.makeMove(node->move, states.back());
    }
    
    return node;
}

float MCTSEngine::evaluate(Board& board, MCTSNode* node) {
    // Check for terminal state
    GameResult result = board.result();
    if (result != GameResult::Ongoing) {
            node->isTerminal = true;
        node->terminalResult = result;
        
        if (result == GameResult::Draw) {
            return 0.0f;
        } else if ((result == GameResult::WhiteWins && 
                    board.sideToMove() == Color::White) ||
                   (result == GameResult::BlackWins && 
                    board.sideToMove() == Color::Black)) {
            return 1.0f;  // We won
        } else {
            return -1.0f;  // We lost
        }
    }
    
    // Get neural network evaluation
    auto [policy, value] = network_->evaluate(board);
    
    // Expand node
    auto moves = MoveGen::generateLegal(board);
    
    if (moves.empty()) {
            node->isTerminal = true;
        return 0.0f;  // Likely stalemate or checkmate not detected
    }
    
    std::vector<float> priors;
    priors.reserve(moves.size());
    
    float totalPrior = 0.0f;
    for (const Move& move : moves) {
        int idx = nn::BoardEncoder::moveToIndex(move);
        float prior = (idx >= 0 && idx < nn::POLICY_OUTPUT_SIZE) ? policy[idx] : 0.0f;
        priors.push_back(prior);
        totalPrior += prior;
    }
    
    // Normalize priors
    if (totalPrior > 0) {
        for (float& p : priors) {
            p /= totalPrior;
        }
    } else {
        // Uniform if no valid priors
        float uniform = 1.0f / moves.size();
        for (float& p : priors) {
            p = uniform;
        }
    }
    
    node->expand(moves, priors);
    
    return value;
}

void MCTSEngine::addDirichletNoise() {
    if (!root_ || root_->children.empty()) {
        // Need to expand root first
        auto [policy, value] = network_->evaluate(rootBoard_);
        auto moves = MoveGen::generateLegal(rootBoard_);
        
        // If no legal moves, game is over
        if (moves.empty()) {
            return;
        }
        
        std::vector<float> priors;
        float totalPrior = 0.0f;
        for (const Move& move : moves) {
            int idx = nn::BoardEncoder::moveToIndex(move);
            float prior = (idx >= 0 && idx < nn::POLICY_OUTPUT_SIZE) ? policy[idx] : 0.0f;
            priors.push_back(prior);
            totalPrior += prior;
        }
        
        if (totalPrior > 0) {
            for (float& p : priors) {
                p /= totalPrior;
            }
        } else {
            // Uniform distribution if no valid priors
            float uniform = 1.0f / moves.size();
            for (float& p : priors) {
                p = uniform;
            }
        }
        
        root_->expand(moves, priors);
    }
    
    // If still no children after expansion, don't add noise
    if (root_->children.empty()) {
        return;
    }
    
    // Generate Dirichlet noise
    std::gamma_distribution<float> gamma(config_.dirichletAlpha, 1.0f);
    std::vector<float> noise;
    float noiseSum = 0.0f;
    
    for (size_t i = 0; i < root_->children.size(); i++) {
        float n = gamma(rng_);
        noise.push_back(n);
        noiseSum += n;
    }
    
    // Guard against division by zero
    if (noiseSum < 1e-8f) {
        return;
    }
    
    // Normalize and apply noise
    float epsilon = config_.dirichletEpsilon;
    for (size_t i = 0; i < root_->children.size(); i++) {
        float normalizedNoise = noise[i] / noiseSum;
        float prior = root_->children[i]->stats.priorProbability;
        root_->children[i]->stats.priorProbability = 
            (1.0f - epsilon) * prior + epsilon * normalizedNoise;
    }
}

} // namespace mcts
} // namespace chess
