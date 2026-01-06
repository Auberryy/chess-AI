#include "training/trainer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <ctime>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <signal.h>
#endif

namespace fs = std::filesystem;

namespace chess {
namespace training {

// Global shutdown flag for graceful termination
static std::atomic<bool> g_shutdownRequested{false};
static Trainer* g_trainerInstance = nullptr;

void requestShutdown() {
    g_shutdownRequested = true;
    if (g_trainerInstance) {
        g_trainerInstance->stop();
    }
    std::cout << "\n\n>>> Shutdown requested! Finishing current operation and saving... <<<\n\n";
}

#ifdef _WIN32
BOOL WINAPI consoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_BREAK_EVENT) {
        requestShutdown();
        return TRUE;
    }
    return FALSE;
}
#else
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        requestShutdown();
    }
}
#endif

void setupSignalHandlers() {
#ifdef _WIN32
    SetConsoleCtrlHandler(consoleHandler, TRUE);
#else
    struct sigaction sa;
    sa.sa_handler = signalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
#endif
}

// ============================================================================
// TrainingSample
// ============================================================================

TrainingSample::TrainingSample(const Board& board, const std::vector<float>& p, float v)
    : policy(p), value(v) {
    // Encode board to input tensor format (flattened)
    // This would call BoardEncoder::encode() and flatten the result
    boardInput.resize(nn::INPUT_CHANNELS * 64);
    
    // Simple encoding - proper implementation would use BoardEncoder
    std::fill(boardInput.begin(), boardInput.end(), 0.0f);
    
    // Encode piece positions (12 planes: 6 pieces x 2 colors)
    for (int sq = 0; sq < 64; sq++) {
        Piece piece = board.pieceAt(static_cast<Square>(sq));
        if (piece != Piece::None) {
            int pt = static_cast<int>(piece) % 6;  // Piece type (0-5)
            int c = static_cast<int>(piece) / 6;   // Color (0 or 1)
            int planeIdx = pt + c * 6;
            boardInput[planeIdx * 64 + sq] = 1.0f;
        }
    }
    
    // Side to move (plane 12)
    float stm = board.sideToMove() == Color::White ? 1.0f : 0.0f;
    for (int i = 0; i < 64; i++) {
        boardInput[12 * 64 + i] = stm;
    }
    
    // Castling rights (planes 13-16)
    auto cr = board.castlingRights();
    for (int i = 0; i < 64; i++) {
        boardInput[13 * 64 + i] = (static_cast<int>(cr) & static_cast<int>(CastlingRights::WhiteKingSide)) ? 1.0f : 0.0f;
        boardInput[14 * 64 + i] = (static_cast<int>(cr) & static_cast<int>(CastlingRights::WhiteQueenSide)) ? 1.0f : 0.0f;
        boardInput[15 * 64 + i] = (static_cast<int>(cr) & static_cast<int>(CastlingRights::BlackKingSide)) ? 1.0f : 0.0f;
        boardInput[16 * 64 + i] = (static_cast<int>(cr) & static_cast<int>(CastlingRights::BlackQueenSide)) ? 1.0f : 0.0f;
    }
    
    // En passant (plane 17)
    Square ep = board.enPassantSquare();
    if (ep != NO_SQUARE) {
        boardInput[17 * 64 + ep] = 1.0f;
    }
}

// ============================================================================
// ReplayBuffer
// ============================================================================

ReplayBuffer::ReplayBuffer(size_t maxSize) 
    : maxSize_(maxSize), rng_(std::random_device{}()) {}

void ReplayBuffer::add(const TrainingSample& sample) {
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.push_back(sample);
    while (samples_.size() > maxSize_) {
        samples_.pop_front();
    }
}

void ReplayBuffer::addBatch(const std::vector<TrainingSample>& samples) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& s : samples) {
        samples_.push_back(s);
    }
    while (samples_.size() > maxSize_) {
        samples_.pop_front();
    }
}

std::vector<TrainingSample> ReplayBuffer::sample(size_t batchSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<TrainingSample> batch;
    if (samples_.empty()) return batch;
    
    batchSize = std::min(batchSize, samples_.size());
    batch.reserve(batchSize);
    
    // Sample with replacement
    std::uniform_int_distribution<size_t> dist(0, samples_.size() - 1);
    for (size_t i = 0; i < batchSize; i++) {
        batch.push_back(samples_[dist(rng_)]);
    }
    
    return batch;
}

void ReplayBuffer::save(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to save replay buffer to " << path << std::endl;
        return;
    }
    
    size_t numSamples = samples_.size();
    ofs.write(reinterpret_cast<const char*>(&numSamples), sizeof(numSamples));
    
    for (const auto& sample : samples_) {
        // Board input
        size_t inputSize = sample.boardInput.size();
        ofs.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
        ofs.write(reinterpret_cast<const char*>(sample.boardInput.data()), 
                  inputSize * sizeof(float));
        
        // Policy
        size_t policySize = sample.policy.size();
        ofs.write(reinterpret_cast<const char*>(&policySize), sizeof(policySize));
        ofs.write(reinterpret_cast<const char*>(sample.policy.data()), 
                  policySize * sizeof(float));
        
        // Value
        ofs.write(reinterpret_cast<const char*>(&sample.value), sizeof(sample.value));
    }
    
    std::cout << "Saved " << numSamples << " samples to " << path << std::endl;
}

void ReplayBuffer::load(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to load replay buffer from " << path << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
    
    size_t numSamples;
    ifs.read(reinterpret_cast<char*>(&numSamples), sizeof(numSamples));
    
    for (size_t i = 0; i < numSamples; i++) {
        TrainingSample sample;
        
        // Board input
        size_t inputSize;
        ifs.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
        sample.boardInput.resize(inputSize);
        ifs.read(reinterpret_cast<char*>(sample.boardInput.data()), 
                 inputSize * sizeof(float));
        
        // Policy
        size_t policySize;
        ifs.read(reinterpret_cast<char*>(&policySize), sizeof(policySize));
        sample.policy.resize(policySize);
        ifs.read(reinterpret_cast<char*>(sample.policy.data()), 
                 policySize * sizeof(float));
        
        // Value
        ifs.read(reinterpret_cast<char*>(&sample.value), sizeof(sample.value));
        
        samples_.push_back(std::move(sample));
    }
    
    std::cout << "Loaded " << samples_.size() << " samples from " << path << std::endl;
}

// ============================================================================
// TrainingStats
// ============================================================================

std::chrono::seconds TrainingStats::getSessionTime() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(now - sessionStart);
}

std::chrono::seconds TrainingStats::getAverageSessionTime() const {
    if (numSessions == 0) return std::chrono::seconds{0};
    return totalTrainingTime / numSessions;
}

void TrainingStats::save(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs) return;
    
    ofs << "{\n";
    ofs << "  \"totalGamesPlayed\": " << totalGamesPlayed << ",\n";
    ofs << "  \"totalPositionsTrained\": " << totalPositionsTrained << ",\n";
    ofs << "  \"currentIteration\": " << currentIteration << ",\n";
    ofs << "  \"currentLoss\": " << currentLoss << ",\n";
    ofs << "  \"currentWinRate\": " << currentWinRate << ",\n";
    ofs << "  \"bestWinRate\": " << bestWinRate << ",\n";
    ofs << "  \"numSessions\": " << numSessions << ",\n";
    ofs << "  \"totalTrainingSeconds\": " << totalTrainingTime.count() << "\n";
    ofs << "}\n";
}

void TrainingStats::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return;
    
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.find("totalGamesPlayed") != std::string::npos) {
            sscanf(line.c_str(), " \"totalGamesPlayed\": %d", &totalGamesPlayed);
        } else if (line.find("totalPositionsTrained") != std::string::npos) {
            sscanf(line.c_str(), " \"totalPositionsTrained\": %d", &totalPositionsTrained);
        } else if (line.find("currentIteration") != std::string::npos) {
            sscanf(line.c_str(), " \"currentIteration\": %d", &currentIteration);
        } else if (line.find("currentLoss") != std::string::npos) {
            sscanf(line.c_str(), " \"currentLoss\": %f", &currentLoss);
        } else if (line.find("currentWinRate") != std::string::npos) {
            sscanf(line.c_str(), " \"currentWinRate\": %f", &currentWinRate);
        } else if (line.find("bestWinRate") != std::string::npos) {
            sscanf(line.c_str(), " \"bestWinRate\": %f", &bestWinRate);
        } else if (line.find("numSessions") != std::string::npos) {
            sscanf(line.c_str(), " \"numSessions\": %d", &numSessions);
        } else if (line.find("totalTrainingSeconds") != std::string::npos) {
            long seconds;
            sscanf(line.c_str(), " \"totalTrainingSeconds\": %ld", &seconds);
            totalTrainingTime = std::chrono::seconds{seconds};
        }
    }
}

// ============================================================================
// Trainer
// ============================================================================

Trainer::Trainer(TrainingConfig config) : config_(std::move(config)) {
    // Create directories
    fs::create_directories(config_.checkpointDir);
    fs::create_directories("models");
    
    // Initialize neural network
    network_ = std::make_shared<nn::NeuralNetwork>("", config_.gpuId, 
                                                    config_.gpuMemoryFraction, 
                                                    config_.useAmp);
    
    // Initialize replay buffer
    replayBuffer_ = std::make_unique<ReplayBuffer>(config_.replayBufferSize);
    
    // Setup signal handlers
    g_trainerInstance = this;
    setupSignalHandlers();
    
    std::cout << "========================================\n";
    std::cout << "  Chess AI Trainer (C++)\n";
    std::cout << "  Press Ctrl+C to stop gracefully\n";
    std::cout << "========================================\n\n";
}

Trainer::~Trainer() {
    running_ = false;
    g_trainerInstance = nullptr;
}

void Trainer::train() {
    running_ = true;
    stats_.sessionStart = std::chrono::steady_clock::now();
    stats_.numSessions++;
    
    std::cout << "Starting infinite training loop...\n";
    std::cout << "  Games/iteration: " << config_.numSelfPlayGames << "\n";
    std::cout << "  MCTS simulations: " << config_.mctsSimulations << "\n";
    std::cout << "  Batch size: " << config_.batchSize << "\n";
    std::cout << "  Checkpoint interval: " << config_.checkpointInterval << "\n\n";
    
    // Load checkpoint if exists
    std::string latestCheckpoint = config_.checkpointDir + "/latest.pt";
    if (fs::exists(latestCheckpoint)) {
        std::cout << "Loading checkpoint: " << latestCheckpoint << "\n";
        loadCheckpoint(latestCheckpoint);
    }
    
    // Main training loop
    while (running_ && !g_shutdownRequested) {
        stats_.currentIteration++;
        
        printStatus();
        
        // Phase 1: Self-play
        std::cout << "\n[1] Self-Play (" << config_.numSelfPlayGames << " games)...\n";
        runSelfPlay(config_.numSelfPlayGames);
        
        if (!running_ || g_shutdownRequested) break;
        
        // Phase 2: Training
        std::cout << "\n[2] Training...\n";
        runTraining(config_.epochsPerIteration);
        
        if (!running_ || g_shutdownRequested) break;
        
        // Phase 3: Evaluation (periodic)
        if (stats_.currentIteration % config_.evaluationInterval == 0) {
            std::cout << "\n[3] Evaluation...\n";
            float winRate = evaluateAgainstStockfish(config_.evaluationGames);
            stats_.currentWinRate = winRate;
            
            if (winRate > stats_.bestWinRate) {
                stats_.bestWinRate = winRate;
                network_->save("models/best_model.pt");
                std::cout << "  New best model! Win rate: " << (winRate * 100) << "%\n";
            }
            
            if (evaluationCallback_) {
                if (!evaluationCallback_(winRate)) {
                    std::cout << "Training stopped by evaluation callback\n";
                    break;
                }
            }
        }
        
        // Phase 4: Checkpoint (periodic)
        if (stats_.currentIteration % config_.checkpointInterval == 0) {
            saveCheckpoint(latestCheckpoint);
        }
        
        // Progress callback
        if (progressCallback_) {
            progressCallback_(stats_);
        }
    }
    
    // Final save
    std::cout << "\n========================================\n";
    std::cout << "  Saving final checkpoint...\n";
    std::cout << "========================================\n";
    
    saveCheckpoint(latestCheckpoint);
    network_->save("models/latest_model.pt");
    
    // Update total training time
    stats_.totalTrainingTime += stats_.getSessionTime();
    
    // Print summary
    printSummary();
}

void Trainer::runSelfPlay(int numGames) {
    std::cerr << "[DEBUG] runSelfPlay started with " << numGames << " games\n";
    mcts::MCTSConfig mctsConfig;
    mctsConfig.numSimulations = config_.mctsSimulations;
    mctsConfig.temperatureInitial = config_.temperatureInitial;
    mctsConfig.temperatureThreshold = config_.temperatureThreshold;
    mctsConfig.addNoise = true;
    
    std::cerr << "[DEBUG] Creating MCTSEngine\n";
    mcts::MCTSEngine engine(network_, mctsConfig);
    std::cerr << "[DEBUG] MCTSEngine created\n";
    
    int totalPositions = 0;
    int whiteWins = 0, blackWins = 0, draws = 0;
    
    for (int gameIdx = 0; gameIdx < numGames && running_ && !g_shutdownRequested; gameIdx++) {
        std::cerr << "[DEBUG] Starting game " << gameIdx << "\n";
        Board board;
        std::vector<Board> positions;
        std::vector<std::vector<float>> policies;
        std::vector<float> values;
        std::vector<StateInfo> stateHistory;  // Keep states alive
        
        engine.newGame();
        int moveCount = 0;
        
        GameResult result = board.result();
        while (result == GameResult::Ongoing && moveCount < 500 && running_ && !g_shutdownRequested) {
            std::cerr << "[DEBUG] Move " << moveCount << " - Saving position\n";
            positions.push_back(board);
            
            std::cerr << "[DEBUG] Move " << moveCount << " - Starting search\n";
            // Get MCTS policy
            Move move = engine.search(const_cast<Board&>(board));
            std::cerr << "[DEBUG] Move " << moveCount << " - Search complete\n";
            
            if (!running_ || g_shutdownRequested) break;
            
            auto policy = engine.getPolicy(
                moveCount < config_.temperatureThreshold ? 
                    config_.temperatureInitial : 0.1f);
            
            // Convert to vector
            std::vector<float> policyVec(nn::POLICY_OUTPUT_SIZE, 0.0f);
            for (const auto& [m, prob] : policy) {
                int idx = nn::BoardEncoder::moveToIndex(m);
                if (idx >= 0 && idx < nn::POLICY_OUTPUT_SIZE) {
                    policyVec[idx] = prob;
                }
            }
            policies.push_back(policyVec);
            
            // Make move
            stateHistory.emplace_back();
            board.makeMove(move, stateHistory.back());
            engine.advanceTree(move);
            moveCount++;
            
            result = board.result();
        }
        
        if (!running_ || g_shutdownRequested) break;
        
        // Determine game result
        float gameResult = 0.0f;
        if (result == GameResult::WhiteWins) {
            gameResult = 1.0f;
            whiteWins++;
        } else if (result == GameResult::BlackWins) {
            gameResult = -1.0f;
            blackWins++;
        } else {
            draws++;
        }
        
        // Generate training samples with proper value targets
        auto samples = generateSamplesFromGame(positions, policies, values);
        
        // Assign values based on game outcome
        for (size_t i = 0; i < samples.size(); i++) {
            // Value from perspective of side to move
            bool whiteTurn = (i % 2 == 0);  // Simplified - should check actual board
            samples[i].value = whiteTurn ? gameResult : -gameResult;
        }
        
        replayBuffer_->addBatch(samples);
        totalPositions += samples.size();
        
        std::cout << "\r  Game " << (gameIdx + 1) << "/" << numGames 
                  << " - Moves: " << moveCount 
                  << " - W:" << whiteWins << " D:" << draws << " L:" << blackWins
                  << "        " << std::flush;
    }
    
    std::cout << "\n  Generated " << totalPositions << " positions from " 
              << (whiteWins + draws + blackWins) << " games\n";
    
    stats_.totalGamesPlayed += numGames;
    stats_.totalPositionsTrained += totalPositions;
}

std::vector<TrainingSample> Trainer::generateSamplesFromGame(
    const std::vector<Board>& positions,
    const std::vector<std::vector<float>>& policies,
    const std::vector<float>& values) {
    
    std::vector<TrainingSample> samples;
    samples.reserve(positions.size());
    
    for (size_t i = 0; i < positions.size() && i < policies.size(); i++) {
        samples.emplace_back(positions[i], policies[i], 0.0f);
    }
    
    return samples;
}

void Trainer::runTraining(int numEpochs) {
    if (replayBuffer_->size() < static_cast<size_t>(config_.minSamplesBeforeTraining)) {
        std::cout << "  Buffer too small (" << replayBuffer_->size() 
                  << "/" << config_.minSamplesBeforeTraining << "), skipping training\n";
        return;
    }
    
    float totalLoss = 0.0f;
    int numBatches = 0;
    
    for (int epoch = 0; epoch < numEpochs && running_ && !g_shutdownRequested; epoch++) {
        // Sample a batch
        auto batch = replayBuffer_->sample(config_.batchSize);
        
        // Train on batch
        trainOnBatch(batch);
        float loss = calculateLoss(batch);
        
        totalLoss += loss;
        numBatches++;
        
        std::cout << "\r  Epoch " << (epoch + 1) << "/" << numEpochs 
                  << " - Loss: " << std::fixed << std::setprecision(4) << loss
                  << "        " << std::flush;
    }
    
    if (numBatches > 0) {
        stats_.currentLoss = totalLoss / numBatches;
    }
    std::cout << "\n";
}

void Trainer::trainOnBatch(const std::vector<TrainingSample>& batch) {
    if (batch.empty()) return;
    
    // Convert batch to tensors
    std::vector<torch::Tensor> inputs, policies, values;
    
    for (const auto& sample : batch) {
        // Input tensor
        torch::Tensor input = torch::from_blob(
            const_cast<float*>(sample.boardInput.data()),
            {nn::INPUT_CHANNELS, 8, 8},
            torch::kFloat32).clone();
        inputs.push_back(input);
        
        // Policy tensor
        torch::Tensor policy = torch::from_blob(
            const_cast<float*>(sample.policy.data()),
            {nn::POLICY_OUTPUT_SIZE},
            torch::kFloat32).clone();
        policies.push_back(policy);
        
        // Value tensor
        values.push_back(torch::tensor({sample.value}));
    }
    
    // Stack into batches
    torch::Tensor inputBatch = torch::stack(inputs);
    torch::Tensor policyBatch = torch::stack(policies);
    torch::Tensor valueBatch = torch::cat(values);
    
    // Train
    network_->trainBatch(inputBatch, policyBatch, valueBatch, config_.learningRate);
}

float Trainer::calculateLoss(const std::vector<TrainingSample>& batch) {
    // Simplified loss calculation - would need proper forward pass
    return stats_.currentLoss;  // Return last known loss
}

float Trainer::evaluateAgainstStockfish(int numGames) {
    try {
        StockfishInterface stockfish(config_.stockfishPath);
        if (!stockfish.isReady()) {
            std::cout << "  Stockfish not available, skipping evaluation\n";
            return stats_.currentWinRate;
        }
        
        mcts::MCTSConfig mctsConfig;
        mctsConfig.numSimulations = config_.mctsSimulations;
        mctsConfig.addNoise = false;
        
        mcts::MCTSEngine engine(network_, mctsConfig);
        
        int wins = 0, losses = 0, draws = 0;
        
        for (int i = 0; i < numGames && running_ && !g_shutdownRequested; i++) {
            bool wePlayWhite = (i % 2 == 0);
            auto result = stockfish.playGame(engine, wePlayWhite);
            
            if (result == StockfishInterface::GameResult::Win) wins++;
            else if (result == StockfishInterface::GameResult::Loss) losses++;
            else draws++;
            
            std::cout << "\r  Game " << (i + 1) << "/" << numGames 
                      << " - W:" << wins << " D:" << draws << " L:" << losses
                      << "        " << std::flush;
        }
        
        std::cout << "\n";
        
        float winRate = static_cast<float>(wins + 0.5f * draws) / 
                        (wins + draws + losses);
        return winRate;
        
    } catch (const std::exception& e) {
        std::cout << "  Evaluation error: " << e.what() << "\n";
        return stats_.currentWinRate;
    }
}

void Trainer::saveCheckpoint(const std::string& path) {
    std::cout << "  Saving checkpoint to " << path << "...\n";
    
    // Save model
    network_->save(path);
    
    // Save replay buffer
    std::string bufferPath = config_.checkpointDir + "/replay_buffer.bin";
    replayBuffer_->save(bufferPath);
    
    // Save stats
    std::string statsPath = config_.checkpointDir + "/stats.json";
    stats_.save(statsPath);
    
    std::cout << "  Checkpoint saved!\n";
}

void Trainer::loadCheckpoint(const std::string& path) {
    // Load model
    network_->load(path);
    
    // Load replay buffer
    std::string bufferPath = config_.checkpointDir + "/replay_buffer.bin";
    if (fs::exists(bufferPath)) {
        replayBuffer_->load(bufferPath);
    }
    
    // Load stats
    std::string statsPath = config_.checkpointDir + "/stats.json";
    if (fs::exists(statsPath)) {
        stats_.load(statsPath);
    }
    
    std::cout << "Checkpoint loaded! Resuming from iteration " << stats_.currentIteration << "\n";
}

void Trainer::printStatus() {
    auto elapsed = stats_.getSessionTime();
    auto hours = std::chrono::duration_cast<std::chrono::hours>(elapsed).count();
    auto mins = std::chrono::duration_cast<std::chrono::minutes>(elapsed).count() % 60;
    auto secs = elapsed.count() % 60;
    
    std::cout << "\n========================================\n";
    std::cout << "  Iteration " << stats_.currentIteration;
    std::cout << " | Elapsed: " << hours << "h " << mins << "m " << secs << "s\n";
    std::cout << "  Games: " << stats_.totalGamesPlayed;
    std::cout << " | Positions: " << stats_.totalPositionsTrained;
    std::cout << " | Buffer: " << replayBuffer_->size() << "\n";
    std::cout << "  Loss: " << std::fixed << std::setprecision(4) << stats_.currentLoss;
    std::cout << " | Win Rate: " << std::setprecision(1) << (stats_.currentWinRate * 100) << "%";
    std::cout << " | Best: " << (stats_.bestWinRate * 100) << "%\n";
    std::cout << "========================================\n";
}

void Trainer::printSummary() {
    auto total = stats_.totalTrainingTime;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(total).count();
    auto mins = std::chrono::duration_cast<std::chrono::minutes>(total).count() % 60;
    
    std::cout << "\n========================================\n";
    std::cout << "  Training Summary\n";
    std::cout << "========================================\n";
    std::cout << "  Total iterations: " << stats_.currentIteration << "\n";
    std::cout << "  Total games: " << stats_.totalGamesPlayed << "\n";
    std::cout << "  Total positions: " << stats_.totalPositionsTrained << "\n";
    std::cout << "  Total time: " << hours << "h " << mins << "m\n";
    std::cout << "  Best win rate: " << (stats_.bestWinRate * 100) << "%\n";
    std::cout << "========================================\n";
    std::cout << "  Goodbye!\n";
    std::cout << "========================================\n";
}

// ============================================================================
// StockfishInterface implementation
// ============================================================================

StockfishInterface::StockfishInterface(const std::string &path)
    : stockfishPath_(path)
{
    init();
}

StockfishInterface::~StockfishInterface()
{
    cleanup();
}

#ifdef _WIN32

void StockfishInterface::init()
{
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(SECURITY_ATTRIBUTES);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE inputReadTmp, inputWrite;
    HANDLE outputRead, outputWriteTmp;

    if (!CreatePipe(&inputReadTmp, &inputWrite, &sa, 0))
        return;
    if (!CreatePipe(&outputRead, &outputWriteTmp, &sa, 0))
    {
        CloseHandle(inputReadTmp);
        CloseHandle(inputWrite);
        return;
    }

    HANDLE inputRead, outputWrite;
    DuplicateHandle(GetCurrentProcess(), inputReadTmp, GetCurrentProcess(),
                    &inputRead, 0, TRUE, DUPLICATE_SAME_ACCESS);
    DuplicateHandle(GetCurrentProcess(), outputWriteTmp, GetCurrentProcess(),
                    &outputWrite, 0, TRUE, DUPLICATE_SAME_ACCESS);
    CloseHandle(inputReadTmp);
    CloseHandle(outputWriteTmp);

    STARTUPINFOA si = {};
    si.cb = sizeof(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = inputRead;
    si.hStdOutput = outputWrite;
    si.hStdError = outputWrite;

    PROCESS_INFORMATION pi = {};

    std::string cmd = stockfishPath_;
    if (!CreateProcessA(nullptr, const_cast<char *>(cmd.c_str()), nullptr, nullptr,
                        TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
    {
        CloseHandle(inputRead);
        CloseHandle(inputWrite);
        CloseHandle(outputRead);
        CloseHandle(outputWrite);
        return;
    }

    CloseHandle(inputRead);
    CloseHandle(outputWrite);
    CloseHandle(pi.hThread);

    processHandle_ = pi.hProcess;
    inputWrite_ = inputWrite;
    outputRead_ = outputRead;

    // Initialize UCI
    send("uci");
    receive("uciok");
    send("isready");
    receive("readyok");

    ready_ = true;
}

void StockfishInterface::cleanup()
{
    if (processHandle_)
    {
        send("quit");
        WaitForSingleObject(static_cast<HANDLE>(processHandle_), 1000);
        TerminateProcess(static_cast<HANDLE>(processHandle_), 0);
        CloseHandle(static_cast<HANDLE>(processHandle_));
        CloseHandle(static_cast<HANDLE>(inputWrite_));
        CloseHandle(static_cast<HANDLE>(outputRead_));
    }
}

void StockfishInterface::send(const std::string &cmd)
{
    std::string line = cmd + "\n";
    DWORD written;
    WriteFile(static_cast<HANDLE>(inputWrite_), line.c_str(), static_cast<DWORD>(line.size()), &written, nullptr);
}

std::string StockfishInterface::receive(const std::string &expected, int timeoutMs)
{
    std::string result;
    char buffer[4096];
    DWORD bytesRead;

    auto start = std::chrono::steady_clock::now();

    while (true)
    {
        DWORD available;
        PeekNamedPipe(static_cast<HANDLE>(outputRead_), nullptr, 0, nullptr, &available, nullptr);

        if (available > 0)
        {
            if (ReadFile(static_cast<HANDLE>(outputRead_), buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0)
            {
                buffer[bytesRead] = '\0';
                result += buffer;

                if (!expected.empty() && result.find(expected) != std::string::npos)
                {
                    break;
                }
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeoutMs)
        {
            break;
        }

        Sleep(1);
    }

    return result;
}

#else // Unix implementation

void StockfishInterface::init()
{
    if (pipe(toPipe_) == -1 || pipe(fromPipe_) == -1)
        return;

    pid_ = fork();
    if (pid_ == -1)
        return;

    if (pid_ == 0)
    {
        // Child process
        close(toPipe_[1]);
        close(fromPipe_[0]);

        dup2(toPipe_[0], STDIN_FILENO);
        dup2(fromPipe_[1], STDOUT_FILENO);
        dup2(fromPipe_[1], STDERR_FILENO);

        close(toPipe_[0]);
        close(fromPipe_[1]);

        execlp(stockfishPath_.c_str(), stockfishPath_.c_str(), nullptr);
        _exit(1);
    }

    // Parent process
    close(toPipe_[0]);
    close(fromPipe_[1]);

    // Set non-blocking
    int flags = fcntl(fromPipe_[0], F_GETFL, 0);
    fcntl(fromPipe_[0], F_SETFL, flags | O_NONBLOCK);

    send("uci");
    receive("uciok");
    send("isready");
    receive("readyok");

    ready_ = true;
}

void StockfishInterface::cleanup()
{
    if (pid_ > 0)
    {
        send("quit");
        usleep(100000);
        kill(pid_, SIGTERM);
        waitpid(pid_, nullptr, 0);
        close(toPipe_[1]);
        close(fromPipe_[0]);
    }
}

void StockfishInterface::send(const std::string &cmd)
{
    std::string line = cmd + "\n";
    write(toPipe_[1], line.c_str(), line.size());
}

std::string StockfishInterface::receive(const std::string &expected, int timeoutMs)
{
    std::string result;
    char buffer[4096];

    auto start = std::chrono::steady_clock::now();

    while (true)
    {
        ssize_t bytesRead = read(fromPipe_[0], buffer, sizeof(buffer) - 1);

        if (bytesRead > 0)
        {
            buffer[bytesRead] = '\0';
            result += buffer;

            if (!expected.empty() && result.find(expected) != std::string::npos)
            {
                break;
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > timeoutMs)
        {
            break;
        }

        usleep(1000);
    }

    return result;
}

#endif

Move StockfishInterface::getBestMove(const Board &board, int depth, int moveTimeMs)
{
    if (!ready_)
        return NULL_MOVE;

    send("position fen " + board.toFEN());

    std::string goCmd = "go";
    if (moveTimeMs > 0)
    {
        goCmd += " movetime " + std::to_string(moveTimeMs);
    }
    else
    {
        goCmd += " depth " + std::to_string(depth);
    }

    send(goCmd);

    std::string response = receive("bestmove", 30000);

    size_t pos = response.find("bestmove ");
    if (pos == std::string::npos)
        return NULL_MOVE;

    std::string moveStr = response.substr(pos + 9, 5);
    // Trim to actual move length
    size_t spacePos = moveStr.find(' ');
    if (spacePos != std::string::npos)
    {
        moveStr = moveStr.substr(0, spacePos);
    }

    return Move::fromUCI(moveStr);
}

float StockfishInterface::getEvaluation(const Board &board, int depth)
{
    if (!ready_)
        return 0;

    send("position fen " + board.toFEN());
    send("go depth " + std::to_string(depth));

    std::string response = receive("bestmove", 30000);

    // Parse score from info lines
    float score = 0;
    size_t scorePos = response.rfind("score cp ");
    if (scorePos != std::string::npos)
    {
        score = std::stof(response.substr(scorePos + 9)) / 100.0f;
    }

    scorePos = response.rfind("score mate ");
    if (scorePos != std::string::npos)
    {
        int mateIn = std::stoi(response.substr(scorePos + 11));
        score = (mateIn > 0) ? 100.0f : -100.0f;
    }

    return score;
}

StockfishInterface::GameResult StockfishInterface::playGame(
    mcts::MCTSEngine &ourEngine, bool wePlayWhite, int maxMoves)
{

    Board board;
    board.setStartingPosition();
    std::vector<StateInfo> stateHistory;  // Keep states alive during game

    ourEngine.newGame();
    send("ucinewgame");
    send("isready");
    receive("readyok");

    for (int moveNum = 0; moveNum < maxMoves; moveNum++)
    {
        chess::GameResult result = board.result();
        if (result != chess::GameResult::Ongoing)
        {
            if (result == chess::GameResult::Draw)
            {
                return GameResult::Draw;
            }
            bool whiteWon = (result == chess::GameResult::WhiteWins);
            return (whiteWon == wePlayWhite) ? GameResult::Win : GameResult::Loss;
        }

        Move move;
        bool ourTurn = (board.sideToMove() == Color::White) == wePlayWhite;

        if (ourTurn)
        {
            move = ourEngine.search(board);
        }
        else
        {
            move = getBestMove(board, 10, 100);
        }

        if (move.isNull())
        {
            return GameResult::Draw; // Something went wrong
        }

        stateHistory.emplace_back();
        board.makeMove(move, stateHistory.back());

        if (ourTurn)
        {
            ourEngine.advanceTree(move);
        }
    }

    // Max moves reached - draw
    return GameResult::Draw;
}

} // namespace training
} // namespace chess
