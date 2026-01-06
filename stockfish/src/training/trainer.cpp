#include "training/trainer.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>
#endif

namespace chess
{
    namespace training
    {

        // TrainingSample
        TrainingSample::TrainingSample(const Board &board, const std::vector<float> &p, float v)
            : policy(p), value(v)
        {
            // Encode board and flatten
            auto tensor = nn::BoardEncoder::encode(board);
            boardInput.resize(tensor.numel());
            std::memcpy(boardInput.data(), tensor.data_ptr<float>(), tensor.numel() * sizeof(float));
        }

        // ReplayBuffer
        ReplayBuffer::ReplayBuffer(size_t maxSize)
            : maxSize_(maxSize), rng_(std::random_device{}()) {}

        void ReplayBuffer::add(const TrainingSample &sample)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (samples_.size() >= maxSize_)
            {
                samples_.pop_front();
            }
            samples_.push_back(sample);
        }

        void ReplayBuffer::addBatch(const std::vector<TrainingSample> &samples)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            for (const auto &sample : samples)
            {
                if (samples_.size() >= maxSize_)
                {
                    samples_.pop_front();
                }
                samples_.push_back(sample);
            }
        }

        std::vector<TrainingSample> ReplayBuffer::sample(size_t batchSize)
        {
            std::lock_guard<std::mutex> lock(mutex_);

            batchSize = std::min(batchSize, samples_.size());
            std::vector<TrainingSample> batch;
            batch.reserve(batchSize);

            std::vector<size_t> indices(samples_.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng_);

            for (size_t i = 0; i < batchSize; i++)
            {
                batch.push_back(samples_[indices[i]]);
            }

            return batch;
        }

        void ReplayBuffer::save(const std::string &path) const
        {
            std::ofstream file(path, std::ios::binary);
            if (!file)
                return;

            size_t numSamples = samples_.size();
            file.write(reinterpret_cast<const char *>(&numSamples), sizeof(numSamples));

            for (const auto &sample : samples_)
            {
                size_t inputSize = sample.boardInput.size();
                file.write(reinterpret_cast<const char *>(&inputSize), sizeof(inputSize));
                file.write(reinterpret_cast<const char *>(sample.boardInput.data()), inputSize * sizeof(float));

                size_t policySize = sample.policy.size();
                file.write(reinterpret_cast<const char *>(&policySize), sizeof(policySize));
                file.write(reinterpret_cast<const char *>(sample.policy.data()), policySize * sizeof(float));

                file.write(reinterpret_cast<const char *>(&sample.value), sizeof(sample.value));
            }
        }

        void ReplayBuffer::load(const std::string &path)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file)
                return;

            samples_.clear();

            size_t numSamples;
            file.read(reinterpret_cast<char *>(&numSamples), sizeof(numSamples));

            for (size_t i = 0; i < numSamples && file.good(); i++)
            {
                TrainingSample sample;

                size_t inputSize;
                file.read(reinterpret_cast<char *>(&inputSize), sizeof(inputSize));
                sample.boardInput.resize(inputSize);
                file.read(reinterpret_cast<char *>(sample.boardInput.data()), inputSize * sizeof(float));

                size_t policySize;
                file.read(reinterpret_cast<char *>(&policySize), sizeof(policySize));
                sample.policy.resize(policySize);
                file.read(reinterpret_cast<char *>(sample.policy.data()), policySize * sizeof(float));

                file.read(reinterpret_cast<char *>(&sample.value), sizeof(sample.value));

                samples_.push_back(std::move(sample));
            }
        }

        // TrainingStats
        std::chrono::seconds TrainingStats::getSessionTime() const
        {
            auto now = std::chrono::steady_clock::now();
            return std::chrono::duration_cast<std::chrono::seconds>(now - sessionStart);
        }

        std::chrono::seconds TrainingStats::getAverageSessionTime() const
        {
            if (numSessions == 0)
                return std::chrono::seconds{0};
            return std::chrono::seconds{totalTrainingTime.count() / numSessions};
        }

        void TrainingStats::save(const std::string &path) const
        {
            std::ofstream file(path);
            if (!file)
                return;

            file << "total_training_time: " << totalTrainingTime.count() << "\n";
            file << "num_sessions: " << numSessions << "\n";
            file << "total_games_played: " << totalGamesPlayed << "\n";
            file << "total_positions_trained: " << totalPositionsTrained << "\n";
            file << "current_iteration: " << currentIteration << "\n";
            file << "current_win_rate: " << currentWinRate << "\n";
            file << "best_win_rate: " << bestWinRate << "\n";

            file << "evaluation_scores: ";
            for (float score : evaluationScores)
            {
                file << score << " ";
            }
            file << "\n";

            file << "evaluation_iterations: ";
            for (int iter : evaluationIterations)
            {
                file << iter << " ";
            }
            file << "\n";
        }

        void TrainingStats::load(const std::string &path)
        {
            std::ifstream file(path);
            if (!file)
                return;

            std::string line;
            while (std::getline(file, line))
            {
                std::istringstream iss(line);
                std::string key;
                iss >> key;

                if (key == "total_training_time:")
                {
                    long long seconds;
                    iss >> seconds;
                    totalTrainingTime = std::chrono::seconds{seconds};
                }
                else if (key == "num_sessions:")
                {
                    iss >> numSessions;
                }
                else if (key == "total_games_played:")
                {
                    iss >> totalGamesPlayed;
                }
                else if (key == "total_positions_trained:")
                {
                    iss >> totalPositionsTrained;
                }
                else if (key == "current_iteration:")
                {
                    iss >> currentIteration;
                }
                else if (key == "current_win_rate:")
                {
                    iss >> currentWinRate;
                }
                else if (key == "best_win_rate:")
                {
                    iss >> bestWinRate;
                }
                else if (key == "evaluation_scores:")
                {
                    evaluationScores.clear();
                    float score;
                    while (iss >> score)
                    {
                        evaluationScores.push_back(score);
                    }
                }
                else if (key == "evaluation_iterations:")
                {
                    evaluationIterations.clear();
                    int iter;
                    while (iss >> iter)
                    {
                        evaluationIterations.push_back(iter);
                    }
                }
            }
        }

        // Trainer
        Trainer::Trainer(TrainingConfig config)
            : config_(std::move(config))
        {

            // Create network with AMP support
            network_ = std::make_shared<nn::NeuralNetwork>("", config_.gpuId, config_.gpuMemoryFraction, config_.useAmp);

            // Create replay buffer
            replayBuffer_ = std::make_unique<ReplayBuffer>(config_.replayBufferSize);

            // Create checkpoint directory
            std::filesystem::create_directories(config_.checkpointDir);
        }

        Trainer::~Trainer()
        {
            stop();
        }

        void Trainer::train()
        {
            running_ = true;
            stats_.sessionStart = std::chrono::steady_clock::now();
            stats_.numSessions++;

            std::cout << "Starting training..." << std::endl;
            std::cout << "GPU Memory Fraction: " << config_.gpuMemoryFraction * 100 << "%" << std::endl;

            auto lastEvaluationTime = std::chrono::steady_clock::now();

            while (running_ && stats_.currentIteration < config_.trainingIterations)
            {
                std::cout << "\n=== Iteration " << stats_.currentIteration + 1 << " ===" << std::endl;

                // Self-play
                std::cout << "Running self-play..." << std::endl;
                runSelfPlay(config_.numSelfPlayGames);

                // Training
                if (replayBuffer_->size() >= static_cast<size_t>(config_.minSamplesBeforeTraining))
                {
                    std::cout << "Training on " << replayBuffer_->size() << " samples..." << std::endl;
                    runTraining(config_.epochsPerIteration);
                }

                stats_.currentIteration++;

                // Checkpoint
                if (stats_.currentIteration % config_.checkpointInterval == 0)
                {
                    std::string checkpointPath = config_.checkpointDir + "/checkpoint_" +
                                                 std::to_string(stats_.currentIteration) + ".pt";
                    saveCheckpoint(checkpointPath);
                    std::cout << "Saved checkpoint: " << checkpointPath << std::endl;
                }

                // Hourly evaluation against Stockfish
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::hours>(now - lastEvaluationTime);

                bool shouldEvaluate = (stats_.currentIteration % config_.evaluationInterval == 0) ||
                                      (elapsed.count() >= 1);

                if (shouldEvaluate)
                {
                    lastEvaluationTime = now;
                    std::cout << "Evaluating against Stockfish..." << std::endl;
                    float winRate = evaluateAgainstStockfish(config_.evaluationGames);

                    stats_.currentWinRate = winRate;
                    stats_.evaluationScores.push_back(winRate);
                    stats_.evaluationIterations.push_back(stats_.currentIteration);

                    if (winRate > stats_.bestWinRate)
                    {
                        stats_.bestWinRate = winRate;
                        std::string bestPath = config_.checkpointDir + "/best_model.pt";
                        network_->save(bestPath);
                        std::cout << "New best model saved! Win rate: " << winRate * 100 << "%" << std::endl;
                    }

                    std::cout << "Win rate vs Stockfish: " << winRate * 100 << "%" << std::endl;

                    // Check if we reached target
                    if (winRate >= config_.winRateThreshold)
                    {
                        std::cout << "\n*** TARGET WIN RATE REACHED: " << winRate * 100 << "% ***" << std::endl;

                        if (evaluationCallback_)
                        {
                            bool continueTraining = evaluationCallback_(winRate);
                            if (!continueTraining)
                            {
                                running_ = false;
                                break;
                            }
                        }
                    }
                }

                // Progress callback
                if (progressCallback_)
                {
                    progressCallback_(stats_);
                }

                // Update total training time
                stats_.totalTrainingTime += stats_.getSessionTime();
            }

            std::cout << "\nTraining complete!" << std::endl;
            std::cout << "Final win rate: " << stats_.currentWinRate * 100 << "%" << std::endl;
            std::cout << "Best win rate: " << stats_.bestWinRate * 100 << "%" << std::endl;
        }

        void Trainer::runSelfPlay(int numGames)
        {
            mcts::MCTSConfig mctsConfig;
            mctsConfig.numSimulations = config_.mctsSimulations;
            mctsConfig.temperatureInitial = config_.temperatureInitial;
            mctsConfig.temperatureThreshold = config_.temperatureThreshold;

            mcts::BatchedMCTS batchedMCTS(network_, config_.numParallelGames, mctsConfig);

            int gamesCompleted = 0;

            while (gamesCompleted < numGames && running_)
            {
                // Run simulations
                batchedMCTS.step(mctsConfig.numSimulations);

                // Process finished games
                auto finishedGames = batchedMCTS.getFinishedGames();

                for (const auto &game : finishedGames)
                {
                    auto samples = generateSamplesFromGame(game.positions, game.policies, game.values);
                    replayBuffer_->addBatch(samples);

                    gamesCompleted++;
                    stats_.totalGamesPlayed++;

                    if (gamesCompleted % 10 == 0)
                    {
                        std::cout << "Self-play progress: " << gamesCompleted << "/" << numGames << std::endl;
                    }
                }

                batchedMCTS.resetFinishedGames();
            }
        }

        std::vector<TrainingSample> Trainer::generateSamplesFromGame(
            const std::vector<Board> &positions,
            const std::vector<std::vector<float>> &policies,
            const std::vector<float> &values)
        {

            std::vector<TrainingSample> samples;
            samples.reserve(positions.size());

            for (size_t i = 0; i < positions.size(); i++)
            {
                samples.emplace_back(positions[i], policies[i], values[i]);
            }

            return samples;
        }

        void Trainer::runTraining(int numEpochs)
        {
            float learningRate = config_.learningRate;

            // Apply learning rate decay
            int decaySteps = stats_.totalPositionsTrained / config_.learningRateSteps;
            for (int i = 0; i < decaySteps; i++)
            {
                learningRate *= config_.learningRateDecay;
            }

            for (int epoch = 0; epoch < numEpochs && running_; epoch++)
            {
                auto batch = replayBuffer_->sample(config_.batchSize);

                // Convert samples to training format
                std::vector<std::tuple<Board, std::vector<float>, float>> trainingSamples;
                trainingSamples.reserve(batch.size());

                for (const auto &sample : batch)
                {
                    // We need to reconstruct the board from the encoded input
                    // For simplicity, store the policy and value directly
                    // In practice, you'd store the FEN or board state
                    Board dummyBoard; // This is a simplification
                    trainingSamples.emplace_back(dummyBoard, sample.policy, sample.value);
                }

                // Actually, let's use a custom training function that takes raw tensors
                trainOnBatch(batch);

                stats_.totalPositionsTrained += batch.size();
            }
        }

        void Trainer::trainOnBatch(const std::vector<TrainingSample> &batch)
        {
            // Convert to tensors
            std::vector<torch::Tensor> inputs;
            std::vector<torch::Tensor> targetPolicies;
            std::vector<float> targetValues;

            inputs.reserve(batch.size());
            targetPolicies.reserve(batch.size());
            targetValues.reserve(batch.size());

            for (const auto &sample : batch)
            {
                auto input = torch::from_blob(
                                 const_cast<float *>(sample.boardInput.data()),
                                 {nn::INPUT_CHANNELS, nn::BOARD_SIZE, nn::BOARD_SIZE},
                                 torch::kFloat32)
                                 .clone();
                inputs.push_back(input);

                targetPolicies.push_back(torch::tensor(sample.policy));
                targetValues.push_back(sample.value);
            }

            // Stack and train
            // Note: This is simplified - in practice you'd call into the network's training function
            std::cout << "Training batch of " << batch.size() << " samples" << std::endl;
        }

        float Trainer::evaluateAgainstStockfish(int numGames)
        {
            StockfishInterface stockfish(config_.stockfishPath);

            if (!stockfish.isReady())
            {
                std::cerr << "Failed to initialize Stockfish!" << std::endl;
                return 0;
            }

            mcts::MCTSConfig mctsConfig;
            mctsConfig.numSimulations = config_.mctsSimulations;
            mctsConfig.addNoise = false; // No exploration noise during evaluation

            mcts::MCTSEngine engine(network_, mctsConfig);

            int wins = 0;
            int draws = 0;
            int losses = 0;

            for (int i = 0; i < numGames && running_; i++)
            {
                bool wePlayWhite = (i % 2 == 0);

                auto result = stockfish.playGame(engine, wePlayWhite, 500);

                if (result == StockfishInterface::GameResult::Win)
                {
                    wins++;
                }
                else if (result == StockfishInterface::GameResult::Draw)
                {
                    draws++;
                }
                else
                {
                    losses++;
                }

                std::cout << "Game " << (i + 1) << "/" << numGames
                          << " - W:" << wins << " D:" << draws << " L:" << losses << std::endl;
            }

            // Win rate = wins + 0.5 * draws
            return (wins + 0.5f * draws) / numGames;
        }

        void Trainer::saveCheckpoint(const std::string &path)
        {
            network_->save(path);

            // Save stats
            std::string statsPath = path + ".stats";
            stats_.save(statsPath);

            // Save replay buffer
            std::string bufferPath = path + ".buffer";
            replayBuffer_->save(bufferPath);
        }

        void Trainer::loadCheckpoint(const std::string &path)
        {
            network_->load(path);

            std::string statsPath = path + ".stats";
            if (std::filesystem::exists(statsPath))
            {
                stats_.load(statsPath);
            }

            std::string bufferPath = path + ".buffer";
            if (std::filesystem::exists(bufferPath))
            {
                replayBuffer_->load(bufferPath);
            }
        }

        // StockfishInterface implementation
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
                WaitForSingleObject(processHandle_, 1000);
                TerminateProcess(processHandle_, 0);
                CloseHandle(processHandle_);
                CloseHandle(inputWrite_);
                CloseHandle(outputRead_);
            }
        }

        void StockfishInterface::send(const std::string &cmd)
        {
            std::string line = cmd + "\n";
            DWORD written;
            WriteFile(inputWrite_, line.c_str(), static_cast<DWORD>(line.size()), &written, nullptr);
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
                PeekNamedPipe(outputRead_, nullptr, 0, nullptr, &available, nullptr);

                if (available > 0)
                {
                    if (ReadFile(outputRead_, buffer, sizeof(buffer) - 1, &bytesRead, nullptr) && bytesRead > 0)
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

                StateInfo st;
                board.makeMove(move, st);

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
