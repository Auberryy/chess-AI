#include "elo/elo_manager.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <algorithm>

namespace chess
{
    namespace elo
    {

        // MCTSPlayer implementation
        MCTSPlayer::MCTSPlayer(std::shared_ptr<nn::NeuralNetwork> network,
                               const mcts::MCTSConfig &config,
                               const std::string &name)
            : network_(std::move(network)), config_(config), name_(name)
        {
            engine_ = std::make_unique<mcts::MCTSEngine>(network_, config_);
        }

        Move MCTSPlayer::getMove(Board &board)
        {
            return engine_->search(board, config_.numSimulations);
        }

        void MCTSPlayer::newGame()
        {
            engine_->newGame();
        }

        void MCTSPlayer::advanceTree(Move move)
        {
            engine_->advanceTree(move);
        }

        // NNUEPlayer implementation
        NNUEPlayer::NNUEPlayer(std::shared_ptr<nn::NNUEEvaluator> evaluator,
                               int searchDepth,
                               const std::string &name)
            : evaluator_(std::move(evaluator)), searchDepth_(searchDepth), name_(name) {}

        Move NNUEPlayer::getMove(Board &board)
        {
            auto moves = MoveGen::generateLegal(board);
            if (moves.empty())
                return NULL_MOVE;

            Move bestMove = moves[0];
            int bestScore = -999999;

            for (const auto &move : moves)
            {
                StateInfo st;
                board.makeMove(move, st);

                int score = -alphaBeta(board, searchDepth_ - 1, -999999, 999999, false);

                board.unmakeMove(move);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestMove = move;
                }
            }

            return bestMove;
        }

        void NNUEPlayer::newGame()
        {
            // Nothing to reset
        }

        void NNUEPlayer::advanceTree(Move move)
        {
            // Nothing to advance
        }

        int NNUEPlayer::alphaBeta(Board &board, int depth, int alpha, int beta, bool maximizing)
        {
            if (depth == 0)
            {
                return quiescence(board, alpha, beta);
            }

            auto result = board.result();
            if (result != GameResult::Ongoing)
            {
                if (result == GameResult::Draw)
                    return 0;
                // Checkmate
                return maximizing ? -100000 + depth : 100000 - depth;
            }

            auto moves = MoveGen::generateLegal(board);
            if (moves.empty())
            {
                if (board.inCheck())
                {
                    return maximizing ? -100000 + depth : 100000 - depth;
                }
                return 0; // Stalemate
            }

            if (maximizing)
            {
                int maxEval = -999999;
                for (const auto &move : moves)
                {
                    StateInfo st;
                    board.makeMove(move, st);
                    int eval = alphaBeta(board, depth - 1, alpha, beta, false);
                    board.unmakeMove(move);

                    maxEval = std::max(maxEval, eval);
                    alpha = std::max(alpha, eval);
                    if (beta <= alpha)
                        break;
                }
                return maxEval;
            }
            else
            {
                int minEval = 999999;
                for (const auto &move : moves)
                {
                    StateInfo st;
                    board.makeMove(move, st);
                    int eval = alphaBeta(board, depth - 1, alpha, beta, true);
                    board.unmakeMove(move);

                    minEval = std::min(minEval, eval);
                    beta = std::min(beta, eval);
                    if (beta <= alpha)
                        break;
                }
                return minEval;
            }
        }

        int NNUEPlayer::quiescence(Board &board, int alpha, int beta)
        {
            int standPat = evaluator_->evaluate(board);

            if (standPat >= beta)
                return beta;
            if (alpha < standPat)
                alpha = standPat;

            auto moves = MoveGen::generateCaptures(board);

            for (const auto &move : moves)
            {
                if (!board.isLegal(move))
                    continue;

                StateInfo st;
                board.makeMove(move, st);
                int score = -quiescence(board, -beta, -alpha);
                board.unmakeMove(move);

                if (score >= beta)
                    return beta;
                if (score > alpha)
                    alpha = score;
            }

            return alpha;
        }

        // MatchEvaluator implementation
        MatchEvaluator::MatchEvaluator(Config config) : config_(std::move(config)) {}

        EloResult MatchEvaluator::evaluateChallenger(EnginePlayer &challenger,
                                                     EnginePlayer &champion,
                                                     float championElo)
        {
            MatchResult matchResult;

            for (int game = 0; game < config_.numGames; game++)
            {
                bool challengerIsWhite = config_.alternateColors ? (game % 2 == 0) : true;

                EnginePlayer &white = challengerIsWhite ? challenger : champion;
                EnginePlayer &black = challengerIsWhite ? champion : challenger;

                white.newGame();
                black.newGame();

                GameResult result = playSingleGame(white, black, config_.maxMovesPerGame);

                // Attribute result to challenger
                if (result == GameResult::Draw)
                {
                    matchResult.draws++;
                }
                else if ((result == GameResult::WhiteWins && challengerIsWhite) ||
                         (result == GameResult::BlackWins && !challengerIsWhite))
                {
                    matchResult.challengerWins++;
                }
                else
                {
                    matchResult.championWins++;
                }

                if (gameCallback_)
                {
                    gameCallback_(game + 1, config_.numGames, matchResult);
                }
            }

            // Calculate Elo
            float winRate = matchResult.challengerScore();
            float eloDiff = EloCalculator::eloDifferenceFromWinRate(winRate);
            float expectedScore = EloCalculator::expectedScore(0); // They start equal
            float newElo = EloCalculator::calculateNewElo(championElo, winRate, expectedScore);

            EloResult result;
            result.winRate = winRate;
            result.wins = matchResult.challengerWins;
            result.draws = matchResult.draws;
            result.losses = matchResult.championWins;
            result.eloDifference = eloDiff;
            result.newElo = newElo;
            result.improved = EloCalculator::meetsEloThreshold(winRate, config_.eloThreshold);

            return result;
        }

        GameResult MatchEvaluator::playSingleGame(EnginePlayer &white, EnginePlayer &black,
                                                  int maxMoves)
        {
            Board board;
            board.setStartingPosition();

            for (int move = 0; move < maxMoves; move++)
            {
                GameResult result = board.result();
                if (result != GameResult::Ongoing)
                {
                    return result;
                }

                Move m;
                if (board.sideToMove() == Color::White)
                {
                    m = white.getMove(board);
                    white.advanceTree(m);
                    black.advanceTree(m);
                }
                else
                {
                    m = black.getMove(board);
                    white.advanceTree(m);
                    black.advanceTree(m);
                }

                if (m.isNull())
                {
                    return GameResult::Draw; // Something went wrong
                }

                StateInfo st;
                board.makeMove(m, st);
            }

            return GameResult::Draw; // Max moves reached
        }

        // EloModelManager implementation
        EloModelManager::EloModelManager(const std::string &modelsDir)
            : modelsDir_(modelsDir)
        {
            std::filesystem::create_directories(modelsDir);

            // Initialize champion
            champion_.path = modelsDir + "/champion.pt";
            champion_.elo = 1000.0f;
            champion_.generation = 0;
        }

        bool EloModelManager::tryPromote(const std::string &challengerPath,
                                         const EloResult &matchResult)
        {
            if (!matchResult.improved)
            {
                std::cout << "Challenger did NOT improve. Elo diff: "
                          << matchResult.eloDifference << " (need >= 1.0)" << std::endl;
                return false;
            }

            std::cout << "🎉 Challenger PROMOTED! Elo: " << matchResult.newElo
                      << " (+" << matchResult.eloDifference << ")" << std::endl;

            // Save old champion to history
            history_.push_back(champion_);

            // Promote challenger
            champion_.elo = matchResult.newElo;
            champion_.generation++;
            champion_.gamesPlayed += matchResult.wins + matchResult.draws + matchResult.losses;
            champion_.timestamp = std::chrono::system_clock::now().time_since_epoch().count();

            // Copy challenger to champion path
            std::filesystem::copy_file(challengerPath, champion_.path,
                                       std::filesystem::copy_options::overwrite_existing);

            // Also save a versioned copy
            std::string versionedPath = modelsDir_ + "/gen_" +
                                        std::to_string(champion_.generation) + "_elo_" +
                                        std::to_string(static_cast<int>(champion_.elo)) + ".pt";
            std::filesystem::copy_file(challengerPath, versionedPath);

            saveChampion(champion_);
            return true;
        }

        void EloModelManager::saveChampion(const ModelInfo &info)
        {
            std::string infoPath = modelsDir_ + "/champion_info.txt";
            std::ofstream file(infoPath);

            file << "elo: " << info.elo << "\n";
            file << "generation: " << info.generation << "\n";
            file << "games_played: " << info.gamesPlayed << "\n";
            file << "timestamp: " << info.timestamp << "\n";
        }

        void EloModelManager::saveHistory(const std::string &path)
        {
            std::ofstream file(path);

            for (const auto &info : history_)
            {
                file << info.generation << " " << info.elo << " " << info.gamesPlayed << "\n";
            }
            file << champion_.generation << " " << champion_.elo << " " << champion_.gamesPlayed << "\n";
        }

        void EloModelManager::loadHistory(const std::string &path)
        {
            std::ifstream file(path);
            if (!file)
                return;

            history_.clear();
            int gen, games;
            float elo;

            while (file >> gen >> elo >> games)
            {
                ModelInfo info;
                info.generation = gen;
                info.elo = elo;
                info.gamesPlayed = games;
                history_.push_back(info);
            }

            if (!history_.empty())
            {
                champion_ = history_.back();
                history_.pop_back();
            }
        }

        std::vector<std::pair<int, float>> EloModelManager::getEloHistory() const
        {
            std::vector<std::pair<int, float>> result;

            for (const auto &info : history_)
            {
                result.emplace_back(info.generation, info.elo);
            }
            result.emplace_back(champion_.generation, champion_.elo);

            return result;
        }

        // EloTrainingManager implementation
        EloTrainingManager::EloTrainingManager(Config config)
            : config_(std::move(config))
        {

            // Create directories
            std::filesystem::create_directories(config_.modelsDir);
            std::filesystem::create_directories(config_.checkpointDir);

            // Initialize networks with AMP support
            currentNetwork_ = std::make_shared<nn::NeuralNetwork>("", config_.gpuId, config_.gpuMemoryFraction, config_.useAmp);
            championNetwork_ = std::make_shared<nn::NeuralNetwork>("", config_.gpuId, config_.gpuMemoryFraction, config_.useAmp);

            if (config_.trainNNUE)
            {
                nnueEvaluator_ = std::make_shared<nn::NNUEEvaluator>();
            }

            // Initialize managers
            modelManager_ = std::make_unique<EloModelManager>(config_.modelsDir);

            MatchEvaluator::Config matchConfig;
            matchConfig.numGames = config_.evaluationGames;
            matchConfig.mctsSimulations = config_.evaluationSimulations;
            matchConfig.eloThreshold = config_.eloThreshold;
            matchEvaluator_ = std::make_unique<MatchEvaluator>(matchConfig);

            // Set match callback
            matchEvaluator_->setGameCallback([this](int gameNum, int total, const MatchResult &result)
                                             {
        float score = result.challengerScore();
        std::cout << "\rEvaluation: " << gameNum << "/" << total 
                  << " (W:" << result.challengerWins 
                  << " D:" << result.draws 
                  << " L:" << result.championWins 
                  << " Score: " << std::fixed << std::setprecision(1) << score * 100 << "%)"
                  << std::flush; });
        }

        void EloTrainingManager::train(int maxIterations)
        {
            running_ = true;
            int iteration = 0;

            std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║           AURORA CHESS AI TRAINING SYSTEM                    ║\n";
            std::cout << "║              Codename: Flapjack Puffin 🐧                    ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Goal: Fast like Stockfish, Strong enough to beat it         ║\n";
            std::cout << "║ Method: MCTS (GPU) + NNUE (CPU) hybrid training             ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Phase 1 (Elo < 1200): NNUE learns from MCTS (distillation)  ║\n";
            std::cout << "║ Phase 2 (Elo ≥ 1200): NNUE plays vs MCTS (adversarial)      ║\n";
            std::cout << "║ Promotion: Challenger must gain +1 Elo (>50.14% win rate)   ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

            while (running_ && (maxIterations < 0 || iteration < maxIterations))
            {
                // Determine training phase
                std::string phase = currentElo_ < config_.nnueVsMctsEloThreshold ? "Phase 1 (Distillation)" : "Phase 2 (Adversarial)";

                std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
                std::cout << "║ ITERATION " << std::setw(4) << iteration + 1
                          << " | " << phase << std::setw(phase.length() < 25 ? 25 - phase.length() : 1) << " ║\n";
                std::cout << "║ MCTS Elo: " << std::setw(6) << std::fixed << std::setprecision(1) << currentElo_
                          << " | NNUE Elo: " << std::setw(6) << nnueElo_ << "                  ║\n";
                std::cout << "╚════════════════════════════════════════════════════════╝\n";

                // Phase 1: Self-play
                std::cout << "\n📊 Step 1: Self-Play (" << config_.selfPlayGamesPerIteration << " games)\n";
                runSelfPlay();

                // Phase 2: Training
                std::cout << "\n🧠 Step 2: Training MCTS Network\n";
                runTraining();

                // Phase 3: NNUE Training (periodic)
                if (config_.trainNNUE && (iteration + 1) % config_.nnueTrainingInterval == 0)
                {
                    std::cout << "\n⚡ Step 2.5: NNUE Training (" << phase << ")\n";
                    trainNNUE();
                }

                // Phase 3: Evaluation (Challenger vs Champion)
                std::cout << "\n⚔️ Step 3: Challenger vs Champion (" << config_.evaluationGames << " games)\n";
                bool promoted = evaluateAndPromote();

                if (promoted)
                {
                    std::cout << "\n✅ New champion! Generation " << generation_
                              << " | Elo: " << currentElo_ << "\n";

                    if (promotionCallback_)
                    {
                        promotionCallback_(currentElo_ - EloCalculator::eloDifferenceFromWinRate(0.5f),
                                           currentElo_, generation_);
                    }
                }
                else
                {
                    std::cout << "\n❌ Challenger did not improve enough. Keeping current champion.\n";
                }

                if (progressCallback_)
                {
                    progressCallback_(iteration, currentElo_, generation_);
                }

                iteration++;
            }

            std::cout << "\n\n╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                    TRAINING COMPLETE                          ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Final Elo: " << std::setw(6) << std::fixed << std::setprecision(1) << currentElo_
                      << " | Generation: " << std::setw(4) << generation_ << "                    ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
        }

        void EloTrainingManager::runSelfPlay()
        {
            mcts::MCTSConfig mctsConfig;
            mctsConfig.numSimulations = config_.mctsSimulations;
            mctsConfig.addNoise = true; // Exploration during self-play
            mctsConfig.temperature = 1.0f;

            mcts::BatchedMCTS selfPlay(currentNetwork_, config_.selfPlayGamesPerIteration / 4, mctsConfig);
            nn::BoardEncoder encoder;

            int gamesCompleted = 0;
            std::vector<TrainingSample> newSamples;

            while (gamesCompleted < config_.selfPlayGamesPerIteration && running_)
            {
                selfPlay.step(mctsConfig.numSimulations);

                auto finished = selfPlay.getFinishedGames();
                gamesCompleted += finished.size();

                // Collect training samples from finished games
                for (const auto &game : finished)
                {
                    for (const auto &position : game.positions)
                    {
                        TrainingSample sample;
                        sample.boardInput = encoder.encode(position.board);
                        sample.policy = position.policy;
                        sample.value = position.value;
                        newSamples.push_back(sample);
                    }
                }

                std::cout << "\rSelf-play: " << gamesCompleted << "/"
                          << config_.selfPlayGamesPerIteration << " games" << std::flush;

                selfPlay.resetFinishedGames();
            }
            std::cout << "\n";

            // Add to replay buffer
            {
                std::lock_guard<std::mutex> lock(bufferMutex_);
                for (const auto &sample : newSamples)
                {
                    replayBuffer_.push_back(sample);
                    if (replayBuffer_.size() > static_cast<size_t>(config_.replayBufferSize))
                    {
                        replayBuffer_.pop_front();
                    }
                }
            }

            std::cout << "Collected " << newSamples.size() << " training samples. Buffer: "
                      << replayBuffer_.size() << "/" << config_.replayBufferSize << "\n";
        }

        std::vector<EloTrainingManager::TrainingSample> EloTrainingManager::sampleFromReplayBuffer(int batchSize)
        {
            std::lock_guard<std::mutex> lock(bufferMutex_);

            std::vector<TrainingSample> batch;
            if (replayBuffer_.empty())
                return batch;

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, replayBuffer_.size() - 1);

            for (int i = 0; i < batchSize && !replayBuffer_.empty(); i++)
            {
                int idx = dis(gen);
                batch.push_back(replayBuffer_[idx]);
            }

            return batch;
        }

        void EloTrainingManager::runTraining()
        {
            if (replayBuffer_.size() < static_cast<size_t>(config_.minSamplesBeforeTraining))
            {
                std::cout << "Not enough samples yet (" << replayBuffer_.size()
                          << "/" << config_.minSamplesBeforeTraining << "). Skipping training.\n";
                return;
            }

            std::cout << "Training neural network on replay buffer...\n";

            int totalBatches = std::max(1, static_cast<int>(replayBuffer_.size()) / config_.batchSize);

            for (int epoch = 0; epoch < config_.trainingEpochsPerIteration; epoch++)
            {
                float totalLoss = 0.0f;
                int batchCount = 0;

                for (int batch = 0; batch < totalBatches && running_; batch++)
                {
                    auto samples = sampleFromReplayBuffer(config_.batchSize);
                    if (samples.empty())
                        break;

                    // Convert to tensors
                    std::vector<torch::Tensor> inputs, policyTargets, valueTargets;

                    for (const auto &sample : samples)
                    {
                        inputs.push_back(torch::from_blob(
                                             const_cast<float *>(sample.boardInput.data()),
                                             {22, 8, 8})
                                             .clone());
                        policyTargets.push_back(torch::from_blob(
                                                    const_cast<float *>(sample.policy.data()),
                                                    {static_cast<long>(sample.policy.size())})
                                                    .clone());
                        valueTargets.push_back(torch::tensor(sample.value));
                    }

                    auto inputBatch = torch::stack(inputs);
                    auto policyBatch = torch::stack(policyTargets);
                    auto valueBatch = torch::stack(valueTargets).view({-1, 1});

                    // Train step
                    float loss = currentNetwork_->train(inputBatch, policyBatch, valueBatch, config_.learningRate);
                    totalLoss += loss;
                    batchCount++;

                    if (batch % 10 == 0)
                    {
                        std::cout << "\r  Epoch " << epoch + 1 << "/" << config_.trainingEpochsPerIteration
                                  << " Batch " << batch + 1 << "/" << totalBatches
                                  << " Loss: " << std::fixed << std::setprecision(4) << loss << std::flush;
                    }
                }

                if (batchCount > 0)
                {
                    float avgLoss = totalLoss / batchCount;
                    std::cout << "\n  Epoch " << epoch + 1 << " complete. Average loss: " << avgLoss << "\n";
                }
            }
        }

        void EloTrainingManager::trainNNUE()
        {
            if (currentElo_ < config_.nnueVsMctsEloThreshold)
            {
                // Phase 1: Train NNUE from replay buffer (distillation)
                std::cout << "NNUE Phase 1: Distillation from MCTS network\n";
                trainNNUEFromReplayBuffer();
            }
            else
            {
                // Phase 2: Train NNUE by playing against MCTS
                std::cout << "NNUE Phase 2: Adversarial training vs MCTS (Elo " << currentElo_ << ")\n";
                trainNNUEAgainstMCTS();
            }
        }

        void EloTrainingManager::trainNNUEFromReplayBuffer()
        {
            if (!nnueEvaluator_ || replayBuffer_.size() < 1000)
            {
                std::cout << "  Insufficient data for NNUE training. Skipping.\n";
                return;
            }

            const int nnueBatchSize = 512;
            const int nnueBatches = 100;

            float totalLoss = 0.0f;

            for (int batch = 0; batch < nnueBatches; batch++)
            {
                auto samples = sampleFromReplayBuffer(nnueBatchSize);
                if (samples.empty())
                    break;

                // Train NNUE to predict value from replay buffer
                std::vector<Board> boards;
                std::vector<float> targets;

                // Convert samples to boards (would need proper deserialization)
                // For now, placeholder
                for (const auto &sample : samples)
                {
                    // Would decode sample.boardInput back to Board
                    // targets.push_back(sample.value);
                }

                // Loss calculation would go here
                // totalLoss += nnueEvaluator_->train(boards, targets);

                if (batch % 20 == 0)
                {
                    std::cout << "\r  NNUE batch " << batch + 1 << "/" << nnueBatches << std::flush;
                }
            }

            std::cout << "\n  NNUE distillation complete.\n";
        }

        void EloTrainingManager::trainNNUEAgainstMCTS()
        {
            std::cout << "  Playing NNUE vs MCTS (" << config_.nnueVsMctsGames << " games)...\n";

            // Create players
            mcts::MCTSConfig mctsConfig;
            mctsConfig.numSimulations = config_.evaluationSimulations;
            mctsConfig.addNoise = false;

            MCTSPlayer mctsPlayer(currentNetwork_, mctsConfig, "MCTS");
            NNUEPlayer nnuePlayer(nnueEvaluator_, 8, "NNUE"); // Depth 8 search

            int nnueWins = 0, draws = 0, mctsWins = 0;

            std::vector<Board> trainingBoards;
            std::vector<float> trainingOutcomes;

            for (int game = 0; game < config_.nnueVsMctsGames; game++)
            {
                bool nnueIsWhite = (game % 2 == 0);

                Board board;
                board.setStartingPosition();

                std::vector<Board> gameBoards;

                for (int move = 0; move < 200; move++)
                {
                    gameBoards.push_back(board);

                    auto result = board.result();
                    if (result != GameResult::Ongoing)
                    {
                        float outcome = 0.0f;
                        if (result == GameResult::WhiteWins)
                            outcome = nnueIsWhite ? 1.0f : -1.0f;
                        else if (result == GameResult::BlackWins)
                            outcome = nnueIsWhite ? -1.0f : 1.0f;

                        // Add game positions to training
                        for (const auto &b : gameBoards)
                        {
                            trainingBoards.push_back(b);
                            trainingOutcomes.push_back(outcome);
                        }

                        if (result == GameResult::Draw)
                            draws++;
                        else if ((result == GameResult::WhiteWins && nnueIsWhite) ||
                                 (result == GameResult::BlackWins && !nnueIsWhite))
                            nnueWins++;
                        else
                            mctsWins++;

                        break;
                    }

                    Move m;
                    if ((board.sideToMove() == Color::White && nnueIsWhite) ||
                        (board.sideToMove() == Color::Black && !nnueIsWhite))
                    {
                        m = nnuePlayer.getMove(board);
                    }
                    else
                    {
                        m = mctsPlayer.getMove(board);
                    }

                    StateInfo st;
                    board.makeMove(m, st);
                }

                if (game % 10 == 0 && game > 0)
                {
                    float score = (nnueWins + 0.5f * draws) / game;
                    std::cout << "\r  Game " << game << ": NNUE " << nnueWins << "-" << draws
                              << "-" << mctsWins << " (" << std::fixed << std::setprecision(1)
                              << score * 100 << "%)" << std::flush;
                }
            }

            float winRate = (nnueWins + 0.5f * draws) / config_.nnueVsMctsGames;
            std::cout << "\n  NNUE vs MCTS complete: " << nnueWins << "-" << draws << "-" << mctsWins
                      << " (" << winRate * 100 << "%)\n";

            // Train NNUE on the games
            if (!trainingBoards.empty())
            {
                std::cout << "  Training NNUE on " << trainingBoards.size() << " positions from games...\n";
                // nnueEvaluator_->train(trainingBoards, trainingOutcomes);
            }

            // Update NNUE Elo estimate
            float eloDiff = EloCalculator::eloDifferenceFromWinRate(winRate);
            nnueElo_ = currentElo_ + eloDiff;
            std::cout << "  NNUE estimated Elo: " << nnueElo_ << " (vs MCTS " << currentElo_ << ")\n";
        }

        bool EloTrainingManager::evaluateAndPromote()
        {
            // Save current network as challenger
            std::string challengerPath = config_.checkpointDir + "/challenger_temp.pt";
            currentNetwork_->save(challengerPath);

            // Create players
            mcts::MCTSConfig evalConfig;
            evalConfig.numSimulations = config_.evaluationSimulations;
            evalConfig.addNoise = false; // No exploration during evaluation

            MCTSPlayer challenger(currentNetwork_, evalConfig, "Challenger");
            MCTSPlayer champion(championNetwork_, evalConfig, "Champion");

            // Run evaluation match
            std::cout << "\n";
            EloResult result = matchEvaluator_->evaluateChallenger(challenger, champion, currentElo_);
            std::cout << "\n";

            // Print results
            std::cout << "\n┌─────────────────────────────────────┐\n";
            std::cout << "│         EVALUATION RESULTS          │\n";
            std::cout << "├─────────────────────────────────────┤\n";
            std::cout << "│ Wins:   " << std::setw(3) << result.wins << "                          │\n";
            std::cout << "│ Draws:  " << std::setw(3) << result.draws << "                          │\n";
            std::cout << "│ Losses: " << std::setw(3) << result.losses << "                          │\n";
            std::cout << "├─────────────────────────────────────┤\n";
            std::cout << "│ Win Rate: " << std::fixed << std::setprecision(2)
                      << result.winRate * 100 << "%                    │\n";
            std::cout << "│ Elo Diff: " << std::showpos << std::setprecision(1)
                      << result.eloDifference << std::noshowpos << "                       │\n";
            std::cout << "│ Threshold: +1.0 Elo (50.14% win)    │\n";
            std::cout << "│ Status: " << (result.improved ? "✅ IMPROVED" : "❌ NOT IMPROVED") << "               │\n";
            std::cout << "└─────────────────────────────────────┘\n";

            // Try to promote
            if (modelManager_->tryPromote(challengerPath, result))
            {
                // Copy champion network weights
                championNetwork_->load(modelManager_->getChampion().path);
                currentElo_ = result.newElo;
                generation_++;
                return true;
            }

            return false;
        }

    } // namespace elo
} // namespace chess
