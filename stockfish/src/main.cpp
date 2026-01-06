#include "core/board.hpp"
#include "nn/network.hpp"
#include "nn/nnue.hpp"
#include "mcts/mcts.hpp"
#include "training/trainer.hpp"
#include "elo/elo_manager.hpp"
#include "ui/ui.hpp"

#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

void signalHandler(int signum)
{
    std::cout << "\nInterrupt signal received. Shutting down gracefully...\n";
    g_running = false;
}

void printUsage(const char *programName)
{
    std::cout << "Chess AI Training System\n";
    std::cout << "========================\n\n";
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --train           Start training from scratch\n";
    std::cout << "  --elo-training    Elo-based training with challenger/champion system\n";
    std::cout << "  --resume <path>   Resume training from checkpoint\n";
    std::cout << "  --play            Play against the AI\n";
    std::cout << "  --evaluate <path> Evaluate model against Stockfish\n";
    std::cout << "  --gui             Enable graphical UI (default)\n";
    std::cout << "  --no-gui          Use console UI\n";
    std::cout << "  --stockfish <path> Path to Stockfish executable\n";
    std::cout << "  --gpu <id>        GPU device ID (default: 0)\n";
    std::cout << "  --gpu-memory <f>  GPU memory fraction 0.0-1.0 (default: 0.5)\n";
    std::cout << "  --use-amp         Enable AMP/FP16 mixed precision (default: enabled)\n";
    std::cout << "  --no-amp          Disable AMP, use FP32 only\n";
    std::cout << "  --simulations <n> MCTS simulations per move (default: 800)\n";
    std::cout << "  --games <n>       Self-play games per iteration (default: 100)\n";
    std::cout << "  --eval-games <n>  Evaluation games (default: 100)\n";
    std::cout << "  --elo-threshold   Elo gain needed to save model (default: 1.0)\n";
    std::cout << "  --train-nnue      Also train NNUE alongside MCTS network\n";
    std::cout << "  --help            Show this help message\n";
}

int runTraining(chess::training::TrainingConfig &config, bool useGui, const std::string &checkpointPath)
{
    std::cout << "Initializing Chess AI Training System...\n";
    std::cout << "GPU Memory Allocation: " << config.gpuMemoryFraction * 100 << "%\n\n";

    // Create trainer
    chess::training::Trainer trainer(config);

    // Load checkpoint if resuming
    if (!checkpointPath.empty())
    {
        std::cout << "Loading checkpoint: " << checkpointPath << "\n";
        trainer.loadCheckpoint(checkpointPath);
    }

    if (useGui)
    {
        // GUI mode
        chess::ui::UIConfig uiConfig;
        uiConfig.windowTitle = "Chess AI Training Monitor";

        chess::ui::TrainingUI ui(uiConfig);
        if (!ui.init())
        {
            std::cerr << "Failed to initialize GUI. Falling back to console mode.\n";
            useGui = false;
        }

        if (useGui)
        {
            std::atomic<bool> trainingActive{true};

            // Setup callbacks
            ui.setStopCallback([&]()
                               {
                trainer.stop();
                trainingActive = false; });

            ui.setPauseCallback([&](bool pause)
                                {
                                    // Pause/resume logic
                                });

            // Training progress callback
            trainer.setProgressCallback([&](const chess::training::TrainingStats &stats)
                                        {
                chess::ui::TrainingStatus status;
                status.sessionTime = stats.getSessionTime();
                status.averageSessionTime = stats.getAverageSessionTime();
                status.totalTime = stats.totalTrainingTime;
                status.numSessions = stats.numSessions;
                status.currentWinRate = stats.currentWinRate;
                status.bestWinRate = stats.bestWinRate;
                status.winRateHistory = stats.evaluationScores;
                status.iterationHistory = stats.evaluationIterations;
                status.currentIteration = stats.currentIteration;
                status.totalIterations = config.trainingIterations;
                status.gamesPlayed = stats.totalGamesPlayed;
                status.positionsTrained = stats.totalPositionsTrained;
                status.currentLoss = stats.currentLoss;
                status.isTraining = trainingActive;
                status.statusMessage = "Training in progress...";
                
                ui.updateStatus(status); });

            // Evaluation callback (triggered when win rate >= 50%)
            trainer.setEvaluationCallback([&](float winRate) -> bool
                                          {
                ui.showContinueDialog(winRate);
                
                // Wait for user response
                while (!ui.dialogResponded() && g_running) {
                    ui.update();
                    std::this_thread::sleep_for(std::chrono::milliseconds(16));
                }
                
                return ui.shouldContinue(); });

            // Start training in background thread
            std::thread trainingThread([&]()
                                       {
                trainer.train();
                trainingActive = false; });

            // UI main loop
            while (g_running && ui.update())
            {
                if (!trainingActive)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }

            // Cleanup
            trainer.stop();
            if (trainingThread.joinable())
            {
                trainingThread.join();
            }
        }
    }

    if (!useGui)
    {
        // Console mode
        chess::ui::ConsoleUI consoleUI;
        consoleUI.start();

        trainer.setProgressCallback([&](const chess::training::TrainingStats &stats)
                                    { consoleUI.updateStatus(stats); });

        trainer.setEvaluationCallback([&](float winRate) -> bool
                                      {
            consoleUI.showContinuePrompt(winRate);
            return consoleUI.shouldContinue(); });

        trainer.train();
        consoleUI.stop();
    }

    std::cout << "\nTraining complete!\n";
    std::cout << "Final statistics:\n";
    const auto &stats = trainer.getStats();
    std::cout << "  Total training time: " << stats.totalTrainingTime.count() / 3600 << " hours\n";
    std::cout << "  Games played: " << stats.totalGamesPlayed << "\n";
    std::cout << "  Best win rate: " << stats.bestWinRate * 100 << "%\n";

    return 0;
}

int playAgainstAI(const std::string &modelPath, int simulations)
{
    std::cout << "Loading AI model...\n";

    auto network = std::make_shared<chess::nn::NeuralNetwork>(modelPath, 0, 0.5f, true);

    chess::mcts::MCTSConfig mctsConfig;
    mctsConfig.numSimulations = simulations;
    mctsConfig.addNoise = false;

    chess::mcts::MCTSEngine engine(network, mctsConfig);

    chess::Board board;
    board.setStartingPosition();

    std::cout << "\nYou are playing as White. Enter moves in UCI format (e.g., e2e4)\n";
    std::cout << "Type 'quit' to exit.\n\n";

    while (g_running)
    {
        board.print();

        auto result = board.result();
        if (result != chess::GameResult::Ongoing)
        {
            if (result == chess::GameResult::WhiteWins)
            {
                std::cout << "You win!\n";
            }
            else if (result == chess::GameResult::BlackWins)
            {
                std::cout << "AI wins!\n";
            }
            else
            {
                std::cout << "Draw!\n";
            }
            break;
        }

        if (board.sideToMove() == chess::Color::White)
        {
            // Human move
            std::cout << "Your move: ";
            std::string input;
            std::cin >> input;

            if (input == "quit")
                break;

            chess::Move move = chess::Move::fromUCI(input);
            auto legalMoves = chess::MoveGen::generateLegal(board);

            bool isLegal = false;
            for (const auto &m : legalMoves)
            {
                if (m.from() == move.from() && m.to() == move.to())
                {
                    move = m; // Get the full move with proper flags
                    isLegal = true;
                    break;
                }
            }

            if (!isLegal)
            {
                std::cout << "Illegal move. Try again.\n";
                continue;
            }

            chess::StateInfo st;
            board.makeMove(move, st);
            engine.advanceTree(move);
        }
        else
        {
            // AI move
            std::cout << "AI is thinking...\n";
            chess::Move move = engine.search(board);

            std::cout << "AI plays: " << move.toUCI() << "\n";

            chess::StateInfo st;
            board.makeMove(move, st);
            engine.advanceTree(move);
        }
    }

    return 0;
}

int evaluateModel(const std::string &modelPath, const std::string &stockfishPath, int numGames)
{
    std::cout << "Evaluating model against Stockfish...\n";
    std::cout << "Model: " << modelPath << "\n";
    std::cout << "Stockfish: " << stockfishPath << "\n";
    std::cout << "Games: " << numGames << "\n\n";

    auto network = std::make_shared<chess::nn::NeuralNetwork>(modelPath, 0, 0.5f, true);

    chess::training::TrainingConfig config;
    config.stockfishPath = stockfishPath;
    config.evaluationGames = numGames;

    chess::training::Trainer trainer(config);
    float winRate = trainer.evaluateAgainstStockfish(numGames);

    std::cout << "\nFinal Results:\n";
    std::cout << "Win Rate: " << winRate * 100 << "%\n";

    return 0;
}

int runEloTraining(int gpuId, float gpuMemory, bool useAmp, int simulations, int selfPlayGames,
                   int evalGames, float eloThreshold, bool trainNNUE, bool useGui)
{
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║             AURORA CHESS AI - TRAINING MODE                  ║\n";
    std::cout << "║              Codename: Flapjack Puffin 🐧                    ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
    std::cout << "║ GPU: " << gpuId << " | Memory: " << static_cast<int>(gpuMemory * 100) << "% | NNUE: "
              << (trainNNUE ? "Yes" : "No ") << " | AMP: " << (useAmp ? "Yes" : "No ") << "        ║\n";
    std::cout << "║ Elo threshold: +" << eloThreshold << " (>"
              << static_cast<int>(chess::elo::EloCalculator::winRateForEloGain(eloThreshold) * 100)
              << "% win rate needed)              ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    chess::elo::EloTrainingManager::Config config;
    config.gpuId = gpuId;
    config.gpuMemoryFraction = gpuMemory;
    config.useAmp = useAmp;
    config.mctsSimulations = simulations;
    config.selfPlayGamesPerIteration = selfPlayGames;
    config.evaluationGames = evalGames;
    config.evaluationSimulations = simulations / 2; // Faster evaluation
    config.eloThreshold = eloThreshold;
    config.trainNNUE = trainNNUE;
    config.modelsDir = "models";
    config.checkpointDir = "checkpoints";

    chess::elo::EloTrainingManager manager(config);

    if (useGui)
    {
        chess::ui::UIConfig uiConfig;
        uiConfig.windowTitle = "Chess AI Elo Training";

        chess::ui::TrainingUI ui(uiConfig);
        if (!ui.init())
        {
            std::cerr << "Failed to initialize GUI. Using console mode.\n";
            useGui = false;
        }

        if (useGui)
        {
            std::atomic<bool> trainingActive{true};

            manager.setProgressCallback([&](int iteration, float elo, int generation)
                                        {
                chess::ui::TrainingStatus status;
                status.currentIteration = iteration;
                status.currentWinRate = chess::elo::EloCalculator::winRateFromEloDifference(0);
                status.statusMessage = "Generation " + std::to_string(generation) + 
                                      " | Elo: " + std::to_string(static_cast<int>(elo));
                status.isTraining = trainingActive;
                ui.updateStatus(status); });

            manager.setPromotionCallback([&](float oldElo, float newElo, int generation)
                                         { std::cout << "\n🎉 PROMOTED! Gen " << generation << ": Elo "
                                                     << oldElo << " → " << newElo << "\n"; });

            std::thread trainingThread([&]()
                                       {
                manager.train(-1);  // Train indefinitely
                trainingActive = false; });

            while (g_running && ui.update())
            {
                if (!trainingActive)
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(16));
            }

            manager.stop();
            if (trainingThread.joinable())
            {
                trainingThread.join();
            }
        }
    }

    if (!useGui)
    {
        manager.setProgressCallback([](int iteration, float elo, int generation)
                                    { std::cout << "\n📊 Progress: Iteration " << iteration
                                                << " | Generation " << generation
                                                << " | Elo: " << elo << "\n"; });

        manager.setPromotionCallback([](float oldElo, float newElo, int generation)
                                     { std::cout << "\n🎉 NEW CHAMPION! Generation " << generation
                                                 << " | Elo: " << oldElo << " → " << newElo
                                                 << " (+" << (newElo - oldElo) << ")\n"; });

        manager.train(-1); // Train indefinitely until interrupted
    }

    return 0;
}

int main(int argc, char *argv[])
{
    // Setup signal handler
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // Parse arguments
    std::string mode = "train";
    std::string checkpointPath;
    std::string stockfishPath = "stockfish";
    bool useGui = true;
    bool trainNNUE = false;
    bool useAmp = true; // AMP enabled by default for RTX 3090
    int gpuId = 0;
    float gpuMemory = 0.5f;
    int simulations = 800;
    int games = 100;
    int evalGames = 100;
    float eloThreshold = 1.0f;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--train")
        {
            mode = "train";
        }
        else if (arg == "--elo-training")
        {
            mode = "elo-training";
        }
        else if (arg == "--resume" && i + 1 < argc)
        {
            mode = "train";
            checkpointPath = argv[++i];
        }
        else if (arg == "--play")
        {
            mode = "play";
        }
        else if (arg == "--evaluate" && i + 1 < argc)
        {
            mode = "evaluate";
            checkpointPath = argv[++i];
        }
        else if (arg == "--gui")
        {
            useGui = true;
        }
        else if (arg == "--no-gui")
        {
            useGui = false;
        }
        else if (arg == "--use-amp")
        {
            useAmp = true;
        }
        else if (arg == "--no-amp")
        {
            useAmp = false;
        }
        else if (arg == "--stockfish" && i + 1 < argc)
        {
            stockfishPath = argv[++i];
        }
        else if (arg == "--gpu" && i + 1 < argc)
        {
            gpuId = std::stoi(argv[++i]);
        }
        else if (arg == "--gpu-memory" && i + 1 < argc)
        {
            gpuMemory = std::stof(argv[++i]);
        }
        else if (arg == "--simulations" && i + 1 < argc)
        {
            simulations = std::stoi(argv[++i]);
        }
        else if (arg == "--games" && i + 1 < argc)
        {
            games = std::stoi(argv[++i]);
        }
        else if (arg == "--eval-games" && i + 1 < argc)
        {
            evalGames = std::stoi(argv[++i]);
        }
        else if (arg == "--elo-threshold" && i + 1 < argc)
        {
            eloThreshold = std::stof(argv[++i]);
        }
        else if (arg == "--train-nnue")
        {
            trainNNUE = true;
        }
    }

    // Configure training
    chess::training::TrainingConfig config;
    config.gpuId = gpuId;
    config.gpuMemoryFraction = gpuMemory;
    config.useAmp = useAmp;
    config.stockfishPath = stockfishPath;
    config.mctsSimulations = simulations;
    config.numSelfPlayGames = games;

    // Run appropriate mode
    if (mode == "train")
    {
        return runTraining(config, useGui, checkpointPath);
    }
    else if (mode == "elo-training")
    {
        return runEloTraining(gpuId, gpuMemory, useAmp, simulations, games, evalGames,
                              eloThreshold, trainNNUE, useGui);
    }
    else if (mode == "play")
    {
        return playAgainstAI(checkpointPath, simulations);
    }
    else if (mode == "evaluate")
    {
        return evaluateModel(checkpointPath, stockfishPath, evalGames);
    }

    return 0;
}
