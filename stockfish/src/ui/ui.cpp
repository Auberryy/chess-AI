#include "ui/ui.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

// Dear ImGui includes
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// GLFW/OpenGL
#include <GLFW/glfw3.h>

namespace chess
{
    namespace ui
    {

        TrainingUI::TrainingUI(UIConfig config) : config_(std::move(config)) {}

        TrainingUI::~TrainingUI()
        {
            if (window_)
            {
                ImGui_ImplOpenGL3_Shutdown();
                ImGui_ImplGlfw_Shutdown();
                ImGui::DestroyContext();
                glfwDestroyWindow(static_cast<GLFWwindow *>(window_));
                glfwTerminate();
            }
        }

        bool TrainingUI::init()
        {
            // Initialize GLFW
            if (!glfwInit())
            {
                std::cerr << "Failed to initialize GLFW" << std::endl;
                return false;
            }

            // GL 3.3 + GLSL 330
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

            // Create window
            GLFWwindow *glfwWindow = glfwCreateWindow(
                config_.windowWidth, config_.windowHeight,
                config_.windowTitle.c_str(), nullptr, nullptr);

            if (!glfwWindow)
            {
                std::cerr << "Failed to create GLFW window" << std::endl;
                glfwTerminate();
                return false;
            }

            window_ = glfwWindow;
            glfwMakeContextCurrent(glfwWindow);
            glfwSwapInterval(config_.vsync ? 1 : 0);

            // Initialize ImGui
            IMGUI_CHECKVERSION();
            ImGui::CreateContext();
            ImGuiIO &io = ImGui::GetIO();
            io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
            io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

            // Setup style
            ImGui::StyleColorsDark();
            ImGuiStyle &style = ImGui::GetStyle();
            style.WindowRounding = 8.0f;
            style.FrameRounding = 4.0f;
            style.GrabRounding = 4.0f;
            style.Colors[ImGuiCol_WindowBg] = ImVec4(0.1f, 0.1f, 0.12f, 0.95f);
            style.Colors[ImGuiCol_Header] = ImVec4(0.2f, 0.4f, 0.6f, 0.8f);
            style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.3f, 0.5f, 0.7f, 0.9f);
            style.Colors[ImGuiCol_Button] = ImVec4(0.2f, 0.4f, 0.6f, 0.8f);
            style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.3f, 0.5f, 0.7f, 0.9f);

            // Platform/Renderer bindings
            ImGui_ImplGlfw_InitForOpenGL(glfwWindow, true);
            ImGui_ImplOpenGL3_Init("#version 330");

            return true;
        }

        bool TrainingUI::update()
        {
            GLFWwindow *glfwWindow = static_cast<GLFWwindow *>(window_);

            if (glfwWindowShouldClose(glfwWindow))
            {
                return false;
            }

            glfwPollEvents();

            // Start ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Render UI
            renderMainWindow();

            if (showDialog_)
            {
                renderDialog();
            }

            // Rendering
            ImGui::Render();
            int displayW, displayH;
            glfwGetFramebufferSize(glfwWindow, &displayW, &displayH);
            glViewport(0, 0, displayW, displayH);
            glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            glfwSwapBuffers(glfwWindow);

            return true;
        }

        void TrainingUI::updateStatus(const TrainingStatus &status)
        {
            std::lock_guard<std::mutex> lock(statusMutex_);
            status_ = status;
        }

        void TrainingUI::showContinueDialog(float winRate)
        {
            dialogWinRate_ = winRate;
            dialogResponded_ = false;
            showDialog_ = true;
        }

        void TrainingUI::renderMainWindow()
        {
            std::lock_guard<std::mutex> lock(statusMutex_);

            ImGuiViewport *viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);

            ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                           ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                                           ImGuiWindowFlags_NoBringToFrontOnFocus;

            ImGui::Begin("Chess AI Training Monitor", nullptr, windowFlags);

            // Header
            ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]);
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Chess AI Training Monitor");
            ImGui::PopFont();
            ImGui::Separator();

            // Status indicator
            if (status_.isTraining)
            {
                ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f), "● Training");
            }
            else if (status_.isPaused)
            {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.2f, 1.0f), "● Paused");
            }
            else
            {
                ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "● Idle");
            }
            ImGui::SameLine();
            ImGui::Text("%s", status_.statusMessage.c_str());

            ImGui::Spacing();

            // Main content area with columns
            ImGui::Columns(2, "MainColumns", false);
            ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() * 0.4f);

            // Left column - Stats
            renderTimePanel();
            ImGui::Spacing();
            renderPerformancePanel();
            ImGui::Spacing();
            renderProgressPanel();
            ImGui::Spacing();
            renderResourcePanel();
            ImGui::Spacing();
            renderControlPanel();

            // Right column - Graph
            ImGui::NextColumn();
            renderGraph();

            ImGui::Columns(1);

            ImGui::End();
        }

        void TrainingUI::renderTimePanel()
        {
            ImGui::BeginChild("TimePanel", ImVec2(0, 130), true);

            ImGui::TextColored(ImVec4(0.8f, 0.8f, 1.0f, 1.0f), "⏱ Training Time");
            ImGui::Separator();

            ImGui::Text("Session Time:     %s", formatDuration(status_.sessionTime).c_str());
            ImGui::Text("Average Session:  %s", formatDuration(status_.averageSessionTime).c_str());
            ImGui::Text("Total Time:       %s", formatDuration(status_.totalTime).c_str());
            ImGui::Text("Sessions:         %d", status_.numSessions);

            ImGui::EndChild();
        }

        void TrainingUI::renderPerformancePanel()
        {
            ImGui::BeginChild("PerformancePanel", ImVec2(0, 130), true);

            ImGui::TextColored(ImVec4(0.8f, 1.0f, 0.8f, 1.0f), "📊 Performance vs Stockfish");
            ImGui::Separator();

            // Win rate with color coding
            ImVec4 winRateColor;
            if (status_.currentWinRate >= 0.5f)
            {
                winRateColor = ImVec4(0.2f, 1.0f, 0.2f, 1.0f); // Green
            }
            else if (status_.currentWinRate >= 0.3f)
            {
                winRateColor = ImVec4(1.0f, 1.0f, 0.2f, 1.0f); // Yellow
            }
            else
            {
                winRateColor = ImVec4(1.0f, 0.4f, 0.4f, 1.0f); // Red
            }

            ImGui::Text("Current Win Rate:");
            ImGui::SameLine();
            ImGui::TextColored(winRateColor, "%.1f%%", status_.currentWinRate * 100);

            ImGui::Text("Best Win Rate:   ");
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%.1f%%", status_.bestWinRate * 100);

            // Progress bar to 50% target
            float targetProgress = std::min(status_.currentWinRate / 0.5f, 1.0f);
            ImGui::Text("Progress to 50%% Target:");
            ImGui::ProgressBar(targetProgress, ImVec2(-1, 0),
                               (status_.currentWinRate >= 0.5f) ? "TARGET REACHED!" : "");

            ImGui::EndChild();
        }

        void TrainingUI::renderProgressPanel()
        {
            ImGui::BeginChild("ProgressPanel", ImVec2(0, 130), true);

            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.8f, 1.0f), "🎯 Training Progress");
            ImGui::Separator();

            float iterProgress = status_.totalIterations > 0 ? static_cast<float>(status_.currentIteration) / status_.totalIterations : 0;

            ImGui::Text("Iteration:        %d / %d", status_.currentIteration, status_.totalIterations);
            ImGui::ProgressBar(iterProgress, ImVec2(-1, 0));

            ImGui::Text("Games Played:     %d", status_.gamesPlayed);
            ImGui::Text("Positions Trained: %d", status_.positionsTrained);
            ImGui::Text("Current Loss:     %.4f", status_.currentLoss);

            ImGui::EndChild();
        }

        void TrainingUI::renderResourcePanel()
        {
            ImGui::BeginChild("ResourcePanel", ImVec2(0, 100), true);

            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.8f, 1.0f), "💻 GPU Resources (RTX 3090)");
            ImGui::Separator();

            float memPercent = status_.gpuMemoryTotalMB > 0 ? status_.gpuMemoryUsedMB / status_.gpuMemoryTotalMB : 0;

            ImGui::Text("Memory: %.0f / %.0f MB (%.1f%%)",
                        status_.gpuMemoryUsedMB, status_.gpuMemoryTotalMB, memPercent * 100);
            ImGui::ProgressBar(memPercent, ImVec2(-1, 0));

            ImGui::Text("GPU Utilization: %.1f%%", status_.gpuUtilization);
            ImGui::ProgressBar(status_.gpuUtilization / 100.0f, ImVec2(-1, 0));

            ImGui::EndChild();
        }

        void TrainingUI::renderControlPanel()
        {
            ImGui::BeginChild("ControlPanel", ImVec2(0, 80), true);

            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "🎮 Controls");
            ImGui::Separator();

            if (status_.isTraining)
            {
                if (ImGui::Button("⏸ Pause", ImVec2(100, 30)))
                {
                    if (pauseCallback_)
                        pauseCallback_(true);
                }
            }
            else
            {
                if (ImGui::Button("▶ Resume", ImVec2(100, 30)))
                {
                    if (pauseCallback_)
                        pauseCallback_(false);
                }
            }

            ImGui::SameLine();

            if (ImGui::Button("⏹ Stop", ImVec2(100, 30)))
            {
                if (stopCallback_)
                    stopCallback_();
            }

            ImGui::EndChild();
        }

        void TrainingUI::renderGraph()
        {
            ImGui::BeginChild("GraphPanel", ImVec2(0, -1), true);

            ImGui::TextColored(ImVec4(0.8f, 1.0f, 1.0f, 1.0f), "📈 Win Rate History");
            ImGui::Separator();

            // Draw graph area
            ImVec2 canvasPos = ImGui::GetCursorScreenPos();
            ImVec2 canvasSize = ImGui::GetContentRegionAvail();
            canvasSize.y = std::max(canvasSize.y, 200.0f);

            ImDrawList *drawList = ImGui::GetWindowDrawList();

            // Background
            drawList->AddRectFilled(canvasPos,
                                    ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
                                    IM_COL32(20, 20, 25, 255));

            // Grid lines
            for (int i = 0; i <= 10; i++)
            {
                float y = canvasPos.y + canvasSize.y * (1.0f - i / 10.0f);
                drawList->AddLine(ImVec2(canvasPos.x, y),
                                  ImVec2(canvasPos.x + canvasSize.x, y),
                                  IM_COL32(60, 60, 70, 100));

                // Label
                char label[16];
                snprintf(label, sizeof(label), "%d%%", i * 10);
                drawList->AddText(ImVec2(canvasPos.x + 5, y - 8), IM_COL32(150, 150, 150, 255), label);
            }

            // 50% target line
            float targetY = canvasPos.y + canvasSize.y * 0.5f;
            drawList->AddLine(ImVec2(canvasPos.x, targetY),
                              ImVec2(canvasPos.x + canvasSize.x, targetY),
                              IM_COL32(100, 200, 100, 200), 2.0f);
            drawList->AddText(ImVec2(canvasPos.x + canvasSize.x - 80, targetY - 15),
                              IM_COL32(100, 200, 100, 255), "50% Target");

            // Plot win rate history
            if (status_.winRateHistory.size() >= 2)
            {
                std::vector<ImVec2> points;

                for (size_t i = 0; i < status_.winRateHistory.size(); i++)
                {
                    float x = canvasPos.x + 50 + (canvasSize.x - 100) * i / (status_.winRateHistory.size() - 1);
                    float y = canvasPos.y + canvasSize.y * (1.0f - status_.winRateHistory[i]);
                    points.push_back(ImVec2(x, y));
                }

                // Draw line
                for (size_t i = 1; i < points.size(); i++)
                {
                    drawList->AddLine(points[i - 1], points[i], IM_COL32(100, 150, 255, 255), 2.0f);
                }

                // Draw points
                for (const auto &p : points)
                {
                    drawList->AddCircleFilled(p, 4.0f, IM_COL32(100, 150, 255, 255));
                }
            }

            ImGui::Dummy(canvasSize);
            ImGui::EndChild();
        }

        void TrainingUI::renderDialog()
        {
            ImGui::OpenPopup("Continue Training?");

            ImVec2 center = ImGui::GetMainViewport()->GetCenter();
            ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            ImGui::SetNextWindowSize(ImVec2(400, 200));

            if (ImGui::BeginPopupModal("Continue Training?", nullptr,
                                       ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove))
            {
                ImGui::TextWrapped("🎉 Congratulations!");
                ImGui::Spacing();
                ImGui::TextWrapped("The AI has reached %.1f%% win rate against Stockfish!",
                                   dialogWinRate_ * 100);
                ImGui::Spacing();
                ImGui::TextWrapped("Do you want to continue training to improve further?");
                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                if (ImGui::Button("Continue Training", ImVec2(150, 40)))
                {
                    shouldContinue_ = true;
                    dialogResponded_ = true;
                    showDialog_ = false;
                    if (continueCallback_)
                        continueCallback_();
                    ImGui::CloseCurrentPopup();
                }

                ImGui::SameLine();

                if (ImGui::Button("Stop Training", ImVec2(150, 40)))
                {
                    shouldContinue_ = false;
                    dialogResponded_ = true;
                    showDialog_ = false;
                    if (stopCallback_)
                        stopCallback_();
                    ImGui::CloseCurrentPopup();
                }

                ImGui::EndPopup();
            }
        }

        std::string TrainingUI::formatDuration(std::chrono::seconds duration) const
        {
            auto hours = std::chrono::duration_cast<std::chrono::hours>(duration);
            auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration % std::chrono::hours(1));
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration % std::chrono::minutes(1));

            std::ostringstream oss;
            oss << std::setfill('0') << std::setw(2) << hours.count() << ":"
                << std::setfill('0') << std::setw(2) << minutes.count() << ":"
                << std::setfill('0') << std::setw(2) << seconds.count();
            return oss.str();
        }

        // ConsoleUI implementation
        ConsoleUI::ConsoleUI() : rng_(std::random_device{}()) {}

        ConsoleUI::~ConsoleUI()
        {
            stop();
        }

        void ConsoleUI::start()
        {
            running_ = true;
            inputThread_ = std::thread(&ConsoleUI::inputLoop, this);
        }

        void ConsoleUI::stop()
        {
            running_ = false;
            if (inputThread_.joinable())
            {
                inputThread_.join();
            }
        }

        void ConsoleUI::updateStatus(const training::TrainingStats &stats)
        {
            std::lock_guard<std::mutex> lock(statsMutex_);
            lastStats_ = stats;
            printStatus();
        }

        void ConsoleUI::printStatus()
        {
// Clear screen (cross-platform)
#ifdef _WIN32
            system("cls");
#else
            system("clear");
#endif

            std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║            CHESS AI TRAINING MONITOR                         ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

            auto sessionTime = lastStats_.getSessionTime();
            auto avgTime = lastStats_.getAverageSessionTime();

            std::cout << "║ SESSION TIME:      " << std::setw(10)
                      << sessionTime.count() / 3600 << "h "
                      << (sessionTime.count() % 3600) / 60 << "m "
                      << sessionTime.count() % 60 << "s" << std::setw(28) << "║\n";

            std::cout << "║ AVERAGE SESSION:   " << std::setw(10)
                      << avgTime.count() / 3600 << "h "
                      << (avgTime.count() % 3600) / 60 << "m" << std::setw(32) << "║\n";

            std::cout << "║ TOTAL TIME:        " << std::setw(10)
                      << lastStats_.totalTrainingTime.count() / 3600 << "h "
                      << (lastStats_.totalTrainingTime.count() % 3600) / 60 << "m" << std::setw(32) << "║\n";

            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ ITERATION:         " << std::setw(10) << lastStats_.currentIteration << std::setw(32) << "║\n";
            std::cout << "║ GAMES PLAYED:      " << std::setw(10) << lastStats_.totalGamesPlayed << std::setw(32) << "║\n";
            std::cout << "║ POSITIONS TRAINED: " << std::setw(10) << lastStats_.totalPositionsTrained << std::setw(32) << "║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";

            std::cout << "║ WIN RATE VS STOCKFISH: ";
            std::cout << std::fixed << std::setprecision(1) << std::setw(6)
                      << lastStats_.currentWinRate * 100 << "%" << std::setw(31) << "║\n";

            std::cout << "║ BEST WIN RATE:         ";
            std::cout << std::fixed << std::setprecision(1) << std::setw(6)
                      << lastStats_.bestWinRate * 100 << "%" << std::setw(31) << "║\n";

            // Progress bar
            int barWidth = 40;
            int progress = static_cast<int>(lastStats_.currentWinRate / 0.5f * barWidth);
            progress = std::min(progress, barWidth);

            std::cout << "║ Progress to 50%: [";
            for (int i = 0; i < barWidth; i++)
            {
                if (i < progress)
                    std::cout << "█";
                else
                    std::cout << "░";
            }
            std::cout << "] ║\n";

            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Commands: [P]ause  [R]esume  [S]top  [Q]uit                   ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
        }

        void ConsoleUI::showContinuePrompt(float winRate)
        {
            std::cout << "\n";
            std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                    🎉 MILESTONE REACHED! 🎉                   ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Win rate of " << std::fixed << std::setprecision(1)
                      << winRate * 100 << "% achieved against Stockfish!              ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Continue training? (Y/N):                                    ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

            char response;
            std::cin >> response;
            shouldContinue_ = (response == 'Y' || response == 'y');
        }

        void ConsoleUI::inputLoop()
        {
            while (running_)
            {
                if (std::cin.peek() != EOF)
                {
                    char cmd;
                    std::cin >> cmd;

                    switch (tolower(cmd))
                    {
                    case 'p':
                        std::cout << "Pausing training...\n";
                        break;
                    case 'r':
                        std::cout << "Resuming training...\n";
                        break;
                    case 's':
                        std::cout << "Stopping training...\n";
                        running_ = false;
                        break;
                    case 'q':
                        running_ = false;
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

    } // namespace ui
} // namespace chess
