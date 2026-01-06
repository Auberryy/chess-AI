#include "training/stockfish_interface.hpp"
#include <iostream>
#include <sstream>
#include <thread>
#include <regex>
#include <algorithm>

namespace chess {
namespace training {

StockfishInterface::~StockfishInterface() {
    stop();
}

#ifdef _WIN32
// Windows implementation

bool StockfishInterface::start(const std::string& stockfishPath) {
    if (running_) return true;
    
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = nullptr;
    
    HANDLE hStdinRead, hStdoutWrite;
    
    // Create pipes
    if (!CreatePipe(&hStdoutRead_, &hStdoutWrite, &saAttr, 0)) {
        std::cerr << "Failed to create stdout pipe\n";
        return false;
    }
    SetHandleInformation(hStdoutRead_, HANDLE_FLAG_INHERIT, 0);
    
    if (!CreatePipe(&hStdinRead, &hStdinWrite_, &saAttr, 0)) {
        std::cerr << "Failed to create stdin pipe\n";
        CloseHandle(hStdoutRead_);
        CloseHandle(hStdoutWrite);
        return false;
    }
    SetHandleInformation(hStdinWrite_, HANDLE_FLAG_INHERIT, 0);
    
    // Start process
    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    si.hStdInput = hStdinRead;
    si.hStdOutput = hStdoutWrite;
    si.hStdError = hStdoutWrite;
    si.dwFlags |= STARTF_USESTDHANDLES;
    
    PROCESS_INFORMATION pi = {};
    
    if (!CreateProcessA(
        stockfishPath.c_str(),
        nullptr,
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &si,
        &pi
    )) {
        std::cerr << "Failed to start Stockfish: " << GetLastError() << "\n";
        CloseHandle(hStdinRead);
        CloseHandle(hStdinWrite_);
        CloseHandle(hStdoutRead_);
        CloseHandle(hStdoutWrite);
        return false;
    }
    
    hProcess_ = pi.hProcess;
    CloseHandle(pi.hThread);
    CloseHandle(hStdinRead);
    CloseHandle(hStdoutWrite);
    
    running_ = true;
    
    // Initialize UCI
    sendCommand("uci");
    std::string response = readUntil("uciok");
    if (response.find("uciok") == std::string::npos) {
        std::cerr << "Stockfish UCI initialization failed\n";
        stop();
        return false;
    }
    
    // Set up for analysis
    sendCommand("setoption name Hash value 128");
    sendCommand("setoption name Threads value 1");
    sendCommand("isready");
    readUntil("readyok");
    
    std::cout << "Stockfish started successfully\n";
    return true;
}

void StockfishInterface::stop() {
    if (!running_) return;
    
    sendCommand("quit");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    if (hProcess_) {
        TerminateProcess(hProcess_, 0);
        CloseHandle(hProcess_);
        hProcess_ = nullptr;
    }
    if (hStdinWrite_) {
        CloseHandle(hStdinWrite_);
        hStdinWrite_ = nullptr;
    }
    if (hStdoutRead_) {
        CloseHandle(hStdoutRead_);
        hStdoutRead_ = nullptr;
    }
    
    running_ = false;
}

void StockfishInterface::writeLine(const std::string& line) {
    if (!running_) return;
    std::string cmd = line + "\n";
    DWORD written;
    WriteFile(hStdinWrite_, cmd.c_str(), (DWORD)cmd.size(), &written, nullptr);
    FlushFileBuffers(hStdinWrite_);
}

std::string StockfishInterface::readLine(int timeoutMs) {
    if (!running_) return "";
    
    std::string line;
    char c;
    DWORD bytesRead;
    auto start = std::chrono::steady_clock::now();
    
    while (true) {
        DWORD available = 0;
        if (!PeekNamedPipe(hStdoutRead_, nullptr, 0, nullptr, &available, nullptr)) {
            break;
        }
        
        if (available > 0) {
            if (ReadFile(hStdoutRead_, &c, 1, &bytesRead, nullptr) && bytesRead > 0) {
                if (c == '\n') return line;
                if (c != '\r') line += c;
            }
        } else {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeoutMs) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    return line;
}

#else
// Unix implementation

bool StockfishInterface::start(const std::string& stockfishPath) {
    if (running_) return true;
    
    int stdinPipe[2], stdoutPipe[2];
    
    if (pipe(stdinPipe) < 0 || pipe(stdoutPipe) < 0) {
        std::cerr << "Failed to create pipes\n";
        return false;
    }
    
    pid_ = fork();
    if (pid_ < 0) {
        std::cerr << "Fork failed\n";
        return false;
    }
    
    if (pid_ == 0) {
        // Child process
        close(stdinPipe[1]);
        close(stdoutPipe[0]);
        
        dup2(stdinPipe[0], STDIN_FILENO);
        dup2(stdoutPipe[1], STDOUT_FILENO);
        dup2(stdoutPipe[1], STDERR_FILENO);
        
        close(stdinPipe[0]);
        close(stdoutPipe[1]);
        
        execl(stockfishPath.c_str(), stockfishPath.c_str(), nullptr);
        exit(1);
    }
    
    // Parent process
    close(stdinPipe[0]);
    close(stdoutPipe[1]);
    
    stdinFd_ = stdinPipe[1];
    stdoutFd_ = stdoutPipe[0];
    running_ = true;
    
    // Initialize UCI
    sendCommand("uci");
    std::string response = readUntil("uciok");
    if (response.find("uciok") == std::string::npos) {
        std::cerr << "Stockfish UCI initialization failed\n";
        stop();
        return false;
    }
    
    sendCommand("setoption name Hash value 128");
    sendCommand("setoption name Threads value 1");
    sendCommand("isready");
    readUntil("readyok");
    
    std::cout << "Stockfish started successfully\n";
    return true;
}

void StockfishInterface::stop() {
    if (!running_) return;
    
    sendCommand("quit");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    if (pid_ > 0) {
        kill(pid_, SIGTERM);
        waitpid(pid_, nullptr, 0);
        pid_ = -1;
    }
    if (stdinFd_ >= 0) {
        close(stdinFd_);
        stdinFd_ = -1;
    }
    if (stdoutFd_ >= 0) {
        close(stdoutFd_);
        stdoutFd_ = -1;
    }
    
    running_ = false;
}

void StockfishInterface::writeLine(const std::string& line) {
    if (!running_) return;
    std::string cmd = line + "\n";
    write(stdinFd_, cmd.c_str(), cmd.size());
}

std::string StockfishInterface::readLine(int timeoutMs) {
    if (!running_) return "";
    
    std::string line;
    char c;
    auto start = std::chrono::steady_clock::now();
    
    fd_set fds;
    struct timeval tv;
    
    while (true) {
        FD_ZERO(&fds);
        FD_SET(stdoutFd_, &fds);
        
        tv.tv_sec = 0;
        tv.tv_usec = 10000;  // 10ms
        
        int ret = select(stdoutFd_ + 1, &fds, nullptr, nullptr, &tv);
        if (ret > 0 && FD_ISSET(stdoutFd_, &fds)) {
            if (read(stdoutFd_, &c, 1) == 1) {
                if (c == '\n') return line;
                if (c != '\r') line += c;
            }
        }
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > timeoutMs) break;
    }
    
    return line;
}

#endif

// Common methods

void StockfishInterface::sendCommand(const std::string& command) {
    writeLine(command);
}

std::string StockfishInterface::readUntil(const std::string& pattern, int timeoutMs) {
    std::string result;
    auto start = std::chrono::steady_clock::now();
    
    while (true) {
        std::string line = readLine(1000);
        if (!line.empty()) {
            result += line + "\n";
            if (line.find(pattern) != std::string::npos) {
                return result;
            }
        }
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > timeoutMs) break;
    }
    
    return result;
}

void StockfishInterface::setOption(const std::string& name, const std::string& value) {
    sendCommand("setoption name " + name + " value " + value);
}

void StockfishInterface::setOption(const std::string& name, int value) {
    setOption(name, std::to_string(value));
}

void StockfishInterface::setPosition(const std::string& fen, const std::vector<std::string>& moves) {
    currentFen_ = fen;
    std::string cmd = "position ";
    
    if (fen == "startpos" || fen.empty()) {
        cmd += "startpos";
    } else {
        cmd += "fen " + fen;
    }
    
    if (!moves.empty()) {
        cmd += " moves";
        for (const auto& m : moves) {
            cmd += " " + m;
        }
    }
    
    sendCommand(cmd);
}

std::optional<StockfishResult> StockfishInterface::searchDepth(int depth) {
    sendCommand("go depth " + std::to_string(depth));
    std::string output = readUntil("bestmove", 60000);
    
    if (output.find("bestmove") == std::string::npos) {
        return std::nullopt;
    }
    
    return parseSearchOutput(output);
}

std::optional<StockfishResult> StockfishInterface::searchTime(int milliseconds) {
    sendCommand("go movetime " + std::to_string(milliseconds));
    std::string output = readUntil("bestmove", milliseconds + 5000);
    
    if (output.find("bestmove") == std::string::npos) {
        return std::nullopt;
    }
    
    return parseSearchOutput(output);
}

std::optional<StockfishResult> StockfishInterface::searchNodes(int64_t nodes) {
    sendCommand("go nodes " + std::to_string(nodes));
    std::string output = readUntil("bestmove", 60000);
    
    if (output.find("bestmove") == std::string::npos) {
        return std::nullopt;
    }
    
    return parseSearchOutput(output);
}

std::optional<float> StockfishInterface::staticEval() {
    sendCommand("eval");
    std::string output = readUntil("Total evaluation", 5000);
    
    // Parse "Total evaluation: +0.50 (white side)"
    std::regex evalRegex(R"(Total evaluation:\s*([+-]?\d+\.?\d*))");
    std::smatch match;
    
    if (std::regex_search(output, match, evalRegex)) {
        return std::stof(match[1].str()) * 100;  // Convert to centipawns
    }
    
    return std::nullopt;
}

StockfishResult StockfishInterface::parseSearchOutput(const std::string& output) {
    StockfishResult result = {};
    
    std::istringstream iss(output);
    std::string line;
    std::string lastInfoLine;
    
    while (std::getline(iss, line)) {
        if (line.find("info depth") != std::string::npos && 
            line.find("score") != std::string::npos) {
            lastInfoLine = line;
        }
        
        if (line.find("bestmove") != std::string::npos) {
            // Parse bestmove
            std::istringstream bmss(line);
            std::string token;
            bmss >> token;  // "bestmove"
            bmss >> result.bestMove;
            
            if (bmss >> token && token == "ponder") {
                bmss >> result.ponder;
            }
        }
    }
    
    // Parse last info line
    if (!lastInfoLine.empty()) {
        std::istringstream infoSS(lastInfoLine);
        std::string token;
        
        while (infoSS >> token) {
            if (token == "depth") {
                infoSS >> result.depth;
            } else if (token == "score") {
                infoSS >> token;
                if (token == "cp") {
                    infoSS >> result.centipawns;
                    result.isMate = false;
                } else if (token == "mate") {
                    infoSS >> result.mateIn;
                    result.isMate = true;
                    // Convert mate to high centipawn value
                    result.centipawns = result.mateIn > 0 ? 10000.0f : -10000.0f;
                }
            } else if (token == "nodes") {
                infoSS >> result.nodes;
            } else if (token == "pv") {
                // Read rest of line as PV
                std::string move;
                while (infoSS >> move) {
                    result.pv.push_back(move);
                }
            }
        }
    }
    
    return result;
}

} // namespace training
} // namespace chess
