#include "../third_party/httplib.h"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

int main() {
    httplib::Server server;
    std::atomic<bool> disconnected{false};
    server.Post("/generate", [&](const httplib::Request &req, httplib::Response &res) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < deadline) {
            if (req.is_connection_closed && req.is_connection_closed()) {
                disconnected = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        res.set_content("finished", "text/plain");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port < 0) return 1;
    std::thread worker([&] { server.listen_after_bind(); });
    while (!server.is_running()) std::this_thread::yield();
    {
        httplib::Client client("127.0.0.1", port);
        client.set_read_timeout(0, 100000);
        (void)client.Post("/generate", "request", "text/plain");
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!disconnected && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    server.stop();
    worker.join();
    if (!disconnected) {
        std::cerr << "nonstream client timeout was not detected\n";
        return 1;
    }
    std::cout << "nonstream client timeout detected without explicit cancellation\n";
}
