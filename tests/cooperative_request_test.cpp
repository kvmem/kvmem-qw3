#include "cooperative_request.hpp"
#include <atomic>
#include <iostream>
#include <memory>
#include <vector>

using qw3::detail::CooperativeRequest;
static void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct Event { unsigned request, step, reply = 0; };

int main() try {
    std::atomic<unsigned> executing{0}, completed{0}, unwound{0};
    struct Guard { std::atomic<unsigned> &counter; ~Guard() { ++counter; } };
    std::vector<std::unique_ptr<CooperativeRequest<Event>>> requests;
    for (unsigned i = 0; i < 4; ++i) {
        requests.emplace_back(new CooperativeRequest<Event>([&, i](auto &request) {
            Guard guard{unwound};
            unsigned value = i;
            for (unsigned j = 0; j < 100; ++j) {
                check(executing.fetch_add(1) == 0, "controllers ran concurrently");
                Event event{i, j, 0};
                --executing;
                request.yield(event);
                check(executing.fetch_add(1) == 0, "controller resumed before predecessor yielded");
                check(event.reply == i + j + 1, "scheduler reply was not visible");
                value += event.reply;
                --executing;
            }
            check(value == i + 5050 + 100 * i, "request stack was corrupted");
            ++completed;
        }));
    }
    for (unsigned j = 0; j < 101; ++j) {
        for (unsigned r = 0; r < 4; ++r) {
            const unsigned i = j % 2 ? 3 - r : r;
            auto *event = requests[i]->resume();
            if (j == 100) {
                check(event == nullptr && !requests[i]->error(), "request did not complete");
            } else {
                check(event && event->request == i && event->step == j, "yield identity changed");
                event->reply = i + j + 1;
            }
        }
    }
    requests.clear();
    check(completed == 4 && unwound == 4 && executing == 0, "completed stacks did not unwind");
    {
        CooperativeRequest<Event> paused([&](auto &request) {
            Guard guard{unwound}; Event event{0, 0}; request.yield(event);
            ++completed;
        });
        check(paused.resume() != nullptr, "cancel test did not suspend");
    }
    check(completed == 4 && unwound == 5, "cancellation did not unwind the suspended stack");
    {
        CooperativeRequest<Event> never_started([&](auto &) { ++completed; });
    }
    check(completed == 4, "unadmitted request executed during teardown");
    // A private query can suspend with an active rollback scope while a peer
    // keeps decoding. Destroying the suspended owner must restore only its
    // own state, without ever resuming the private branch after cancellation.
    struct Transaction {
        unsigned &state;
        unsigned snapshot;
        explicit Transaction(unsigned &s) : state(s), snapshot(s) {}
        ~Transaction() { state = snapshot; }
    };
    unsigned query_state = 7, peer_state = 41;
    bool private_resumed = false;
    {
        CooperativeRequest<Event> query([&](auto &request) {
            Transaction rollback(query_state);
            query_state = 70;
            Event checkpoint{0, 0}; request.yield(checkpoint);
            private_resumed = true;
        });
        CooperativeRequest<Event> peer([&](auto &request) {
            for (unsigned step = 0; step < 5; ++step) {
                ++peer_state;
                Event next{1, step}; request.yield(next);
            }
        });
        check(query.resume() && query_state == 70, "private query did not suspend inside transaction");
        for (unsigned step = 0; step < 5; ++step) {
            auto *event = peer.resume();
            check(event && event->step == step && query_state == 70, "peer disturbed private query snapshot");
        }
        check(!peer.resume() && !peer.error(), "peer decode did not complete");
    }
    check(query_state == 7 && peer_state == 46 && !private_resumed,
          "private cancellation failed to roll back independently");
    CooperativeRequest<Event> failed([](auto &) { throw std::runtime_error("controller failure"); });
    check(!failed.resume() && bool(failed.error()), "controller error was lost");
    try { std::rethrow_exception(failed.error()); }
    catch (const std::runtime_error &e) { check(std::string(e.what()) == "controller failure", "wrong failure"); }
    std::cout << "cooperative_requests yields=400 completion=4 cancelled=1 unadmitted=1 exception=1 private_rollback=1 status=pass\n";
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
