#pragma once

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace qw3::detail {
// A stack-preserving request controller. The scheduler resumes ONE controller
// at a time and waits until it yields before running another controller or GPU
// batch. Thus existing transactional request code retains its locals/RAII while
// shared backend access remains serialized. Event storage belongs to the paused
// controller and is valid until the scheduler resumes it.
template<class Event> class CooperativeRequest {
public:
    using Body = std::function<void(CooperativeRequest &)>;
    explicit CooperativeRequest(Body body) : worker_([this, body = std::move(body)] {
        try {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return state_ == State::Running || stopping_; });
                if (stopping_) throw std::runtime_error("request controller stopped");
            }
            body(*this);
        } catch (...) { error_ = std::current_exception(); }
        std::lock_guard<std::mutex> lock(mutex_);
        event_ = nullptr;
        state_ = State::Done;
        cv_.notify_all();
    }) {}
    ~CooperativeRequest() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            cv_.notify_all();
        }
        if (worker_.joinable()) worker_.join();
    }
    CooperativeRequest(const CooperativeRequest &) = delete;
    CooperativeRequest &operator=(const CooperativeRequest &) = delete;

    Event *resume() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (state_ == State::Done) return nullptr;
        if (state_ == State::Running || stopping_)
            throw std::logic_error("request controller is already running or stopped");
        state_ = State::Running;
        cv_.notify_all();
        cv_.wait(lock, [&] { return state_ != State::Running; });
        return event_;
    }
    void yield(Event &event) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (state_ != State::Running)
            throw std::logic_error("only the running request may yield");
        event_ = &event;
        state_ = State::Yielded;
        cv_.notify_all();
        cv_.wait(lock, [&] { return state_ == State::Running || stopping_; });
        event_ = nullptr;
        if (stopping_) throw std::runtime_error("request controller stopped");
    }
    std::exception_ptr error() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::Done) throw std::logic_error("request has not finished");
        return error_;
    }
private:
    enum class State { Ready, Running, Yielded, Done };
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    State state_ = State::Ready;
    Event *event_ = nullptr;
    bool stopping_ = false;
    std::exception_ptr error_;
    std::thread worker_;
};
} // namespace qw3::detail
