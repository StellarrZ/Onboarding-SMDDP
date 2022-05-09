#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>


namespace my {

    class mySemaphore {
    private:
        int _cnt;
        std::mutex _mtx;
        std::condition_variable _condiVar;

    public:
        explicit mySemaphore(int cntInit = 0): _cnt(cntInit) {}

        void p() {
            std::unique_lock<std::mutex> lock(_mtx);
            if (--_cnt < 0) {
                _condiVar.wait(lock);
            }
        }

        void v() {
            std::lock_guard<std::mutex> lock(_mtx);
            if (++_cnt <= 0) {
                _condiVar.notify_one();
            }
        }
    };

} // namespace my