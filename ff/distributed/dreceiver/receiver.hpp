// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <ff/ff.hpp>
#include <ff/distributed/dreceiver/session.hpp>

namespace ff::distributed::receiver {

/** FastFlow emitter coordinating independently progressing receive sessions.
 * Completed messages pass to ff_send_out(), normally connected to a local
 * routing adapter. Success means local forwarding, not application completion.
 * The group termination counter stops admission of new batches; already started
 * batches, including separate payloads, are drained before this node returns.
 */
class Receiver : public ff_monode_t<message2_t> {
    std::string endpoint_;
    std::size_t expectedSessions_;
    ProtocolOptions options_;

    void deliver(Message message) {
        // FastFlow takes raw ownership on success. A rejecting or throwing
        // callback must leave the message unconsumed so ownership can be restored.
        auto* outgoing = message.release();
        try {
            if (!ff_send_out(outgoing))
                throw std::runtime_error("distributed local routing rejected a message");
        } catch (...) {
            message.reset(outgoing);
            throw;
        }
    }

public:
    Receiver(std::string endpoint, std::size_t expectedSessions, ProtocolOptions options = {})
        : endpoint_(std::move(endpoint)), expectedSessions_(expectedSessions), options_(options) {}

    int svc_init() override {
        if (MTCL::Manager::listen(endpoint_) < 0) {
            error("distributed receiver listen failed\n");
            return -1;
        }
        return 0;
    }

    message2_t* svc(message2_t*) override {
        std::vector<std::unique_ptr<Session>> sessions;
        for (;;) {
            const bool accepting = ff::termination_counter.load() > 0;
            if (accepting && sessions.size() < expectedSessions_) {
                auto handle = MTCL::Manager::getNext(std::chrono::microseconds(0));
                if (handle.isValid() && handle.isNewConnection())
                    sessions.push_back(std::make_unique<Session>(std::move(handle), options_));
            }
            bool pending = false;
            for (const auto& session : sessions) {
                const auto admission = accepting ? BatchAdmission::AcceptNew : BatchAdmission::DrainPending;
                session->progress(admission, [this](Message message) { deliver(std::move(message)); });
                pending = session->hasPendingBatch() || pending;
            }
            if (ff::termination_counter.load() <= 0 && !pending) break;
            std::this_thread::yield();
        }
        return this->EOS;
    }
};

} // namespace ff::distributed::receiver
