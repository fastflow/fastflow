// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <ff/ff.hpp>
#include <ff/distributed/dsender/selection.hpp>
#include <ff/distributed/protocol/message.hpp>

namespace ff::distributed::sender {

using Endpoints = std::vector<std::tuple<std::string, std::string, std::vector<ff_node*>>>;
using Channels = std::unordered_map<ff_node*, IngressEgressChannels_t>;

/** FastFlow collector coordinating the sender components.
 * Takes ownership of each incoming envelope. Routing and Selection choose a
 * session; Batch applies the accumulation policy; Session performs the send.
 * Shutdown flushes batches and completes all sends before closing writes.
 * Endpoints and channel topology are borrowed and must outlive this node.
 */
class Sender : public ff_minode_t<message2_t> {
    const Endpoints& endpoints_;
    const Channels& channels_;
    SenderOptions options_;
    Routing routing_;
    Selection selection_;
    std::vector<std::unique_ptr<Session>> sessions_;

    void progress() {
        for (const auto& session : sessions_) {
            session->progressSends();
            session->pollConfirmation();
        }
    }

    void awaitToken(Session& session) {
        while (!session.hasToken()) {
            progress();
            std::this_thread::yield();
        }
    }

    void flush(Session& session) {
        if (session.batch().empty()) return;
        awaitToken(session);
        session.sendBatch();
    }

    void send(Session& session, const message2_t& message) {
        if (options_.batchRecords == 1 || message.size >= options_.batchBytes ||
            message.size > options_.protocol.separatePayloadAbove) {
            flush(session);
            awaitToken(session);
            session.sendSingle(message, options_.protocol.separatePayloadAbove);
            return;
        }
        session.batch().append(message);
        if (session.batch().recordCount() >= options_.batchRecords ||
            session.batch().payloadBytes() >= options_.batchBytes ||
            messageKind(message) == MessageKind::ProducerEnd || message.type == ChannelType::FBK)
            flush(session);
    }

    Session& select(int source) {
        const auto& allowed = routing_.allowed(source);
        for (;;) {
            if (const auto id = selection_.choose(allowed, sessions_)) return *sessions_.at(*id);
            progress();
            std::this_thread::yield();
        }
    }

protected:
    // Connection attachment is also used by tests with real local MTCL handles.
    SessionId attachSession(MTCL::HandleUser handle) {
        sessions_.push_back(std::make_unique<Session>(std::move(handle), options_));
        return sessions_.size() - 1;
    }
    void addDestination(int destination, SessionId session) { routing_.addDestination(destination, session); }
    void allowSource(int source, SessionId session) { routing_.allowSource(source, session); }
    bool progressSends() {
        bool finished = true;
        for (const auto& session : sessions_) {
            const bool completed = session->progressSends();
            finished = completed && finished;
        }
        return finished;
    }

public:
    Sender(const Endpoints& endpoints, const Channels& channels, SenderOptions options = {})
        : endpoints_(endpoints), channels_(channels), options_(options), selection_(options.loadBalancing) {
        options_.validate();
    }

    int svc_init() override {
        try {
            for (const auto& endpoint : endpoints_) {
                auto handle = MTCL::Manager::connect(std::get<1>(endpoint), options_.connectRetries, 10);
                if (!handle.isValid()) transportError("distributed connect");
                const auto id = attachSession(std::move(handle));
                for (const auto* node : std::get<2>(endpoint)) routing_.addDestination(node->mioID, id);
            }
            for (const auto& entry : channels_)
                for (const auto& channel : entry.second.second)
                    if (std::get<2>(channel) == ChannelLocality::REMOTE)
                        routing_.allowSource(entry.first->mioID, routing_.destination(std::get<0>(channel)->mioID));
            return 0;
        } catch (const std::exception& exception) {
            error("distributed sender initialization: %s\n", exception.what());
            return -1;
        }
    }

    message2_t* svc(message2_t* task) override {
        try {
            Message message(task);
            const auto kind = messageKind(*message);
            if (kind == MessageKind::Flush) {
                for (const auto id : routing_.allowed(message->src)) flush(*sessions_.at(id));
            } else if (message->dest >= 0) {
                send(*sessions_.at(routing_.destination(message->dest)), *message);
            } else if (kind == MessageKind::ProducerEnd) {
                for (const auto id : routing_.allowed(message->src)) {
                    const auto end = makeProducerEnd(message->src);
                    send(*sessions_.at(id), *end);
                }
            } else {
                send(select(message->src), *message);
            }
            return this->GO_ON;
        } catch (const std::exception& exception) {
            error("distributed sender: %s\n", exception.what());
            std::abort();
        }
    }

    void svc_end() override {
        try {
            for (const auto& session : sessions_) flush(*session);
            for (const auto& session : sessions_) session->completeSends();
            for (const auto& session : sessions_) session->closeWrites();
            for (;;) {
                bool outstanding = false;
                for (const auto& session : sessions_) outstanding = outstanding || session->outstanding() != 0;
                if (!outstanding) break;
                progress();
                std::this_thread::yield();
            }
        } catch (const std::exception& exception) {
            error("distributed sender shutdown: %s\n", exception.what());
            std::abort();
        }
    }
};

} // namespace ff::distributed::sender
