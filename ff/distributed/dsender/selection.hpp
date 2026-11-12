// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <vector>
#include <ff/distributed/dsender/routing.hpp>
#include <ff/distributed/dsender/session.hpp>

namespace ff::distributed::sender {

/** Chooses only among sessions allowed by Routing.
 * Existing batches are filled first. Otherwise select a session with a token,
 * using the configured policy. nullopt asks the sender to progress and retry;
 * a confirmation on an unrelated session never expands the allowed set.
 */
class Selection {
    LoadBalancing policy_;
    std::size_t next_ = 0;
    std::mt19937 random_{std::mt19937::default_seed};
public:
    explicit Selection(LoadBalancing policy) : policy_(policy) {
        if (policy == LoadBalancing::Random) random_.seed(std::random_device{}());
    }

    std::optional<SessionId> choose(const std::set<SessionId>& allowed,
                                   const std::vector<std::unique_ptr<Session>>& sessions) {
        // Complete an existing batch before distributing new messages.
        std::optional<SessionId> fullest;
        std::size_t largest = 0;
        for (const auto id : allowed) {
            const auto count = sessions.at(id)->batch().recordCount();
            if (count > largest) {
                largest = count;
                fullest = id;
            }
        }
        if (fullest) return fullest;

        auto start = next_ % allowed.size();
        if (policy_ == LoadBalancing::Random)
            start = std::uniform_int_distribution<std::size_t>(0, allowed.size() - 1)(random_);
        for (std::size_t offset = 0; offset < allowed.size(); ++offset) {
            const auto index = (start + offset) % allowed.size();
            const auto id = *std::next(allowed.begin(), static_cast<std::ptrdiff_t>(index));
            if (sessions.at(id)->hasToken()) {
                next_ = index + 1;
                return id;
            }
        }
        return std::nullopt;
    }
};

} // namespace ff::distributed::sender
