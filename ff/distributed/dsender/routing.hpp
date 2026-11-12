// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace ff::distributed::sender {

using SessionId = std::size_t;

/** Topology lookup, independent of buffering and token availability.
 * A destination identifies one session. An anonymous message may use only a
 * session explicitly allowed for its source. Missing routes fail immediately.
 */
class Routing {
    std::unordered_map<int, SessionId> destinations_;
    std::unordered_map<int, std::set<SessionId>> sources_;
public:
    void addDestination(int destination, SessionId session) { destinations_[destination] = session; }
    void allowSource(int source, SessionId session) { sources_[source].insert(session); }

    SessionId destination(int node) const {
        const auto found = destinations_.find(node);
        if (found == destinations_.end())
            throw std::runtime_error("no remote session for destination node " + std::to_string(node));
        return found->second;
    }

    const std::set<SessionId>& allowed(int source) const {
        const auto found = sources_.find(source);
        if (found == sources_.end() || found->second.empty())
            throw std::runtime_error("no remote sessions for source node " + std::to_string(source));
        return found->second;
    }
};

} // namespace ff::distributed::sender
