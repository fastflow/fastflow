// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cerrno>
#include <system_error>
#if __has_include(<ff/distributed/MTCL/include/mtcl.hpp>)
#include <ff/distributed/MTCL/include/mtcl.hpp>
#else
#include <MTCL/include/mtcl.hpp>
#endif

namespace ff::distributed {

[[noreturn]] inline void transportError(const char* operation) {
    throw std::system_error(errno ? errno : EIO, std::generic_category(), operation);
}

inline bool wouldBlock() { return errno == EAGAIN || errno == EWOULDBLOCK; }

// MTCL::test reports termination, including failures in the TCP backend.
// wait must still be checked before considering a completed operation successful.
inline void complete(MTCL::Request& request) {
    if (MTCL::wait(request) < 0) transportError("distributed MTCL operation");
}

} // namespace ff::distributed
