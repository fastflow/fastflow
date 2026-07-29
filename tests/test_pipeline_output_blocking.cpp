#include <cstdio>
#include <cstring>
#include <vector>

#include <ff/ff.hpp>

using namespace ff;

namespace {

constexpr std::size_t kTaskCount = 64;

struct Task {
    std::size_t id;
    long value;
};

struct Source : ff_node_t<Task> {
    Task* svc(Task*) {
        for (std::size_t i = 0; i < kTaskCount; ++i) {
            Task* task = new Task{i, static_cast<long>(i + 1)};
            if (!ff_send_out(task)) {
                delete task;
                send_failed = true;
                break;
            }
        }
        return EOS;
    }

    bool send_failed = false;
};

struct Transform : ff_node_t<Task> {
    explicit Transform(long offset) : offset(offset) {}

    Task* svc(Task* task) {
        task->value += offset;
        return task;
    }

    const long offset;
};

template <bool UseSendOut>
struct CompositeProducer : ff_node_t<Task> {
    explicit CompositeProducer(long offset) : offset(offset) {}

    Task* svc(Task* task) {
        task->value += offset;
        if (UseSendOut) {
            if (!ff_send_out(task)) {
                delete task;
                send_failed = true;
            }
            return GO_ON;
        }
        return task;
    }

    const long offset;
    bool send_failed = false;
};

struct Sink : ff_node_t<Task> {
    Sink(const char* name, long expected_offset)
        : name(name),
          expected_offset(expected_offset),
          values(kTaskCount, 0),
          seen(kTaskCount, false) {}

    Task* svc(Task* task) {
        ++count;
        if (task->id >= kTaskCount) {
            invalid_id = true;
        } else if (seen[task->id]) {
            duplicate_id = true;
        } else {
            seen[task->id] = true;
            values[task->id] = task->value;
        }
        delete task;
        return GO_ON;
    }

    bool verify() const {
        bool ok = true;
        if (count != kTaskCount) {
            std::fprintf(stderr, "%s: expected %zu outputs, received %zu\n",
                         name, kTaskCount, count);
            ok = false;
        }
        if (invalid_id) {
            std::fprintf(stderr, "%s: received an out-of-range task id\n", name);
            ok = false;
        }
        if (duplicate_id) {
            std::fprintf(stderr, "%s: received a duplicate task id\n", name);
            ok = false;
        }
        for (std::size_t i = 0; i < kTaskCount; ++i) {
            const long expected = static_cast<long>(i + 1) + expected_offset;
            if (!seen[i]) {
                std::fprintf(stderr, "%s: missing output for task %zu\n", name, i);
                ok = false;
            } else if (values[i] != expected) {
                std::fprintf(stderr,
                             "%s: task %zu has value %ld, expected %ld\n",
                             name, i, values[i], expected);
                ok = false;
            }
        }
        return ok;
    }

    const char* const name;
    const long expected_offset;
    std::vector<long> values;
    std::vector<bool> seen;
    std::size_t count = 0;
    bool invalid_id = false;
    bool duplicate_id = false;
};

template <bool UseSendOut>
bool run_a2a_case(const char* name, long producer_offset) {
    constexpr long first_offset = 10;
    constexpr long downstream_offset = 1000;

    Source source;
    Transform first(first_offset);
    CompositeProducer<UseSendOut> producer(producer_offset);
    Transform downstream(downstream_offset);
    Sink sink(name, first_offset + producer_offset + downstream_offset);

    std::vector<ff_node*> first_set{&first};
    std::vector<ff_node*> second_set{&producer};
    ff_a2a a2a;
    a2a.add_firstset(first_set);
    a2a.add_secondset(second_set);

    ff_pipeline pipeline;
    pipeline.add_stage(&source);
    pipeline.add_stage(&a2a);
    pipeline.add_stage(&downstream);
    pipeline.add_stage(&sink);

    if (pipeline.run_and_wait_end() < 0) {
        std::fprintf(stderr, "%s: pipeline execution failed\n", name);
        return false;
    }
    if (source.send_failed || producer.send_failed) {
        std::fprintf(stderr, "%s: ff_send_out failed\n", name);
        return false;
    }
    return sink.verify();
}

template <bool UseSendOut>
bool run_collectorless_farm_case(const char* name, long producer_offset) {
    constexpr long downstream_offset = 2000;

    Source source;
    CompositeProducer<UseSendOut> producer(producer_offset);
    Transform downstream(downstream_offset);
    Sink sink(name, producer_offset + downstream_offset);

    std::vector<ff_node*> workers{&producer};
    ff_farm farm(workers);
    farm.remove_collector();

    ff_pipeline pipeline;
    pipeline.add_stage(&source);
    pipeline.add_stage(&farm);
    pipeline.add_stage(&downstream);
    pipeline.add_stage(&sink);

    if (pipeline.run_and_wait_end() < 0) {
        std::fprintf(stderr, "%s: pipeline execution failed\n", name);
        return false;
    }
    if (source.send_failed || producer.send_failed) {
        std::fprintf(stderr, "%s: ff_send_out failed\n", name);
        return false;
    }
    return sink.verify();
}

template <bool UseSendOut>
bool run_nested_a2a_collectorless_farm_case(const char* name,
                                            long producer_offset) {
    constexpr long first_offset = 30;
    constexpr long downstream_offset = 3000;

    Source source;
    Transform first(first_offset);
    CompositeProducer<UseSendOut> producer(producer_offset);
    Transform downstream(downstream_offset);
    Sink sink(name, first_offset + producer_offset + downstream_offset);

    std::vector<ff_node*> workers{&producer};
    ff_farm farm(workers);
    farm.remove_collector();

    ff_pipeline nested_pipeline;
    nested_pipeline.add_stage(&farm);

    ff_a2a a2a;
    std::vector<ff_node*> first_set{&first};
    a2a.add_firstset(first_set);
    a2a.add_secondset<ff_pipeline>({&nested_pipeline});

    ff_pipeline pipeline;
    pipeline.add_stage(&source);
    pipeline.add_stage(&a2a);
    pipeline.add_stage(&downstream);
    pipeline.add_stage(&sink);

    if (pipeline.run_and_wait_end() < 0) {
        std::fprintf(stderr, "%s: pipeline execution failed\n", name);
        return false;
    }
    if (source.send_failed || producer.send_failed) {
        std::fprintf(stderr, "%s: ff_send_out failed\n", name);
        return false;
    }
    return sink.verify();
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc > 2) {
        std::fprintf(stderr,
                     "usage: %s [a2a-return|a2a-send|farm-return|farm-send|"
                     "nested-return|nested-send]\n",
                     argv[0]);
        return 2;
    }

    const char* selected = argc == 2 ? argv[1] : nullptr;
    bool ok = true;
    bool ran = false;
    if (!selected || std::strcmp(selected, "a2a-return") == 0) {
        ran = true;
        ok = run_a2a_case<false>("A2A svc return", 100) && ok;
    }
    if (!selected || std::strcmp(selected, "a2a-send") == 0) {
        ran = true;
        ok = run_a2a_case<true>("A2A ff_send_out", 200) && ok;
    }
    if (!selected || std::strcmp(selected, "farm-return") == 0) {
        ran = true;
        ok = run_collectorless_farm_case<false>(
                 "collectorless farm svc return", 300) &&
             ok;
    }
    if (!selected || std::strcmp(selected, "farm-send") == 0) {
        ran = true;
        ok = run_collectorless_farm_case<true>(
                 "collectorless farm ff_send_out", 400) &&
             ok;
    }
    if (!selected || std::strcmp(selected, "nested-return") == 0) {
        ran = true;
        ok = run_nested_a2a_collectorless_farm_case<false>(
                 "nested A2A collectorless farm svc return", 500) &&
             ok;
    }
    if (!selected || std::strcmp(selected, "nested-send") == 0) {
        ran = true;
        ok = run_nested_a2a_collectorless_farm_case<true>(
                 "nested A2A collectorless farm ff_send_out", 600) &&
             ok;
    }

    if (!ran) {
        std::fprintf(stderr, "unknown test case: %s\n", selected);
        return 2;
    }

    if (!ok) return 1;

#ifdef BLOCKING_MODE
    std::puts("All pipeline output blocking cases passed in blocking mode");
#else
    std::puts("All pipeline output blocking cases passed in non-blocking mode");
#endif
    return 0;
}
