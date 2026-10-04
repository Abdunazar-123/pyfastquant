#pragma once

#include <cstddef>
#include <cstdlib>
#include <new>

#ifdef _WIN32
    #include <malloc.h>
#endif

namespace PyFastQuant {
namespace Engine {

static constexpr std::size_t CACHE_LINE_SIZE = 64;

template<typename T>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;

        #ifdef _WIN32
            ptr = _aligned_malloc(n * sizeof(T), CACHE_LINE_SIZE);
            if (!ptr) throw std::bad_alloc();
        #else
            if (posix_memalign(&ptr, CACHE_LINE_SIZE, n * sizeof(T))) {
                throw std::bad_alloc();
            }
        #endif

        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) {
        #ifdef _WIN32
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U>&) const noexcept { return true; }
    template<typename U>
    bool operator!=(const AlignedAllocator<U>&) const noexcept { return false; }
};

} // namespace Engine
} // namespace PyFastQuant
