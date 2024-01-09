#pragma once
#include <cstdlib>
#include <new>
#ifdef _MSC_VER
// Ensure _HAS_EXCEPTIONS is defined
#include <vcruntime.h>
#include <malloc.h>
#endif

#if !((defined(_MSC_VER) && !defined(__clang__)) ? (_HAS_EXCEPTIONS) : (__EXCEPTIONS))
#include <cstdlib>
#endif

// Aligned simple vector.

namespace intgemm {

template <class T> class AlignedVector {
  public:
    AlignedVector() : mem_(nullptr), size_(0) {}

    explicit AlignedVector(std::size_t size, std::size_t alignment = 64 /* CPU cares about this */)
      : size_(size) {
#ifdef _MSC_VER
      mem_ = static_cast<T*>(_aligned_malloc(size * sizeof(T), alignment));
      if (!mem_) {
#  if (defined(_MSC_VER) && !defined(__clang__)) ? (_HAS_EXCEPTIONS) : (__EXCEPTIONS)
        throw std::bad_alloc();
#  else
        std::abort();
#  endif
      }
#else
      if (posix_memalign(reinterpret_cast<void **>(&mem_), alignment, size * sizeof(T))) {
#  if (defined(_MSC_VER) && !defined(__clang__)) ? (_HAS_EXCEPTIONS) : (__EXCEPTIONS)
        throw std::bad_alloc();
#  else
        std::abort();
#  endif
      }
#endif
    }

    template <class InputIt> AlignedVector(InputIt first, InputIt last) 
      : AlignedVector(last - first) {
      std::copy(first, last, begin());
    }

    AlignedVector(AlignedVector &&from) noexcept : mem_(from.mem_), size_(from.size_) {
      from.mem_ = nullptr;
      from.size_ = 0;
    }

    AlignedVector &operator=(AlignedVector &&from) {
      if (this == &from) return *this;
      release();
      mem_ = from.mem_;
      size_ = from.size_;
      from.mem_ = nullptr;
      from.size_ = 0;
      return *this;
    }

    AlignedVector(const AlignedVector&) = delete;
    AlignedVector& operator=(const AlignedVector&) = delete;

    ~AlignedVector() { release(); }

    std::size_t size() const { return size_; }

    T &operator[](std::size_t offset) { return mem_[offset]; }
    const T &operator[](std::size_t offset) const { return mem_[offset]; }

    T *begin() { return mem_; }
    const T *begin() const { return mem_; }
    T *end() { return mem_ + size_; }
    const T *end() const { return mem_ + size_; }

    T *data() { return mem_; }
    const T *data() const { return mem_; }

    template <typename ReturnType>
    ReturnType *as() { return reinterpret_cast<ReturnType*>(mem_); }

  private:
    T *mem_;
    std::size_t size_;

    void release() {
#ifdef _MSC_VER
      _aligned_free(mem_);
#else
      std::free(mem_);
#endif
    }
};

} // namespace intgemm
