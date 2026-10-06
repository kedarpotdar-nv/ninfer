// Windows implementation of InputFile: positional reads through ReadFile with OVERLAPPED
// offsets, and FILE_FLAG_NO_BUFFERING for the aligned direct path (the POSIX file uses O_DIRECT).
#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace ninfer::artifact {
namespace {

std::string last_error_text() {
    const DWORD code = ::GetLastError();
    char* buffer     = nullptr;
    const DWORD size = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<char*>(&buffer), 0, nullptr);
    std::string text = size && buffer ? std::string(buffer, size) : "error " + std::to_string(code);
    if (buffer) { ::LocalFree(buffer); }
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
        text.pop_back();
    }
    return text;
}

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + last_error_text());
}

HANDLE open_handle(const std::filesystem::path& path, DWORD extra_flags) {
    return ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | extra_flags, nullptr);
}

HANDLE as_handle(void* value) noexcept { return static_cast<HANDLE>(value); }

// One ReadFile call at an explicit offset; returns bytes read (0 at EOF).
std::size_t read_at(HANDLE handle, std::uint64_t offset, void* destination, std::size_t count,
                    const std::filesystem::path& path, const char* operation) {
    OVERLAPPED overlapped{};
    overlapped.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD read            = 0;
    const DWORD request   = static_cast<DWORD>(std::min<std::size_t>(count, 64ULL * 1024 * 1024));
    if (!::ReadFile(handle, destination, request, &read, &overlapped)) {
        if (::GetLastError() == ERROR_HANDLE_EOF) { return 0; }
        fail(path, operation);
    }
    return read;
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    const HANDLE handle = open_handle(path_, 0);
    if (handle == INVALID_HANDLE_VALUE) { fail(path_, "open"); }
    handle_ = handle;

    BY_HANDLE_FILE_INFORMATION info{};
    if (!::GetFileInformationByHandle(handle, &info)) {
        ::CloseHandle(handle);
        handle_ = nullptr;
        fail(path_, "stat");
    }
    if (info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_DEVICE)) {
        ::CloseHandle(handle);
        handle_ = nullptr;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32U) | info.nFileSizeLow;
}

InputFile::~InputFile() {
    if (direct_handle_ != nullptr) { ::CloseHandle(as_handle(direct_handle_)); }
    if (handle_ != nullptr) { ::CloseHandle(as_handle(handle_)); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const std::size_t read =
            read_at(as_handle(handle_), offset, destination.data(), destination.size(), path_, "read");
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += read;
        destination = destination.subspan(read);
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<DWORD>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_handle_ == nullptr) {
        const HANDLE handle = open_handle(path_, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN);
        if (handle == INVALID_HANDLE_VALUE) { fail(path_, "open direct"); }
        direct_handle_ = handle;
    }
    // Unbuffered reads must stay sector aligned; the final block of the file may come back short.
    std::size_t total = 0;
    while (total < destination.size()) {
        const std::size_t read = read_at(as_handle(direct_handle_), offset + total,
                                         destination.data() + total, destination.size() - total,
                                         path_, "direct read");
        if (!read) { break; }
        total += read;
    }
    return total;
}

} // namespace ninfer::artifact
