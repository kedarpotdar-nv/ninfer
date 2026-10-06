// Media decoding without FFmpeg (NINFER_ENABLE_FFMPEG=OFF): every request is rejected with the
// InvalidInput error the callers already translate into a client-visible unsupported-media error.
#include "media/decode/decode.h"

namespace ninfer::media::decode {
namespace {

[[noreturn]] void unavailable() {
    throw Error(ErrorKind::InvalidInput,
                "media decoding is not available in this build (compiled without FFmpeg)");
}

} // namespace

ImageInfo inspect_image(std::span<const std::uint8_t>, const Policy&) { unavailable(); }

VideoInfo inspect_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unavailable();
}

Image decode_image(std::span<const std::uint8_t>, const Policy&) { unavailable(); }

Video decode_video(std::span<const std::uint8_t>, const Policy&, double, int, int) {
    unavailable();
}

} // namespace ninfer::media::decode
