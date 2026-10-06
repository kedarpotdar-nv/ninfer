find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# FFmpeg (image/video decoding) and libcurl (remote media URLs) come from pkg-config on Linux.
# Windows builds default to compiling both out; the engine and the text serving path do not
# need them, and media requests then fail with an explicit unsupported-media error.
if(WIN32)
  set(NINFER_MEDIA_DEFAULT OFF)
else()
  set(NINFER_MEDIA_DEFAULT ON)
endif()
option(NINFER_ENABLE_FFMPEG "Decode images and video with FFmpeg" ${NINFER_MEDIA_DEFAULT})
option(NINFER_ENABLE_CURL "Fetch remote media URLs with libcurl" ${NINFER_MEDIA_DEFAULT})
if(NINFER_ENABLE_FFMPEG OR (NINFER_BUILD_PRODUCT_SUPPORT AND NINFER_ENABLE_CURL))
  find_package(PkgConfig REQUIRED)
endif()
if(NINFER_ENABLE_FFMPEG)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  if(NINFER_ENABLE_CURL)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
