add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  $<IF:$<BOOL:${WIN32}>,file_io_windows.cpp,file_io.cpp>
  reader.cpp
  binder.cpp
  materializer.cpp
  views.cpp
)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core ninfer::json)
