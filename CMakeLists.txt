find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale)
include_directories(${FFMPEG_INCLUDE_DIRS})
target_link_libraries(YourApp ${FFMPEG_LIBRARIES})