cmake_minimum_required(VERSION 3.16)
project(VideoPlayer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags
if(MSVC)
    add_compile_options(/O2 /fp:fast /arch:AVX2)
    add_link_options(/SUBSYSTEM:WINDOWS)
endif()

# Find dependencies
find_package(Qt6 REQUIRED COMPONENTS Quick)
find_library(VLC_LIBRARY libvlc PATHS "C:/VLC/sdk/lib")
find_path(VLC_INCLUDE_DIR vlc/vlc.h PATHS "C:/VLC/sdk/include")

# Add executable
qt_add_executable(VideoPlayer
    src/main.cpp
    src/VideoEngine.cpp
)

target_include_directories(VideoPlayer PRIVATE
    ${VLC_INCLUDE_DIR}
    include/
)

# Link libraries
target_link_libraries(VideoPlayer PRIVATE
    Qt6::Quick
    ${VLC_LIBRARY}
    d3d11 dxgi
)

# Deploy VLC plugins (add to post-build step)
add_custom_command(TARGET VideoPlayer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "C:/VLC/plugins"
        $<TARGET_FILE_DIR:VideoPlayer>/plugins
)