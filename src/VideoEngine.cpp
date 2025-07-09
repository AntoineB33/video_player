#include "VideoEngine.h"
#include <QDebug>

// Callback prototypes
static void *lock(void *opaque, void **planes);
static void unlock(void *opaque, void *picture, void *const *planes);
static void display(void *opaque, void *picture);

VideoEngine::VideoEngine(QObject* parent) : QObject(parent) {
    // const char* args[] = {
    //     // Hardware Acceleration
    //     "--avcodec-hw=dxva2",          // DirectX Video Acceleration (Windows-specific)
    //     "--vout=direct3d11",           // Direct3D 11 output (lower latency than D3D9)
        
    //     // Performance Optimizations
    //     "--drop-late-frames",          // Skip frames that can't be displayed on time
    //     "--skip-frames",               // Allow frame skipping during decoding
    //     "--no-osd",                    // Disable on-screen display
    //     "--high-priority",             // Boost process priority
    //     "--no-video-title-show",       // Disable title overlay
        
    //     // Advanced Windows-specific optimizations
    //     "--directx-hw-yuv",            // Use hardware YUV->RGB conversions
    //     "--swscale-mode=4",            // Fastest scaling algorithm
    //     "--rtsp-tcp",                  // Force TCP for streaming (more reliable)
    //     "--live-caching=300",          // Lower buffer for live streams (ms)
        
    //     // Error resilience
    //     "--avcodec-skiploopfilter=1",  // Skip CPU-intensive loop filtering
    //     "--avcodec-skip-frame=1"       // Allow frame skipping during decoding
    // };

    const char* args[] = {
        "--avcodec-hw=dxva2",        // Windows-specific hardware decoding
        "--vout=direct3d11",         // Direct3D11 output (lower latency than DXVA2)
        "--drop-late-frames",        // Don't stall for late frames
        "--skip-frames",             // Maintain smooth playback under load
        "--no-osd",                  // Disable on-screen display
        "--high-priority",           // Boost process priority
        "--no-video-title-show",     // Disable title overlay
        "--avcodec-threads=0",       // Auto-detect optimal thread count
        "--avcodec-skiploopfilter=1" // Skip CPU-intensive loop filtering
        
        // "--clock-synchro=0",    // Disable internal clock sync
    };
    
    m_vlcInstance = libvlc_new(sizeof(args)/sizeof(args[0]), args);
}

void VideoEngine::setOutputWindow(void* handle) {
    if (!m_mediaPlayer) return;
    
    #if defined(Q_OS_WIN)
        libvlc_media_player_set_hwnd(m_mediaPlayer, handle);
    #elif defined(Q_OS_MAC)
        libvlc_media_player_set_nsobject(m_mediaPlayer, handle);
    #elif defined(Q_OS_LINUX)
        libvlc_media_player_set_xwindow(m_mediaPlayer, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(handle)));
    #endif
}

void VideoEngine::openFile(const QString& path) {
    stop();
    m_media = libvlc_media_new_path(m_vlcInstance, path.toUtf8().constData());
    m_mediaPlayer = libvlc_media_player_new_from_media(m_media);
    libvlc_media_release(m_media);
}

// ... (Implement other control methods)