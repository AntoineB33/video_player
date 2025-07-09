// VideoEngine.h
#include <vlc/vlc.h>
#include <QObject>

class VideoEngine : public QObject {
    Q_OBJECT
public:
    Q_INVOKABLE void enableHDR(bool enable);
    explicit VideoEngine(QObject* parent = nullptr);
    ~VideoEngine();

    void openFile(const QString& path);
    void setOutputWindow(void* handle);  // Platform-specific window handle

    // Playback controls
    void play();
    void pause();
    void stop();

private:
    libvlc_instance_t* m_vlcInstance = nullptr;
    libvlc_media_player_t* m_mediaPlayer = nullptr;
    libvlc_media_t* m_media = nullptr;
};