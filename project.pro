# In your project.pro file (Qt Creator)

# Windows
win32 {
    QMAKE_CXXFLAGS += /O2 /fp:fast /arch:AVX2
    LIBS += -lvlc -ld3d11 -ldxgi
}

# macOS
macx {
    QMAKE_CXXFLAGS += -O3 -march=native -flto
    LIBS += -lvlc -framework Metal -framework VideoToolbox
}

# Linux
unix:!macx {
    QMAKE_CXXFLAGS += -O3 -march=native -flto
    LIBS += -lvlc -lvulkan
}