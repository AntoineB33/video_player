win32 {
    QMAKE_CXXFLAGS += /O2 /fp:fast /arch:AVX2
    LIBS += -lvlc -ld3d11 -ldxgi
}