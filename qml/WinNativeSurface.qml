import QtQuick 2.15
import QtQuick.Window 2.15

Item {
    id: root
    property var engine
    
    // Get native window handle
    function getVideoSurface() {
        return nativeSurface.winId;
    }
    
    Window {
        id: nativeSurface
        width: parent.width
        height: parent.height
        visible: true
        flags: Qt.FramelessWindowHint | Qt.WindowTransparentForInput
        
        Component.onCompleted: {
            engine.setOutputWindow(getVideoSurface())
        }
    }
}