import QtQuick 2.15
import QtQuick.Controls 2.15

ApplicationWindow {
    visible: true
    width: 1280
    height: 720
    
    WinNativeSurface {
        id: videoSurface
        anchors.fill: parent
        engine: videoEngine
    }
    
    VideoEngine {
        id: videoEngine
    }
    
    CheckBox {
        text: "Enable HDR"
        onCheckedChanged: videoEngine.enableHDR(checked)
    }
}