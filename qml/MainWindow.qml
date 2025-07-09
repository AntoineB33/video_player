// MainWindow.qml
import QtQuick 2.15
import QtQuick.Window 2.15

Window {
    visible: true
    width: 1280
    height: 720

    VideoSurface {
        id: videoSurface
        anchors.fill: parent
        engine: videoEngine
    }

    VideoEngine {
        id: videoEngine
    }

    // Simple controls
    Row {
        Button {
            text: "Open"
            onClicked: videoEngine.openFile(fileDialog.fileUrl)
        }
        Button {
            text: "Play"
            onClicked: videoEngine.play()
        }
        Button {
            text: "Pause"
            onClicked: videoEngine.pause()
        }
    }

    FileDialog { id: fileDialog }
}