int main() {
    // Initialize GLFW + OpenGL/Vulkan
    PlayerBackend backend;
    backend.Open("assets/video.mp4");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        backend.DecodeFrame();
        renderer.Draw();
        uiController.Draw(); // Conditionally render UI
        glfwSwapBuffers(window);
    }
}