void UIController::Draw() {
    if (show_controls) {
        ImGui::SetNextWindowBgAlpha(0.3f);
        ImGui::Begin("Controls", nullptr, 
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Volume: ");
        ImGui::SliderFloat("##vol", &volume, 0.0f, 1.0f);
        ImGui::End();
    }
}