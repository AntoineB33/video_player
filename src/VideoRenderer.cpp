// OpenGL Texture Upload (Simplified)
void VideoRenderer::UploadFrame(AVFrame* frame) {
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
                 frame->width, frame->height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, frame->data[0]);
}

// Render Quad with Shader
void VideoRenderer::Draw() {
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shader_program);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}