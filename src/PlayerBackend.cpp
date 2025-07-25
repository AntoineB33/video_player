#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

class PlayerBackend {
public:
    bool Open(const char* file) {
        // Open file
        if (avformat_open_input(&format_ctx, file, nullptr, nullptr) != 0)
            return false;

        // Find video stream
        AVCodecParameters* codec_params;
        AVCodec* codec;
        stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
        codec_params = format_ctx->streams[stream_index]->codecpar;

        // Initialize decoder with hardware acceleration
        codec_ctx = avcodec_alloc_context3(codec);
        avcodec_parameters_to_context(codec_ctx, codec_params);
        avcodec_open2(codec_ctx, codec, nullptr);

        // Enable GPU decoding (e.g., CUDA/VAAPI)
        codec_ctx->hw_device_ctx = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_CUDA);
        return true;
    }

    void DecodeFrame() {
        AVPacket packet;
        AVFrame* frame = av_frame_alloc();
        while (av_read_frame(format_ctx, &packet) >= 0) {
            if (packet.stream_index == stream_index) {
                avcodec_send_packet(codec_ctx, &packet);
                if (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    // Send frame to renderer
                    renderer->UploadFrame(frame);
                }
            }
            av_packet_unref(&packet);
        }
    }
};