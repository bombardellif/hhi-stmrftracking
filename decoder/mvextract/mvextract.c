#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/motion_vector.h>
#include <libavutil/parseutils.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>

#include "streambuffer.h"
#include "mvextract.h"

bool init() {
    av_register_all();
    return avformat_network_init() == 0;
}

static bool extract_mvs(AVFrame* frame, StreamBuffer* buffer) {
    bool ret = true;
    AVFrameSideData *sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
    if (sd) {
        const AVMotionVector *mvs = (const AVMotionVector *)sd->data;
        int n = sd->size / sizeof(AVMotionVector);
        int16_t (*array)[4] = (int16_t (*)[4])calloc(n, sizeof(int16_t[4]));
        if (array) {
            for (int i=0; i<n; i++) {
                const AVMotionVector *mv = &mvs[i];
                array[i][0] = mv->dst_x;
                array[i][1] = mv->dst_y;
                array[i][2] = mv->motion_x;
                array[i][3] = mv->motion_y;
            }
            ret = streambuffer_push(buffer, array, n);
        } else {
            ret = false;
        }
    } else {
        // Frame does not have MVs (possibly I Frame), add empty node
        ret = streambuffer_push(buffer, NULL, 0);
    }
    return ret;
}

static void* consume_stream(void* param) {
    StreamBuffer *stream_buff = (StreamBuffer*)param;
    const char *url = stream_buff->url;
    AVFormatContext *format_ctx = NULL;
    AVDictionary *decode_opts = NULL;
    AVCodecContext *codec_ctx = NULL;
    AVFrame *frame = NULL;
    // Open the input video stream
    if (avformat_open_input(&format_ctx, url, NULL, NULL))
        goto finalize;
    if (avformat_find_stream_info(format_ctx, NULL) < 0)
        goto finalize;

    // Check if stream is encoded in H.264 and set decoder up
    AVCodec *codec = NULL;
    AVCodecParameters *codecpar = NULL;
    int stream_id = 0;
    while (stream_id < format_ctx->nb_streams) {
        codecpar = format_ctx->streams[stream_id]->codecpar;
        if (codecpar->codec_type == AVMEDIA_TYPE_VIDEO
        && codecpar->codec_id == AV_CODEC_ID_H264) {
            codec = avcodec_find_decoder(codecpar->codec_id);
            break;
        }
        stream_id++;
    }
    if (!codec)
        goto finalize;
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx)
        goto finalize;
    if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0)
        goto finalize;
    if (av_dict_set(&decode_opts, "flags2", "+export_mvs", 0) < 0)
        goto finalize;
    if (avcodec_open2(codec_ctx, codec, &decode_opts) < 0)
        goto finalize;

    // Decode video stream
    frame = av_frame_alloc();
    if (!frame)
        goto finalize;
    AVPacket packet = {0};
    av_init_packet(&packet);
    bool consume = true;
    while (consume && av_read_frame(format_ctx, &packet) >= 0) {
        if (packet.stream_index == stream_id) {
            int status = avcodec_send_packet(codec_ctx, &packet);

            if (!status) {
                status = avcodec_receive_frame(codec_ctx, frame);
                if (!status)
                    consume = extract_mvs(frame, stream_buff);
                else
                    consume = false;
            } else {
                consume = false;
            }
        }
        av_packet_unref(&packet);
    }
    // Flush decoder
    avcodec_send_packet(codec_ctx, NULL);
finalize:
    // Signalize that buffer won't be fed anymore
    streambuffer_set_eof(stream_buff);
    // Free the whole environment
    av_frame_free(&frame);
    avcodec_free_context(&codec_ctx);
    av_dict_free(&decode_opts);
    avformat_close_input(&format_ctx);
    return NULL;
}

StreamBuffer* read_videostream(const char url[]) {
    // Create a stream buffer (initialized with zeros) and launch thread
    StreamBuffer* new_buffer = streambuffer_new(url);
    if (new_buffer) {
        // start thread that consumes the video stream (buffer producer)
        int stat = pthread_create(
            &(new_buffer->th_producer),
            NULL,
            consume_stream,
            (void*)new_buffer);
        if (stat) {
            streambuffer_free(new_buffer);
            new_buffer = NULL;
        }
    }
    return new_buffer;
}

void stop(StreamBuffer* buff) {
    streambuffer_unset_active(buff);
}

bool get_next_frame(StreamBuffer* buff, int16_t (**out)[4], size_t* size) {
    return streambuffer_pop(buff, out, size);
}

void destroy(StreamBuffer* buff) {
    if (buff) {
        stop(buff);
        // before freeing memory, let the respective thread terminate
        pthread_join(buff->th_producer, NULL);
        streambuffer_free(buff);
    }
}
