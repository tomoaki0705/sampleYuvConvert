#include <iostream>
#include <immintrin.h>

int main(int argc, char**argv)
{
    return 0;
}

inline void convertToYUV(int colorspace, int channels, int input_channels, short* UV_data, short* Y_data, const uchar* pix_data, int y_limit, int x_limit, int step, int u_plane_ofs, int v_plane_ofs)
{
    int i, j;
    const int UV_step = 16;
    int  x_scale = channels > 1 ? 2 : 1, y_scale = x_scale;
    int  Y_step = x_scale*8;

    if( channels > 1 )
    {
        if( colorspace == COLORSPACE_YUV444P && y_limit == 16 && x_limit == 16 )
        {
            for( i = 0; i < y_limit; i += 2, pix_data += step*2, Y_data += Y_step*2, UV_data += UV_step )
            {
#ifdef WITH_NEON
                {
                    uint16x8_t masklo = vdupq_n_u16(255);
                    uint16x8_t lane = vld1q_u16((unsigned short*)(pix_data+v_plane_ofs));
                    uint16x8_t t1 = vaddq_u16(vshrq_n_u16(lane, 8), vandq_u16(lane, masklo));
                    lane = vld1q_u16((unsigned short*)(pix_data + v_plane_ofs + step));
                    uint16x8_t t2 = vaddq_u16(vshrq_n_u16(lane, 8), vandq_u16(lane, masklo));
                    t1 = vaddq_u16(t1, t2);
                    vst1q_s16(UV_data, vsubq_s16(vreinterpretq_s16_u16(t1), vdupq_n_s16(128*4)));

                    lane = vld1q_u16((unsigned short*)(pix_data+u_plane_ofs));
                    t1 = vaddq_u16(vshrq_n_u16(lane, 8), vandq_u16(lane, masklo));
                    lane = vld1q_u16((unsigned short*)(pix_data + u_plane_ofs + step));
                    t2 = vaddq_u16(vshrq_n_u16(lane, 8), vandq_u16(lane, masklo));
                    t1 = vaddq_u16(t1, t2);
                    vst1q_s16(UV_data + 8, vsubq_s16(vreinterpretq_s16_u16(t1), vdupq_n_s16(128*4)));
                }

                {
                    int16x8_t lane = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pix_data)));
                    int16x8_t delta = vdupq_n_s16(128);
                    lane = vsubq_s16(lane, delta);
                    vst1q_s16(Y_data, lane);

                    lane = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pix_data+8)));
                    lane = vsubq_s16(lane, delta);
                    vst1q_s16(Y_data + 8, lane);

                    lane = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pix_data+step)));
                    lane = vsubq_s16(lane, delta);
                    vst1q_s16(Y_data+Y_step, lane);

                    lane = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pix_data + step + 8)));
                    lane = vsubq_s16(lane, delta);
                    vst1q_s16(Y_data+Y_step + 8, lane);
                }
#else
                for( j = 0; j < x_limit; j += 2, pix_data += 2 )
                {
                    Y_data[j] = pix_data[0] - 128;
                    Y_data[j+1] = pix_data[1] - 128;
                    Y_data[j+Y_step] = pix_data[step] - 128;
                    Y_data[j+Y_step+1] = pix_data[step+1] - 128;

                    UV_data[j>>1] = pix_data[v_plane_ofs] + pix_data[v_plane_ofs+1] +
                        pix_data[v_plane_ofs+step] + pix_data[v_plane_ofs+step+1] - 128*4;
                    UV_data[(j>>1)+8] = pix_data[u_plane_ofs] + pix_data[u_plane_ofs+1] +
                        pix_data[u_plane_ofs+step] + pix_data[u_plane_ofs+step+1] - 128*4;

                }

                pix_data -= x_limit*input_channels;
#endif
            }
        }
        else
        {
            for( i = 0; i < y_limit; i++, pix_data += step, Y_data += Y_step )
            {
                for( j = 0; j < x_limit; j++, pix_data += input_channels )
                {
                    int Y, U, V;

                    if( colorspace == COLORSPACE_BGR )
                    {
                        int r = pix_data[2];
                        int g = pix_data[1];
                        int b = pix_data[0];

                        Y = DCT_DESCALE( r*y_r + g*y_g + b*y_b, fixc) - 128;
                        U = DCT_DESCALE( r*cb_r + g*cb_g + b*cb_b, fixc );
                        V = DCT_DESCALE( r*cr_r + g*cr_g + b*cr_b, fixc );
                    }
                    else if( colorspace == COLORSPACE_RGBA )
                    {
                        int r = pix_data[0];
                        int g = pix_data[1];
                        int b = pix_data[2];

                        Y = DCT_DESCALE( r*y_r + g*y_g + b*y_b, fixc) - 128;
                        U = DCT_DESCALE( r*cb_r + g*cb_g + b*cb_b, fixc );
                        V = DCT_DESCALE( r*cr_r + g*cr_g + b*cr_b, fixc );
                    }
                    else
                    {
                        Y = pix_data[0] - 128;
                        U = pix_data[v_plane_ofs] - 128;
                        V = pix_data[u_plane_ofs] - 128;
                    }

                    int j2 = j >> (x_scale - 1);
                    Y_data[j] = (short)Y;
                    UV_data[j2] = (short)(UV_data[j2] + U);
                    UV_data[j2 + 8] = (short)(UV_data[j2 + 8] + V);
                }

                pix_data -= x_limit*input_channels;
                if( ((i+1) & (y_scale - 1)) == 0 )
                {
                    UV_data += UV_step;
                }
            }
        }

    }
    else
    {
        for( i = 0; i < y_limit; i++, pix_data += step, Y_data += Y_step )
        {
            for( j = 0; j < x_limit; j++ )
                Y_data[j] = (short)(pix_data[j]*4 - 128*4);
        }
    }
}


