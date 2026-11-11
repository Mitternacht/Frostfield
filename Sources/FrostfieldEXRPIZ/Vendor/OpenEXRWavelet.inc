/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

/* Encoder-only subset of OpenEXR internal_piz.c, v3.5.2
 * 69b2604fc76e370615438bdc8d2cd95b9349c12e. See LICENSES/OpenEXR-BSD-3-Clause.txt.
 */

#define USHORT_RANGE (1 << 16)
#define BITMAP_SIZE (USHORT_RANGE >> 3)

static void
bitmapFromData (
    const uint16_t* data,
    uint64_t        nData,
    uint8_t*        bitmap,
    uint16_t*       minNonZero,
    uint16_t*       maxNonZero)
{
    uint16_t mnnz = BITMAP_SIZE - 1;
    uint16_t mxnz = 0;

    for (int i = 0; i < BITMAP_SIZE; ++i)
        bitmap[i] = 0;

    for (uint64_t i = 0; i < nData; ++i)
        bitmap[data[i] >> 3] |= (1 << (data[i] & 7));

    bitmap[0] &= ~1; // zero is not explicitly stored in
                     // the bitmap; we assume that the
                     // data always contain zeroes

    for (uint16_t i = 0; i < BITMAP_SIZE; ++i)
    {
        if (bitmap[i])
        {
            if (mnnz > i) mnnz = i;
            if (mxnz < i) mxnz = i;
        }
    }
    *minNonZero = mnnz;
    *maxNonZero = mxnz;
}

static inline uint16_t
forwardLutFromBitmap (const uint8_t* bitmap, uint16_t* lut)
{
    uint16_t k = 0;

    for (uint32_t i = 0; i < USHORT_RANGE; ++i)
    {
        if ((i == 0) || (bitmap[i >> 3] & (1 << (i & 7))))
            lut[i] = k++;
        else
            lut[i] = 0;
    }

    return k - 1;
}

static inline void
applyLut (const uint16_t lut[NO_ALIAS USHORT_RANGE], uint16_t* NO_ALIAS data, uint64_t nData)
{
    /* partially unrolling is noticeably faster */
    uint16_t dvals[8];
    while ( nData > 8 )
    {
        memcpy (dvals, data, sizeof(uint16_t)*8);
        for ( int i = 0; i < 8; ++i )
            dvals[i] = lut[dvals[i]];
        memcpy (data, dvals, sizeof(uint16_t)*8);
        data += 8;
        nData -= 8;
    }

    for (uint64_t i = 0; i < nData; ++i)
        data[i] = lut[data[i]];
}

/**************************************/
//
// Wavelet basis functions without modulo arithmetic; they produce
// the best compression ratios when the wavelet-transformed data are
// Huffman-encoded, but the wavelet transform works only for 14-bit
// data (untransformed data values must be less than (1 << 14)).
//

static inline void
wenc14 (uint16_t a, uint16_t b, uint16_t* l, uint16_t* h)
{
    int16_t as = (int16_t) a;
    int16_t bs = (int16_t) b;

    int16_t ms = (as + bs) >> 1;
    int16_t ds = as - bs;

    *l = (uint16_t) ms;
    *h = (uint16_t) ds;
}

//
// Wavelet basis functions with modulo arithmetic; they work with full
// 16-bit data, but Huffman-encoding the wavelet-transformed data doesn't
// compress the data quite as well.
//

#define NBITS ((int) 16)
#define A_OFFSET ((int) 1 << (NBITS - 1))
#define M_OFFSET ((int) 1 << (NBITS - 1))
#define MOD_MASK ((int) (1 << NBITS) - 1)

static inline void
wenc16 (uint16_t a, uint16_t b, uint16_t* l, uint16_t* h)
{
    int ao = (((int) a) + A_OFFSET) & MOD_MASK;
    int m  = ((ao + ((int) b)) >> 1);
    int d  = ao - ((int) b);

    if (d < 0) m = (m + M_OFFSET) & MOD_MASK;

    d &= MOD_MASK;

    *l = (uint16_t) m;
    *h = (uint16_t) d;
}

static void
wav_2D_encode (uint16_t* in, int nx, int ox, int ny, int oy, uint16_t mx)
{
    int     w14  = (mx < (1 << 14)) ? 1 : 0;
    int     n    = (nx > ny) ? ny : nx;
    int     p    = 1; // == 1 <<  level
    int     p2   = 2; // == 1 << (level+1)
    int64_t oy64 = oy;

    //
    // Hierarchical loop on smaller dimension n
    //

    while (p2 <= n)
    {
        uint16_t* py  = in;
        uint16_t* ey  = in + oy64 * (ny - p2);
        int64_t   oy1 = oy64 * p;
        int64_t   oy2 = oy64 * p2;
        int       ox1 = ox * p;
        int       ox2 = ox * p2;
        uint16_t  i00, i01, i10, i11;

        //
        // Y loop
        //

        for (; py <= ey; py += oy2)
        {
            uint16_t* px = py;
            uint16_t* ex = py + ox * (nx - p2);

            //
            // X loop
            //

            for (; px <= ex; px += ox2)
            {
                uint16_t* p01 = px + ox1;
                uint16_t* p10 = px + oy1;
                uint16_t* p11 = p10 + ox1;

                //
                // 2D wavelet encoding
                //

                if (w14)
                {
                    wenc14 (*px, *p01, &i00, &i01);
                    wenc14 (*p10, *p11, &i10, &i11);
                    wenc14 (i00, i10, px, p10);
                    wenc14 (i01, i11, p01, p11);
                }
                else
                {
                    wenc16 (*px, *p01, &i00, &i01);
                    wenc16 (*p10, *p11, &i10, &i11);
                    wenc16 (i00, i10, px, p10);
                    wenc16 (i01, i11, p01, p11);
                }
            }

            //
            // Encode (1D) odd column (still in Y loop)
            //

            if (nx & p)
            {
                uint16_t* p10 = px + oy1;

                if (w14)
                    wenc14 (*px, *p10, px, p10);
                else
                    wenc16 (*px, *p10, px, p10);
            }
        }

        //
        // Encode (1D) odd line (must loop in X)
        //

        if (ny & p)
        {
            uint16_t* px = py;
            uint16_t* ex = py + ox * (nx - p2);

            for (; px <= ex; px += ox2)
            {
                uint16_t* p01 = px + ox1;

                if (w14)
                    wenc14 (*px, *p01, px, p01);
                else
                    wenc16 (*px, *p01, px, p01);
            }
        }

        //
        // Next level
        //

        p = p2;
        p2 <<= 1;
    }
}
