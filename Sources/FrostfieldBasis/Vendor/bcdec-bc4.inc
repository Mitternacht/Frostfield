/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Sergii Kudlai
 *
 * BC4 endpoint unpacker from bcdec, pinned to
 * 80859ed3b7afb1c527a2a99d70c61457bea72d0c. See LICENSES/bcdec-MIT.txt.
 * Use explicit signed char for the signed endpoint byte.
 */

static void bcdec__bc4_block(const void* compressedBlock, void* decompressedBlock, int destinationPitch, int pixelSize, int isSigned) {
    signed char* sblock;
    unsigned char* ublock;
    int alpha[8];
    int i, j;
    unsigned long long block, indices;

    static int aWeights4[4] = { 13107, 26215, 39321, 52429 };
    static int aWeights6[6] = { 9363, 18724, 28086, 37450, 46812, 56173 };

    block = bcdec__getu64le(compressedBlock);

    if (isSigned) {
        alpha[0] = (signed char)(block & 0xFF);
        alpha[1] = (signed char)((block >> 8) & 0xFF);
        if (alpha[0] < -127) alpha[0] = -127;     /* -128 clamps to -127 */
        if (alpha[1] < -127) alpha[1] = -127;     /* -128 clamps to -127 */
    } else {
        alpha[0] = block & 0xFF;
        alpha[1] = (block >> 8) & 0xFF;
    }

    if (alpha[0] > alpha[1]) {
        /* 6 interpolated alpha values. */
        alpha[2] = (aWeights6[5] * alpha[0] + aWeights6[0] * alpha[1] + 32768) >> 16;   /* 6/7*alpha_0 + 1/7*alpha_1 */
        alpha[3] = (aWeights6[4] * alpha[0] + aWeights6[1] * alpha[1] + 32768) >> 16;   /* 5/7*alpha_0 + 2/7*alpha_1 */
        alpha[4] = (aWeights6[3] * alpha[0] + aWeights6[2] * alpha[1] + 32768) >> 16;   /* 4/7*alpha_0 + 3/7*alpha_1 */
        alpha[5] = (aWeights6[2] * alpha[0] + aWeights6[3] * alpha[1] + 32768) >> 16;   /* 3/7*alpha_0 + 4/7*alpha_1 */
        alpha[6] = (aWeights6[1] * alpha[0] + aWeights6[4] * alpha[1] + 32768) >> 16;   /* 2/7*alpha_0 + 5/7*alpha_1 */
        alpha[7] = (aWeights6[0] * alpha[0] + aWeights6[5] * alpha[1] + 32768) >> 16;   /* 1/7*alpha_0 + 6/7*alpha_1 */
    } else {
        /* 4 interpolated alpha values. */
        alpha[2] = (aWeights4[3] * alpha[0] + aWeights4[0] * alpha[1] + 32768) >> 16;   /* 4/5*alpha_0 + 1/5*alpha_1 */
        alpha[3] = (aWeights4[2] * alpha[0] + aWeights4[1] * alpha[1] + 32768) >> 16;   /* 3/5*alpha_0 + 2/5*alpha_1 */
        alpha[4] = (aWeights4[1] * alpha[0] + aWeights4[2] * alpha[1] + 32768) >> 16;   /* 2/5*alpha_0 + 3/5*alpha_1 */
        alpha[5] = (aWeights4[0] * alpha[0] + aWeights4[3] * alpha[1] + 32768) >> 16;   /* 1/5*alpha_0 + 4/5*alpha_1 */
        alpha[6] = isSigned ? -127 :   0;
        alpha[7] = isSigned ?  127 : 255;
    }

    indices = block >> 16;
    if (isSigned) {
        sblock = (signed char*)decompressedBlock;
        for (i = 0; i < 4; ++i) {
            for (j = 0; j < 4; ++j) {
                sblock[j * pixelSize] = (signed char)alpha[indices & 0x07];
                indices >>= 3;
            }
            sblock += destinationPitch;
        }
    } else {
        ublock = (unsigned char*)decompressedBlock;
        for (i = 0; i < 4; ++i) {
            for (j = 0; j < 4; ++j) {
                ublock[j * pixelSize] = (unsigned char)alpha[indices & 0x07];
                indices >>= 3;
            }
            ublock += destinationPitch;
        }
    }
}
