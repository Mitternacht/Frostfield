/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

/* Encoder-only subset of OpenEXR internal_huf.c, v3.5.2
 * 69b2604fc76e370615438bdc8d2cd95b9349c12e. See LICENSES/OpenEXR-BSD-3-Clause.txt.
 */

#define HUF_ENCBITS 16

#define HUF_ENCSIZE ((1 << HUF_ENCBITS) + 1)

#define SHORT_ZEROCODE_RUN 59
#define LONG_ZEROCODE_RUN 63
#define SHORTEST_LONG_RUN (2 + LONG_ZEROCODE_RUN - SHORT_ZEROCODE_RUN)
#define LONGEST_LONG_RUN (255 + SHORTEST_LONG_RUN)

#ifndef OUR_LIKELY
#    if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#        define OUR_LIKELY(x) (__builtin_expect ((x), 1))
#        define OUR_UNLIKELY(x) (__builtin_expect ((x), 0))
#    else
#        define OUR_LIKELY(x) (x)
#        define OUR_UNLIKELY(x) (x)
#    endif
#endif

#ifndef __cplusplus
// msvc does not seem to properly enable restrict in C compiling. /sigh
#    ifndef _MSC_VER
#        define NO_ALIAS restrict
#    endif
#endif
#ifndef NO_ALIAS
#    define NO_ALIAS
#endif

#define hufLength(code) ((int)((code) & 63))

#define hufCode(code) ((code) >> 6)

#define outputBits(nBits, bits, c, lc, out, outend)             \
    {                                                           \
        int xnBits = nBits;                                     \
        c = (c << xnBits) | (bits);                             \
        lc += xnBits;                                           \
        while ( lc >= 8 )                                       \
        {                                                       \
            if (OUR_UNLIKELY(out >= outend))                    \
                return EXR_ERR_ARGUMENT_OUT_OF_RANGE;           \
            *out++ = (uint8_t) (c >> (lc -= 8));                \
        }                                                       \
    }

//
// Build a "canonical" Huffman code table:
//	- for each (uncompressed) symbol, hcode contains the length
//	  of the corresponding code (in the compressed data)
//	- canonical codes are computed and stored in hcode
//	- the rules for constructing canonical codes are as follows:
//	  * shorter codes (if filled with zeroes to the right)
//	    have a numerically higher value than longer codes
//	  * for codes with the same length, numerical values
//	    increase with numerical symbol values
//	- because the canonical code table can be constructed from
//	  symbol lengths alone, the code table can be transmitted
//	  without sending the actual code values
//	- see http://www.compressconsult.com/huffman/
//

static void
hufCanonicalCodeTable (uint64_t* hcode)
{
    uint64_t n[59];
    uint64_t c;

    //
    // For each i from 0 through 58, count the
    // number of different codes of length i, and
    // store the count in n[i].
    //

    for (int i = 0; i <= 58; ++i)
        n[i] = 0;

    for (int i = 0; i < HUF_ENCSIZE; ++i)
        n[hcode[i]] += 1;

    //
    // For each i from 58 through 1, compute the
    // numerically lowest code with length i, and
    // store that code in n[i].
    //

    c = 0;

    for (int i = 58; i > 0; --i)
    {
        uint64_t nc = ((c + n[i]) >> 1);
        n[i]        = c;
        c           = nc;
    }

    //
    // hcode[i] contains the length, l, of the
    // code for symbol i.  Assign the next available
    // code of length l to the symbol and store both
    // l and the code in hcode[i].
    //

    for (int i = 0; i < HUF_ENCSIZE; ++i)
    {
        uint64_t l = hcode[i];

        if (l > 0) hcode[i] = l | (n[l]++ << 6);
    }
}

//
// Compute Huffman codes (based on frq input) and store them in frq:
//	- code structure is : [63:lsb - 6:msb] | [5-0: bit length];
//	- max code length is 58 bits;
//	- codes outside the range [im-iM] have a null length (unused values);
//	- original frequencies are destroyed;
//	- encoding tables are used by hufEncode() and hufBuildDecTable();
//
// NB: The following code "(*a == *b) && (a > b))" was added to ensure
//     elements in the heap with the same value are sorted by index.
//     This is to ensure, the STL make_heap()/pop_heap()/push_heap() methods
//     produced a resultant sorted heap that is identical across OSes.
//

static inline int
FHeapCompare (uint64_t* a, uint64_t* b)
{
    return ((*a > *b) || ((*a == *b) && (a > b)));
}

static inline void
intern_push_heap (
    uint64_t** first, size_t holeIndex, size_t topIndex, uint64_t* value)
{
    size_t parent = (holeIndex - 1) / 2;
    while (holeIndex > topIndex && FHeapCompare (*(first + parent), value))
    {
        *(first + holeIndex) = *(first + parent);
        holeIndex            = parent;
        parent               = (holeIndex - 1) / 2;
    }
    *(first + holeIndex) = value;
}

static inline void
adjust_heap (uint64_t** first, size_t holeIndex, size_t len, uint64_t* value)
{
    const size_t topIndex    = holeIndex;
    size_t       secondChild = holeIndex;

    while (secondChild < (len - 1) / 2)
    {
        secondChild = 2 * (secondChild + 1);
        if (FHeapCompare (*(first + secondChild), *(first + (secondChild - 1))))
            --secondChild;
        *(first + holeIndex) = *(first + secondChild);
        holeIndex            = secondChild;
    }

    if ((len & 1) == 0 && secondChild == (len - 2) / 2)
    {
        secondChild          = 2 * (secondChild + 1);
        *(first + holeIndex) = *(first + (secondChild - 1));
        holeIndex            = secondChild - 1;
    }

    intern_push_heap (first, holeIndex, topIndex, value);
}

static inline void
push_heap (uint64_t** first, uint64_t** last)
{
    uint64_t* value = *(last - 1);
    intern_push_heap (first, (size_t) (last - first) - 1, 0, value);
}

static inline void
intern_pop_heap (uint64_t** first, uint64_t** last, uint64_t** result)
{
    uint64_t* value = *result;
    *result         = *first;
    adjust_heap (first, 0, (size_t) (last - first), value);
}

static inline void
pop_heap (uint64_t** first, uint64_t** last)
{
    if (last - first > 1)
    {
        --last;
        intern_pop_heap (first, last, last);
    }
}

static void
make_heap (uint64_t** first, uint64_t len)
{
    size_t parent;

    if (len < 2) return;
    parent = (len - 2) / 2;

    while (1)
    {
        uint64_t* value = *(first + parent);
        adjust_heap (first, parent, len, value);
        if (parent == 0) return;
        --parent;
    }
}

static void
hufBuildEncTable (
    uint64_t*  frq,
    uint32_t*  im,
    uint32_t*  iM,
    uint32_t*  hlink,
    uint64_t** fHeap,
    uint64_t*  scode)
{
    //
    // This function assumes that when it is called, array frq
    // indicates the frequency of all possible symbols in the data
    // that are to be Huffman-encoded.  (frq[i] contains the number
    // of occurrences of symbol i in the data.)
    //
    // The loop below does three things:
    //
    // 1) Finds the minimum and maximum indices that point
    //    to non-zero entries in frq:
    //
    //     frq[im] != 0, and frq[i] == 0 for all i < im
    //     frq[iM] != 0, and frq[i] == 0 for all i > iM
    //
    // 2) Fills array fHeap with pointers to all non-zero
    //    entries in frq.
    //
    // 3) Initializes array hlink such that hlink[i] == i
    //    for all array entries.
    //
    uint32_t nf = 0;

    *im = 0;

    while (!frq[*im])
        (*im)++;

    for (uint32_t i = *im; i < HUF_ENCSIZE; i++)
    {
        hlink[i] = i;

        if (frq[i])
        {
            fHeap[nf] = &frq[i];
            ++nf;
            *iM = i;
        }
    }

    //
    // Add a pseudo-symbol, with a frequency count of 1, to frq;
    // adjust the fHeap and hlink array accordingly.  Function
    // hufEncode() uses the pseudo-symbol for run-length encoding.
    //

    (*iM)++;
    frq[*iM]  = 1;
    fHeap[nf] = &frq[*iM];
    ++nf;

    //
    // Build an array, scode, such that scode[i] contains the number
    // of bits assigned to symbol i.  Conceptually this is done by
    // constructing a tree whose leaves are the symbols with non-zero
    // frequency:
    //
    //     Make a heap that contains all symbols with a non-zero frequency,
    //     with the least frequent symbol on top.
    //
    //     Repeat until only one symbol is left on the heap:
    //
    //         Take the two least frequent symbols off the top of the heap.
    //         Create a new node that has first two nodes as children, and
    //         whose frequency is the sum of the frequencies of the first
    //         two nodes.  Put the new node back into the heap.
    //
    // The last node left on the heap is the root of the tree.  For each
    // leaf node, the distance between the root and the leaf is the length
    // of the code for the corresponding symbol.
    //
    // The loop below doesn't actually build the tree; instead we compute
    // the distances of the leaves from the root on the fly.  When a new
    // node is added to the heap, then that node's descendants are linked
    // into a single linear list that starts at the new node, and the code
    // lengths of the descendants (that is, their distance from the root
    // of the tree) are incremented by one.
    //

    make_heap (fHeap, nf);

    memset (scode, 0, sizeof (uint64_t) * HUF_ENCSIZE);

    while (nf > 1)
    {
        uint32_t mm, m;
        //
        // Find the indices, mm and m, of the two smallest non-zero frq
        // values in fHeap, add the smallest frq to the second-smallest
        // frq, and remove the smallest frq value from fHeap.
        //

        mm = (uint32_t) (fHeap[0] - frq);
        pop_heap (&fHeap[0], &fHeap[nf]);
        --nf;

        m = (uint32_t) (fHeap[0] - frq);
        pop_heap (&fHeap[0], &fHeap[nf]);

        frq[m] += frq[mm];
        push_heap (&fHeap[0], &fHeap[nf]);

        //
        // The entries in scode are linked into lists with the
        // entries in hlink serving as "next" pointers and with
        // the end of a list marked by hlink[j] == j.
        //
        // Traverse the lists that start at scode[m] and scode[mm].
        // For each element visited, increment the length of the
        // corresponding code by one bit. (If we visit scode[j]
        // during the traversal, then the code for symbol j becomes
        // one bit longer.)
        //
        // Merge the lists that start at scode[m] and scode[mm]
        // into a single list that starts at scode[m].
        //

        //
        // Add a bit to all codes in the first list.
        //

        for (uint32_t j = m;; j = hlink[j])
        {
            scode[j]++;

            if (hlink[j] == j)
            {
                //
                // Merge the two lists.
                //

                hlink[j] = mm;
                break;
            }
        }

        //
        // Add a bit to all codes in the second list
        //

        for (uint32_t j = mm;; j = hlink[j])
        {
            scode[j]++;

            if (hlink[j] == j) break;
        }
    }

    //
    // Build a canonical Huffman code table, replacing the code
    // lengths in scode with (code, code length) pairs.  Copy the
    // code table from scode into frq.
    //

    hufCanonicalCodeTable (scode);
    memcpy (frq, scode, sizeof (uint64_t) * HUF_ENCSIZE);
}

//
// Pack an encoding table:
//	- only code lengths, not actual codes, are stored
//	- runs of zeroes are compressed as follows:
//
//	  unpacked		packed
//	  --------------------------------
//	  1 zero		0	(6 bits)
//	  2 zeroes		59
//	  3 zeroes		60
//	  4 zeroes		61
//	  5 zeroes		62
//	  n zeroes (6 or more)	63 n-6	(6 + 8 bits)
//

static exr_result_t
hufPackEncTable (
    const uint64_t* hcode, // i : encoding table [HUF_ENCSIZE]
    uint32_t        im,    // i : min hcode index
    uint32_t        iM,    // i : max hcode index
    uint8_t**       pcode, // o : ptr to packed table (updated)
    uint8_t*        pend)         // i : max size of table
{
    uint8_t*     out = *pcode;
    uint64_t     c   = 0;
    int          lc  = 0;

    for (; im <= iM; im++)
    {
        int l = hufLength (hcode[im]);

        if (l == 0)
        {
            uint64_t zerun = 1;

            while ((im < iM) && (zerun < LONGEST_LONG_RUN))
            {
                if (hufLength (hcode[im + 1]) > 0) break;
                im++;
                zerun++;
            }

            if (zerun >= 2)
            {
                if (zerun >= SHORTEST_LONG_RUN)
                {
                    outputBits (6, LONG_ZEROCODE_RUN, c, lc, out, pend);
                    outputBits (8, zerun - SHORTEST_LONG_RUN, c, lc, out, pend);
                }
                else
                {
                    outputBits (6, SHORT_ZEROCODE_RUN + zerun - 2, c, lc, out, pend);
                }
                continue;
            }
        }

        outputBits (6, (uint64_t) l, c, lc, out, pend);
    }

    if (lc > 0)
    {
        if (out >= pend) return EXR_ERR_ARGUMENT_OUT_OF_RANGE;
        *out++ = (uint8_t) (c << (8 - lc));
    }

    *pcode = out;
    return EXR_ERR_SUCCESS;
}

#define outputCode(code, c, lc, out, outend) \
    outputBits (hufLength (code), hufCode (code), c, lc, out, outend)

#define sendCode(sCode, runCount, runCode, c, lc, out, outend)      \
    if ((hufLength (sCode) + hufLength (runCode) + 8) <             \
        (hufLength (sCode) * runCount))                             \
    {                                                               \
        outputCode (sCode, c, lc, out, outend);                     \
        outputCode (runCode, c, lc, out, outend);                   \
        outputBits (8, (uint64_t) runCount, c, lc, out, outend);    \
    }                                                               \
    else                                                            \
    {                                                               \
        while ( runCount-- >= 0 )                                   \
        {                                                           \
            outputCode (sCode, c, lc, out, outend);                 \
        }                                                           \
    }

//
// Encode (compress) ni values based on the Huffman encoding table hcode:
//

static inline exr_result_t
hufEncode (
    const uint64_t* NO_ALIAS hcode,
    const uint16_t* NO_ALIAS in,
    const uint64_t           ni,
    uint32_t                 rlc,
    uint8_t* NO_ALIAS        out,
    uint8_t* NO_ALIAS        outend,
    uint32_t* NO_ALIAS       outbytes)
{
    uint8_t* outStart = out;
    uint64_t c        = 0; // bits not yet written to out
    int      lc       = 0; // number of valid bits in c (LSB)
    uint32_t s        = in[0];
    int      cs       = 0;
    uint64_t runCode  = hcode[rlc];
    //
    // Loop on input values
    //

    for (uint64_t i = 1; i < ni; i++)
    {
        uint32_t ns = in[i];
        //
        // Count same values or send code
        //

        if (cs == 255 || s != ns)
        {
            sendCode (hcode[s], cs, runCode, c, lc, out, outend);
            cs = 0;
            s = ns;
        }
        else
        {
            ++cs;
        }
    }

    //
    // Send remaining code
    //
    sendCode (hcode[s], cs, runCode, c, lc, out, outend);

    if (lc)
    {
        if (out >= outend) return EXR_ERR_ARGUMENT_OUT_OF_RANGE;
        *out = (c << (8 - lc)) & 0xff;
    }

    c = (((uintptr_t) out) - ((uintptr_t) outStart)) * 8 + (uint64_t) (lc);
    if (c > (uint64_t) UINT32_MAX) return EXR_ERR_ARGUMENT_OUT_OF_RANGE;
    *outbytes = (uint32_t) c;

    return EXR_ERR_SUCCESS;
}

static inline void
countFrequencies (uint64_t* freq, const uint16_t* data, uint64_t n)
{
    memset (freq, 0, HUF_ENCSIZE * sizeof (uint64_t));
    for (uint64_t i = 0; i < n; ++i)
        ++freq[data[i]];
}

static inline void
writeUInt (uint8_t* b, uint32_t i)
{
    b[0] = (uint8_t) (i);
    b[1] = (uint8_t) (i >> 8);
    b[2] = (uint8_t) (i >> 16);
    b[3] = (uint8_t) (i >> 24);
}

static uint64_t
internal_exr_huf_compress_spare_bytes (void)
{
    uint64_t ret = 0;
    ret += HUF_ENCSIZE * sizeof (uint64_t);  // freq
    ret += HUF_ENCSIZE * sizeof (uint64_t);  // scode
    ret += HUF_ENCSIZE * sizeof (uint64_t*); // fheap
    ret += HUF_ENCSIZE * sizeof (uint32_t);  // hlink
    return ret;
}

static exr_result_t
internal_huf_compress (
    uint64_t*       encbytes,
    void*           out,
    uint64_t        outsz,
    const uint16_t* raw,
    uint64_t        nRaw,
    void*           spare,
    uint64_t        sparebytes)
{
    exr_result_t rv;
    uint64_t*    freq;
    uint32_t*    hlink;
    uint64_t**   fHeap;
    uint64_t*    scode;
    uint32_t     im = 0;
    uint32_t     iM = 0;
    uint32_t     tableLength, nBits, dataLength;
    uint8_t*     dataStart;
    uint8_t*     compressed = (uint8_t*) out;
    uint8_t*     tableStart = compressed + 20;
    uint8_t*     tableEnd   = tableStart;
    uint8_t*     maxcompout = compressed + outsz;

    if (nRaw == 0)
    {
        *encbytes = 0;
        return EXR_ERR_SUCCESS;
    }

    if (outsz < 20) return EXR_ERR_INVALID_ARGUMENT;
    // The spare buffer is partitioned using fixed offsets, so it only needs
    // to be *at least* the required size. Callers (e.g. the DWA compressor)
    // may legitimately pass a larger, reused scratch buffer.
    if (sparebytes < internal_exr_huf_compress_spare_bytes ())
        return EXR_ERR_INVALID_ARGUMENT;

    freq  = (uint64_t*) spare;
    scode = freq + HUF_ENCSIZE;
    fHeap = (uint64_t**) (scode + HUF_ENCSIZE);
    hlink = (uint32_t*) (fHeap + HUF_ENCSIZE);

    countFrequencies (freq, raw, nRaw);

    hufBuildEncTable (freq, &im, &iM, hlink, fHeap, scode);

    rv = hufPackEncTable (freq, im, iM, &tableEnd, maxcompout);

    if (rv != EXR_ERR_SUCCESS) return rv;
    tableLength =
        (uint32_t) (((uintptr_t) tableEnd) - ((uintptr_t) tableStart));
    dataStart = tableEnd;

    rv = hufEncode (freq, raw, nRaw, iM, dataStart, maxcompout, &nBits);
    if (rv != EXR_ERR_SUCCESS) return rv;

    dataLength = (nBits + 7) / 8;

    writeUInt (compressed, im);
    writeUInt (compressed + 4, iM);
    writeUInt (compressed + 8, tableLength);
    writeUInt (compressed + 12, nBits);
    writeUInt (compressed + 16, 0); // room for future extensions

    *encbytes =
        (((uintptr_t) dataStart) + ((uintptr_t) dataLength) -
         ((uintptr_t) compressed));
    return EXR_ERR_SUCCESS;
}
