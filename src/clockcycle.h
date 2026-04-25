#ifndef CLOCKCYCLE_H
#define CLOCKCYCLE_H

#include <stdint.h>

#define POWER9_TB_FREQ 512000000.0

#if defined(__powerpc64__) || defined(__POWERPC__) || defined(__PPC64__)

static inline uint64_t clock_now(void)
{
    unsigned int tbl, tbu0, tbu1;
    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb  %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);
    return (((uint64_t)tbu0) << 32) | tbl;
}

static inline double clock_seconds(uint64_t cycles)
{
    return (double)cycles / POWER9_TB_FREQ;
}

#else

#include <time.h>

static inline uint64_t clock_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 512000000ULL +
           (uint64_t)((double)ts.tv_nsec * 0.512);
}

static inline double clock_seconds(uint64_t cycles)
{
    return (double)cycles / POWER9_TB_FREQ;
}

#endif

#endif
