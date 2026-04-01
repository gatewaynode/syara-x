#ifndef SYARA_X_H
#define SYARA_X_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Return codes for all syara_* functions.
 */
typedef enum SyaraStatus {
    /**
     * Operation succeeded.
     */
    SyaraOk = 0,
    /**
     * A required pointer argument was null.
     */
    SyaraErrNullPtr = 1,
    /**
     * A string argument contained invalid UTF-8.
     */
    SyaraErrUtf8 = 2,
    /**
     * Rules failed to compile. Call `syara_last_error` for details.
     */
    SyaraErrCompile = 3,
    /**
     * Scan failed. Call `syara_last_error` for details.
     */
    SyaraErrScan = 4,
} SyaraStatus;

/**
 * Opaque handle to compiled rules. Create via `syara_compile_*`, free via
 * `syara_rules_free`.
 */
typedef struct SyaraRules SyaraRules;

/**
 * Result for a single rule.
 *
 * `rule_name` is a null-terminated UTF-8 string owned by the containing
 * `SyaraMatchArray`. Do not free it directly.
 */
typedef struct SyaraMatch {
    /**
     * Null-terminated rule name.
     */
    char *rule_name;
    /**
     * 1 if the rule matched, 0 otherwise.
     */
    int matched;
} SyaraMatch;

/**
 * Array of per-rule results returned by `syara_scan` / `syara_scan_file`.
 * Free with `syara_matches_free`.
 */
typedef struct SyaraMatchArray {
    /**
     * Pointer to `count` `SyaraMatch` elements.
     */
    struct SyaraMatch *matches;
    /**
     * Number of elements in `matches`.
     */
    uintptr_t count;
} SyaraMatchArray;

/**
 * Compile rules from a null-terminated `.syara` source string.
 *
 * On success, writes an allocated `SyaraRules*` to `*out` and returns
 * `SYARA_OK`. The caller must free it with `syara_rules_free`.
 *
 * # Safety
 * `src` must be a valid null-terminated C string. `out` must be a valid
 * non-null pointer to a `SyaraRules*`.
 */
enum SyaraStatus syara_compile_str(const char *src, struct SyaraRules **out);

/**
 * Compile rules from a null-terminated file path.
 *
 * On success, writes an allocated `SyaraRules*` to `*out` and returns
 * `SYARA_OK`. The caller must free it with `syara_rules_free`.
 *
 * # Safety
 * `path` must be a valid null-terminated C string. `out` must be a valid
 * non-null pointer to a `SyaraRules*`.
 */
enum SyaraStatus syara_compile_file(const char *path, struct SyaraRules **out);

/**
 * Scan a null-terminated text string against compiled rules.
 *
 * On success, writes an allocated `SyaraMatchArray*` to `*out` and returns
 * `SYARA_OK`. The caller must free it with `syara_matches_free`.
 *
 * # Safety
 * All pointer arguments must be non-null. `rules` must have been created by
 * `syara_compile_*` and not yet freed. `text` must be a valid null-terminated
 * C string.
 */
enum SyaraStatus syara_scan(const struct SyaraRules *rules,
                            const char *text,
                            struct SyaraMatchArray **out);

/**
 * Scan a file at the given null-terminated path against rules that contain
 * phash patterns. Returns an empty array if no phash rules are defined.
 *
 * On success, writes an allocated `SyaraMatchArray*` to `*out` and returns
 * `SYARA_OK`. The caller must free it with `syara_matches_free`.
 *
 * # Safety
 * All pointer arguments must be non-null. `rules` must have been created by
 * `syara_compile_*` and not yet freed. `path` must be a valid null-terminated
 * C string.
 */
enum SyaraStatus syara_scan_file(const struct SyaraRules *rules,
                                 const char *path,
                                 struct SyaraMatchArray **out);

/**
 * Return the number of compiled rules in a `SyaraRules` handle.
 *
 * Returns 0 if `rules` is null.
 *
 * # Safety
 * `rules` must have been created by `syara_compile_*` and not yet freed, or
 * be null.
 */
uintptr_t syara_rule_count(const struct SyaraRules *rules);

/**
 * Free a `SyaraRules` handle created by `syara_compile_*`.
 *
 * No-op if `rules` is null.
 *
 * # Safety
 * `rules` must have been created by `syara_compile_*` and not yet freed, or
 * be null.
 */
void syara_rules_free(struct SyaraRules *rules);

/**
 * Free a `SyaraMatchArray` created by `syara_scan` or `syara_scan_file`.
 *
 * No-op if `matches` is null.
 *
 * # Safety
 * `matches` must have been created by `syara_scan*` and not yet freed, or be
 * null.
 */
void syara_matches_free(struct SyaraMatchArray *matches);

/**
 * Return the last error message as a null-terminated C string.
 *
 * The returned pointer is valid until the next syara_* call on this thread.
 * Never free this pointer.
 */
const char *syara_last_error(void);

#endif /* SYARA_X_H */
