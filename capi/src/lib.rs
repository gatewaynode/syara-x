/// C FFI for syara-x.
///
/// Ownership model:
///   - `syara_compile_str` / `syara_compile_file` allocate a `SyaraRules` handle.
///     The caller must free it with `syara_rules_free`.
///   - `syara_scan` / `syara_scan_file` allocate a `SyaraMatchArray`.
///     The caller must free it with `syara_matches_free`.
///   - Strings inside `SyaraMatch` are owned by the array; freeing the array frees them.
///   - `syara_last_error` returns a pointer into thread-local storage — valid until
///     the next call on this thread; never free it.
use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;

// ── Thread-local error store ──────────────────────────────────────────────────

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

fn set_last_error(msg: &str) {
    let cs = CString::new(msg.replace('\0', "?")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = cs);
}

// ── Status codes ─────────────────────────────────────────────────────────────

/// Return codes for all syara_* functions.
#[repr(C)]
pub enum SyaraStatus {
    /// Operation succeeded.
    SyaraOk = 0,
    /// A required pointer argument was null.
    SyaraErrNullPtr = 1,
    /// A string argument contained invalid UTF-8.
    SyaraErrUtf8 = 2,
    /// Rules failed to compile. Call `syara_last_error` for details.
    SyaraErrCompile = 3,
    /// Scan failed. Call `syara_last_error` for details.
    SyaraErrScan = 4,
}

// ── Opaque handle ─────────────────────────────────────────────────────────────

/// Opaque handle to compiled rules. Create via `syara_compile_*`, free via
/// `syara_rules_free`.
pub struct SyaraRules(syara_x::CompiledRules);

// ── Result types ──────────────────────────────────────────────────────────────

/// Result for a single rule.
///
/// `rule_name` is a null-terminated UTF-8 string owned by the containing
/// `SyaraMatchArray`. Do not free it directly.
#[repr(C)]
pub struct SyaraMatch {
    /// Null-terminated rule name.
    pub rule_name: *mut c_char,
    /// 1 if the rule matched, 0 otherwise.
    pub matched: c_int,
}

/// Array of per-rule results returned by `syara_scan` / `syara_scan_file`.
/// Free with `syara_matches_free`.
#[repr(C)]
pub struct SyaraMatchArray {
    /// Pointer to `count` `SyaraMatch` elements.
    pub matches: *mut SyaraMatch,
    /// Number of elements in `matches`.
    pub count: usize,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert a `Vec<syara_x::Match>` into a heap-allocated `SyaraMatchArray`.
fn into_c_array(matches: Vec<syara_x::Match>) -> *mut SyaraMatchArray {
    let mut c_matches: Vec<SyaraMatch> = matches
        .into_iter()
        .map(|m| {
            let rule_name = CString::new(m.rule_name.replace('\0', "?"))
                .unwrap_or_default()
                .into_raw();
            SyaraMatch {
                rule_name,
                matched: m.matched as c_int,
            }
        })
        .collect();

    c_matches.shrink_to_fit();
    let count = c_matches.len();
    let matches_ptr = c_matches.as_mut_ptr();
    std::mem::forget(c_matches);

    Box::into_raw(Box::new(SyaraMatchArray {
        matches: matches_ptr,
        count,
    }))
}

// ── Public C API ──────────────────────────────────────────────────────────────

/// Compile rules from a null-terminated `.syara` source string.
///
/// On success, writes an allocated `SyaraRules*` to `*out` and returns
/// `SYARA_OK`. The caller must free it with `syara_rules_free`.
///
/// # Safety
/// `src` must be a valid null-terminated C string. `out` must be a valid
/// non-null pointer to a `SyaraRules*`.
#[no_mangle]
pub unsafe extern "C" fn syara_compile_str(
    src: *const c_char,
    out: *mut *mut SyaraRules,
) -> SyaraStatus {
    if src.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return SyaraStatus::SyaraErrNullPtr;
    }
    let src_str = match unsafe { CStr::from_ptr(src) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8: {e}"));
            return SyaraStatus::SyaraErrUtf8;
        }
    };
    match syara_x::compile_str(src_str) {
        Ok(rules) => {
            unsafe { *out = Box::into_raw(Box::new(SyaraRules(rules))) };
            SyaraStatus::SyaraOk
        }
        Err(e) => {
            set_last_error(&e.to_string());
            SyaraStatus::SyaraErrCompile
        }
    }
}

/// Compile rules from a null-terminated file path.
///
/// On success, writes an allocated `SyaraRules*` to `*out` and returns
/// `SYARA_OK`. The caller must free it with `syara_rules_free`.
///
/// # Safety
/// `path` must be a valid null-terminated C string. `out` must be a valid
/// non-null pointer to a `SyaraRules*`.
#[no_mangle]
pub unsafe extern "C" fn syara_compile_file(
    path: *const c_char,
    out: *mut *mut SyaraRules,
) -> SyaraStatus {
    if path.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return SyaraStatus::SyaraErrNullPtr;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8: {e}"));
            return SyaraStatus::SyaraErrUtf8;
        }
    };
    match syara_x::compile(path_str) {
        Ok(rules) => {
            unsafe { *out = Box::into_raw(Box::new(SyaraRules(rules))) };
            SyaraStatus::SyaraOk
        }
        Err(e) => {
            set_last_error(&e.to_string());
            SyaraStatus::SyaraErrCompile
        }
    }
}

/// Scan a null-terminated text string against compiled rules.
///
/// On success, writes an allocated `SyaraMatchArray*` to `*out` and returns
/// `SYARA_OK`. The caller must free it with `syara_matches_free`.
///
/// # Safety
/// All pointer arguments must be non-null. `rules` must have been created by
/// `syara_compile_*` and not yet freed. `text` must be a valid null-terminated
/// C string.
#[no_mangle]
pub unsafe extern "C" fn syara_scan(
    rules: *const SyaraRules,
    text: *const c_char,
    out: *mut *mut SyaraMatchArray,
) -> SyaraStatus {
    if rules.is_null() || text.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return SyaraStatus::SyaraErrNullPtr;
    }
    let text_str = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8: {e}"));
            return SyaraStatus::SyaraErrUtf8;
        }
    };
    let matches = unsafe { (*rules).0.scan(text_str) };
    unsafe { *out = into_c_array(matches) };
    SyaraStatus::SyaraOk
}

/// Scan a file at the given null-terminated path against rules that contain
/// phash patterns. Returns an empty array if no phash rules are defined.
///
/// On success, writes an allocated `SyaraMatchArray*` to `*out` and returns
/// `SYARA_OK`. The caller must free it with `syara_matches_free`.
///
/// # Safety
/// All pointer arguments must be non-null. `rules` must have been created by
/// `syara_compile_*` and not yet freed. `path` must be a valid null-terminated
/// C string.
#[no_mangle]
pub unsafe extern "C" fn syara_scan_file(
    rules: *const SyaraRules,
    path: *const c_char,
    out: *mut *mut SyaraMatchArray,
) -> SyaraStatus {
    if rules.is_null() || path.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return SyaraStatus::SyaraErrNullPtr;
    }
    let path_str = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(&format!("invalid UTF-8: {e}"));
            return SyaraStatus::SyaraErrUtf8;
        }
    };
    let matches = unsafe { (*rules).0.scan_file(Path::new(path_str)) };
    unsafe { *out = into_c_array(matches) };
    SyaraStatus::SyaraOk
}

/// Return the number of compiled rules in a `SyaraRules` handle.
///
/// Returns 0 if `rules` is null.
///
/// # Safety
/// `rules` must have been created by `syara_compile_*` and not yet freed, or
/// be null.
#[no_mangle]
pub unsafe extern "C" fn syara_rule_count(rules: *const SyaraRules) -> usize {
    if rules.is_null() {
        return 0;
    }
    unsafe { (*rules).0.rule_count() }
}

/// Free a `SyaraRules` handle created by `syara_compile_*`.
///
/// No-op if `rules` is null.
///
/// # Safety
/// `rules` must have been created by `syara_compile_*` and not yet freed, or
/// be null.
#[no_mangle]
pub unsafe extern "C" fn syara_rules_free(rules: *mut SyaraRules) {
    if !rules.is_null() {
        drop(unsafe { Box::from_raw(rules) });
    }
}

/// Free a `SyaraMatchArray` created by `syara_scan` or `syara_scan_file`.
///
/// No-op if `matches` is null.
///
/// # Safety
/// `matches` must have been created by `syara_scan*` and not yet freed, or be
/// null.
#[no_mangle]
pub unsafe extern "C" fn syara_matches_free(matches: *mut SyaraMatchArray) {
    if matches.is_null() {
        return;
    }
    unsafe {
        let array = Box::from_raw(matches);
        // Reclaim and drop each rule_name CString.
        let slice = std::slice::from_raw_parts(array.matches, array.count);
        for m in slice {
            if !m.rule_name.is_null() {
                drop(CString::from_raw(m.rule_name));
            }
        }
        // Reclaim and drop the matches Vec.
        drop(Vec::from_raw_parts(array.matches, array.count, array.count));
        // array (the Box) is dropped here.
    }
}

/// Return the last error message as a null-terminated C string.
///
/// The returned pointer is valid until the next syara_* call on this thread.
/// Never free this pointer.
#[no_mangle]
pub extern "C" fn syara_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    const RULE_SRC: &str = r#"
rule hit_rule {
    strings:
        $s1 = "hello"
    condition:
        $s1
}
rule miss_rule {
    strings:
        $s1 = "goodbye"
    condition:
        $s1
}
"#;

    fn compile(src: &str) -> *mut SyaraRules {
        let src_c = CString::new(src).unwrap();
        let mut ptr: *mut SyaraRules = std::ptr::null_mut();
        let status = unsafe { syara_compile_str(src_c.as_ptr(), &mut ptr) };
        assert!(matches!(status, SyaraStatus::SyaraOk), "compile failed");
        assert!(!ptr.is_null());
        ptr
    }

    #[test]
    fn compile_and_rule_count() {
        let rules = compile(RULE_SRC);
        let count = unsafe { syara_rule_count(rules) };
        assert_eq!(count, 2);
        unsafe { syara_rules_free(rules) };
    }

    #[test]
    fn scan_match_and_miss() {
        let rules = compile(RULE_SRC);
        let text = CString::new("hello world").unwrap();
        let mut out: *mut SyaraMatchArray = std::ptr::null_mut();

        let status = unsafe { syara_scan(rules, text.as_ptr(), &mut out) };
        assert!(matches!(status, SyaraStatus::SyaraOk));
        assert!(!out.is_null());

        unsafe {
            let array = &*out;
            assert_eq!(array.count, 2);

            let slice = std::slice::from_raw_parts(array.matches, array.count);
            let hit = slice.iter().find(|m| m.matched == 1).expect("expected a match");
            let name = CStr::from_ptr(hit.rule_name).to_str().unwrap();
            assert_eq!(name, "hit_rule");

            let miss = slice.iter().find(|m| m.matched == 0).expect("expected a miss");
            let miss_name = CStr::from_ptr(miss.rule_name).to_str().unwrap();
            assert_eq!(miss_name, "miss_rule");

            syara_matches_free(out);
            syara_rules_free(rules);
        }
    }

    #[test]
    fn scan_no_match() {
        let rules = compile(RULE_SRC);
        let text = CString::new("no keywords here").unwrap();
        let mut out: *mut SyaraMatchArray = std::ptr::null_mut();

        unsafe { syara_scan(rules, text.as_ptr(), &mut out) };
        unsafe {
            let array = &*out;
            assert_eq!(array.count, 2);
            let slice = std::slice::from_raw_parts(array.matches, array.count);
            assert!(slice.iter().all(|m| m.matched == 0));
            syara_matches_free(out);
            syara_rules_free(rules);
        }
    }

    #[test]
    fn null_ptr_returns_err_null_ptr() {
        let mut ptr: *mut SyaraRules = std::ptr::null_mut();
        let status = unsafe { syara_compile_str(std::ptr::null(), &mut ptr) };
        assert!(matches!(status, SyaraStatus::SyaraErrNullPtr));

        let error = unsafe { CStr::from_ptr(syara_last_error()).to_str().unwrap() };
        assert!(!error.is_empty());
    }

    #[test]
    fn last_error_on_bad_rules() {
        // Duplicate identifier triggers a compile error.
        let bad = CString::new(
            "rule dup {\n    strings:\n        $s1 = \"a\"\n        $s1 = \"b\"\n    condition:\n        $s1\n}",
        )
        .unwrap();
        let mut ptr: *mut SyaraRules = std::ptr::null_mut();
        let status = unsafe { syara_compile_str(bad.as_ptr(), &mut ptr) };
        assert!(matches!(status, SyaraStatus::SyaraErrCompile));
        assert!(ptr.is_null());

        let error = unsafe { CStr::from_ptr(syara_last_error()).to_str().unwrap() };
        assert!(!error.is_empty());
    }

    #[test]
    fn rule_count_null_returns_zero() {
        let count = unsafe { syara_rule_count(std::ptr::null()) };
        assert_eq!(count, 0);
    }

    #[test]
    fn rules_free_null_is_noop() {
        unsafe { syara_rules_free(std::ptr::null_mut()) }; // must not panic
    }

    #[test]
    fn matches_free_null_is_noop() {
        unsafe { syara_matches_free(std::ptr::null_mut()) }; // must not panic
    }

    #[test]
    fn compile_file_missing_path_returns_compile_err() {
        let path = CString::new("/nonexistent/path/rules.syara").unwrap();
        let mut ptr: *mut SyaraRules = std::ptr::null_mut();
        let status = unsafe { syara_compile_file(path.as_ptr(), &mut ptr) };
        assert!(matches!(status, SyaraStatus::SyaraErrCompile));
        assert!(ptr.is_null());
    }
}
