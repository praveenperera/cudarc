//! cuBLAS and cuBLASLt open by soname on Linux, so a process never holds two copies of
//! cuBLAS or a cuBLAS paired with another install's cuBLASLt
//!
//! Stub libraries built with `cc` stand in for two installs: a toolkit's, which has the
//! unversioned dev symlinks, and a pip wheel's, which has only the sonames. Each stub
//! reports which install it came from, and the stub cuBLAS also reports the install of
//! the cuBLASLt it is bound to. Each scenario runs in a child process, since the dynamic
//! loader reads `LD_LIBRARY_PATH` only at startup.
#![cfg(all(
    target_os = "linux",
    feature = "dynamic-loading",
    feature = "cublas",
    feature = "cublaslt"
))]

use std::ffi::{c_char, CStr};
use std::path::{Path, PathBuf};
use std::process::Command;

const SCENARIO: &str = "CUDARC_SONAME_SCENARIO";
const DIR: &str = "CUDARC_SONAME_DIR";

/// Compiles `source` into `dir/file` with the given soname
fn cc(dir: &Path, source: &str, file: &str, soname: &str, extra: &[&str]) {
    let source_file = dir.join(format!("{file}.c"));
    std::fs::write(&source_file, source).unwrap();
    let status = Command::new("cc")
        .args(["-shared", "-fPIC", "-o"])
        .arg(dir.join(file))
        .arg(format!("-Wl,-soname,{soname}"))
        .arg(&source_file)
        .args(extra)
        .status()
        .expect("building the stub libraries needs a C compiler on PATH as `cc`");
    assert!(status.success(), "cc failed for {file}");
}

/// The first soname this build tries for `lib`, such as `libcublas.so.12` for CUDA 12.
/// The stubs use it so that they, not a real install in the loader cache, are found
fn soname(lib: &str) -> String {
    let prefix = format!("lib{lib}.so.");
    cudarc::get_lib_name_candidates(lib)
        .into_iter()
        .find(|name| name.starts_with(&prefix))
        .unwrap()
}

/// Builds one install's stub cuBLASLt and cuBLAS under this build's sonames, with the
/// dev symlinks when `dev_symlinks` is set
fn install(dir: &Path, origin: &str, dev_symlinks: bool) {
    let (blas_soname, lt_soname) = (soname("cublas"), soname("cublasLt"));
    std::fs::create_dir_all(dir).unwrap();
    let lt = format!("const char *stub_lt_origin(void) {{ return \"{origin}\"; }}\n");
    cc(dir, &lt, &lt_soname, &lt_soname, &[]);
    let blas = format!(
        "const char *stub_lt_origin(void);\n\
         const char *stub_blas_origin(void) {{ return \"{origin}\"; }}\n\
         const char *stub_paired_lt_origin(void) {{ return stub_lt_origin(); }}\n"
    );
    let lib_dir = format!("-L{}", dir.display());
    let link_lt = [lib_dir, format!("-l:{lt_soname}")];
    let link_lt: Vec<&str> = link_lt.iter().map(String::as_str).collect();
    cc(dir, &blas, &blas_soname, &blas_soname, &link_lt);

    if dev_symlinks {
        for (name, soname) in [
            ("libcublas.so", &blas_soname),
            ("libcublasLt.so", &lt_soname),
        ] {
            std::os::unix::fs::symlink(soname, dir.join(name)).unwrap();
        }
    }
}

/// Runs `scenario` in a child copy of this test binary
fn run(scenario: &str, library_path: &[&Path], dir: &Path) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child", "--nocapture", "--test-threads=1"])
        .env(SCENARIO, scenario)
        .env(DIR, dir)
        .env(
            "LD_LIBRARY_PATH",
            std::env::join_paths(library_path).unwrap(),
        )
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "{scenario} failed:\n{stdout}{stderr}"
    );
    // a renamed child would run no test and still exit cleanly
    assert!(
        stdout.contains("1 passed"),
        "{scenario} ran no child:\n{stdout}"
    );
}

#[test]
fn cublas_opens_by_soname() {
    let root = std::env::temp_dir().join(format!("cudarc-sonames-{}", std::process::id()));
    let (toolkit, wheel) = (root.join("toolkit"), root.join("wheel"));
    install(&toolkit, "toolkit", true);
    install(&wheel, "wheel", false);

    // a host, such as Python, loaded the wheel's pair by full path, and the toolkit is
    // on the loader's search path
    run("preloaded", &[&toolkit], &wheel);
    // the wheel's directory comes first on the search path, and only the toolkit has
    // the unversioned names
    run("search-path", &[&wheel, &toolkit], &wheel);

    std::fs::remove_dir_all(&root).unwrap();
}

/// The child side of the scenarios above; does nothing outside one
#[test]
fn child() {
    let Ok(scenario) = std::env::var(SCENARIO) else {
        return;
    };
    let dir = PathBuf::from(std::env::var_os(DIR).unwrap());

    let _preloaded = (scenario == "preloaded").then(|| unsafe {
        [
            libloading::Library::new(dir.join(soname("cublasLt"))).unwrap(),
            libloading::Library::new(dir.join(soname("cublas"))).unwrap(),
        ]
    });

    let blas = unsafe { cudarc::cublas::sys::culib() };
    let lt = unsafe { cudarc::cublaslt::sys::culib() };
    let origins = [
        origin(blas, b"stub_blas_origin\0"),
        origin(blas, b"stub_paired_lt_origin\0"),
        origin(lt, b"stub_lt_origin\0"),
    ];
    // cudarc's cuBLAS, the cuBLASLt it binds, and cudarc's cuBLASLt
    assert_eq!(origins, ["wheel", "wheel", "wheel"], "{scenario}");
}

/// The install a stub reports through `symbol`
fn origin(lib: &libloading::Library, symbol: &[u8]) -> String {
    unsafe {
        let get = lib
            .get::<unsafe extern "C" fn() -> *const c_char>(symbol)
            .unwrap();
        CStr::from_ptr(get()).to_str().unwrap().to_owned()
    }
}
