//! Finding and opening the CUDA libraries at runtime, for the `dynamic-loading` feature
//!
//! Every `sys` module opens its library through [`open_culib`], which tries
//! [`get_lib_name_candidates`] in order and keeps the first library that opens for the
//! rest of the process. `is_culib_present` and `culib` share that one handle, so once a
//! library has loaded no later call searches for it again.
//!
//! On Linux the names are versioned sonames, newest first after the CUDA major the
//! bindings were built for, with the unversioned name last. The unversioned name is a
//! toolkit's dev symlink: opening it can load a second copy of a library the process
//! already holds under its soname, or pair a toolkit's cuBLAS with another install's
//! cuBLASLt, which its soname import then binds.
//!
//! On Windows, the default DLL search looks in the current directory before `PATH`, so a
//! DLL planted in the folder a program runs from would load and run its code. The driver
//! therefore loads only from System32, and every other library goes through
//! [`windows`]'s search, which never looks in the current directory or a relative `PATH`
//! entry.

use std::string::String;
use std::sync::OnceLock;
use std::vec::Vec;

use libloading::Library;

/// Opens a `sys` module's library, or returns the one it already opened
///
/// A library that opens is kept in `cell` for the process, so neither `culib` nor
/// `is_culib_present` searches again once one has loaded. A failed search is not
/// remembered: the next call searches again, since a library may have been installed or
/// the search path changed in between.
///
/// Returns the candidate names that were tried when none opens.
///
/// # Safety
/// Loading a library runs its initialisation routines, as with [`Library::new`].
pub(crate) unsafe fn open_culib(
    cell: &'static OnceLock<Library>,
    lib_names: &[&str],
    windows_system32_only: bool,
) -> Result<&'static Library, Vec<String>> {
    if let Some(lib) = cell.get() {
        return Ok(lib);
    }

    let choices: Vec<String> = lib_names
        .iter()
        .flat_map(|name| get_lib_name_candidates(name))
        .collect();
    match open_first(&choices, windows_system32_only) {
        // a racing call may have stored its own handle first; this one is then dropped,
        // and both refer to the same loaded module
        Some(lib) => Ok(cell.get_or_init(|| lib)),
        None => Err(choices),
    }
}

/// Opens the first of `choices` that loads, through the dynamic loader's own search
#[cfg(not(windows))]
unsafe fn open_first(choices: &[String], windows_system32_only: bool) -> Option<Library> {
    let _ = windows_system32_only;
    choices.iter().find_map(|choice| Library::new(choice).ok())
}

/// Opens the first of `choices` that loads, never from the current directory
///
/// With `windows_system32_only`, each name is loaded by its full path in System32 alone.
/// The CUDA driver (`nvcuda.dll`) is always installed there, and its first candidate
/// name, `cuda.dll`, never is, so the default search would reach the current directory
/// before the real driver. A full path also keeps Windows from handing back a module
/// loaded from elsewhere under the same name, as a bare-name load would even with
/// `LOAD_LIBRARY_SEARCH_SYSTEM32`. Every other library uses [`windows::open`].
#[cfg(windows)]
unsafe fn open_first(choices: &[String], windows_system32_only: bool) -> Option<Library> {
    use libloading::os::windows::{Library as WindowsLibrary, LOAD_LIBRARY_SEARCH_SYSTEM32};

    if windows_system32_only {
        let system32 = windows::system32()?;
        return choices
            .iter()
            .map(|choice| system32.join(choice))
            .filter(|file| file.is_file())
            .find_map(|file| {
                WindowsLibrary::load_with_flags(&file, LOAD_LIBRARY_SEARCH_SYSTEM32).ok()
            })
            .map(Into::into);
    }

    let path = std::env::var_os("PATH").unwrap_or_default();
    let cuda_path = std::env::var_os("CUDA_PATH");
    let dirs = windows::search_dirs(&path, cuda_path.as_deref());
    choices
        .iter()
        .find_map(|choice| windows::open(choice, &dirs))
        .map(Into::into)
}

/// The names to try for `lib_name`, in order
///
/// Versioned names come first, and the unversioned names come last. The first names are
/// the version this build's toolkit ships, which is not always the CUDA major: CUDA 12's
/// cuFFT is `libcufft.so.11` and CUDA 13's is `libcufft.so.12`. Every other major
/// follows, newest first. On Linux these are sonames such as `libcublas.so.13`; on
/// Windows, DLL names such as `cublas64_13.dll` and `nvrtc64_130_0.dll`. The list covers
/// both platforms' patterns, and names the platform doesn't use simply never open.
pub fn get_lib_name_candidates(lib_name: &str) -> Vec<String> {
    use std::env::consts::{DLL_PREFIX, DLL_SUFFIX};

    let pointer_width = if cfg!(target_pointer_width = "32") {
        "32"
    } else if cfg!(target_pointer_width = "64") {
        "64"
    } else {
        panic!("Unsupported target pointer width")
    };

    let cudnn_major = if cfg!(any(feature = "cudnn-09010", feature = "cudnn-09021")) {
        "9"
    } else if cfg!(feature = "cudnn-08009") {
        "8"
    } else {
        "9"
    };

    Names {
        prefix: DLL_PREFIX,
        suffix: DLL_SUFFIX,
        pointer_width,
        major: env!("CUDA_MAJOR_VERSION"),
        minor: env!("CUDA_MINOR_VERSION"),
        cudnn_major,
    }
    .candidates(lib_name)
}

/// The majors tried after the build's own version, newest first. CUDA libraries version
/// their sonames on their own schedule (cuDNN 9, cuRAND 10, NCCL 2, the driver's `.so.1`),
/// so this covers those as well as CUDA's own majors
const OTHER_MAJORS: [&str; 7] = ["13", "12", "11", "10", "9", "2", "1"];

/// One platform's naming scheme for a CUDA build, kept apart from the compile target so
/// tests can check both platforms' names from either
struct Names<'a> {
    prefix: &'a str,
    suffix: &'a str,
    pointer_width: &'a str,
    major: &'a str,
    minor: &'a str,
    cudnn_major: &'a str,
}

impl Names<'_> {
    /// The soname version of `lib_name` in this build's CUDA toolkit, such as `11` for
    /// CUDA 12's cuFFT or `11.0` for CUDA 11's runtime
    fn own_version(&self, lib_name: &str) -> String {
        let Self { major, minor, .. } = *self;
        let fixed = match (lib_name, major) {
            ("cuda" | "nvcuda" | "nvToolsExt", _) => "1",
            ("cufile", _) => "0",
            ("nccl" | "cutensor", _) => "2",
            ("curand", _) => "10",
            ("cudnn", _) => self.cudnn_major,
            ("cufft", "11") => "10",
            ("cufft", "12") => "11",
            ("cufft", "13") => "12",
            ("cusolver" | "cusolverMg" | "cusparse", "11") => "11",
            ("cusolver" | "cusolverMg", "12") => "11",
            ("cusolver" | "cusolverMg" | "cusparse", "13") => "12",
            ("cudart", "11") => "11.0",
            ("nvrtc", "11") => "11.2",
            ("cupti", "11") => return std::format!("11.{minor}"),
            _ => major,
        };
        fixed.into()
    }

    fn candidates(&self, lib_name: &str) -> Vec<String> {
        let Self {
            prefix: p,
            suffix: s,
            pointer_width: w,
            major,
            minor,
            ..
        } = *self;

        let own = self.own_version(lib_name);
        // Windows drops the dot, as in cudart64_110.dll and nvrtc64_112_0.dll
        let own_dll = own.replace('.', "");
        let mut names = std::vec![
            std::format!("{p}{lib_name}{s}.{own}"),
            std::format!("{p}{lib_name}{w}_{own_dll}{s}"),
            std::format!("{p}{lib_name}{w}_{own_dll}_0{s}"),
            std::format!("{p}{lib_name}{w}_{major}{minor}{s}"),
            std::format!("{p}{lib_name}{w}_{major}{minor}_0{s}"),
            std::format!("{p}{lib_name}{w}_{major}0_{minor}{s}"),
            // See issue #246
            std::format!("{p}{lib_name}{w}_{major}0_0{s}"),
        ];
        // See issues #242, #260, #274 and #296
        for other in OTHER_MAJORS {
            names.extend([
                std::format!("{p}{lib_name}{s}.{other}"),
                std::format!("{p}{lib_name}{w}_{other}{s}"),
                // NVRTC's DLL, such as nvrtc64_130_0.dll
                std::format!("{p}{lib_name}{w}_{other}0_0{s}"),
            ]);
        }
        names.extend([
            std::format!("{p}{lib_name}{s}"),
            std::format!("{p}{lib_name}{w}{s}"),
        ]);

        // the own version usually repeats a later pattern; the first copy keeps its place
        let mut unique: Vec<String> = Vec::with_capacity(names.len());
        for name in names {
            if !unique.contains(&name) {
                unique.push(name);
            }
        }
        unique
    }
}

/// The DLL search for every library but the driver
///
/// The search never looks in the current directory or a relative `PATH` entry, and
/// every load restricts where Windows may look, so no step falls back to the default
/// search. For each name in turn it:
///
/// 1. reuses a module this process already loaded under that name, wherever it came
///    from, as Windows' own bare-name load would;
/// 2. loads the bare name from the executable's directory, System32, or a directory the
///    host added with `AddDllDirectory` (`LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`);
/// 3. loads it by full path from each absolute `PATH` entry, where CUDA toolkits put
///    themselves, and then from `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin` (CUDA 13
///    keeps its DLLs in `bin\x64`). These loads add `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`, so
///    a DLL's own imports come from its directory: `cublas64_13.dll` binds the
///    `cublasLt64_13.dll` beside it rather than another install's copy found earlier.
///
/// A file that is found but fails to load, such as one for another architecture or one
/// missing an import, leaves the later directories and names to try.
#[cfg(any(windows, test))]
pub(crate) mod windows {
    use std::ffi::OsStr;
    use std::path::PathBuf;
    use std::vec::Vec;

    /// The directories searched by full path, in order: each absolute `PATH` entry, then
    /// `%CUDA_PATH%\bin\x64` and `%CUDA_PATH%\bin` when `CUDA_PATH` is absolute
    ///
    /// Relative and empty entries resolve against the current directory, so they are
    /// skipped. A current directory that is also an absolute entry is searched as that
    /// entry.
    pub(crate) fn search_dirs(path: &OsStr, cuda_path: Option<&OsStr>) -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = std::env::split_paths(path)
            .filter(|dir| dir.is_absolute())
            .collect();
        if let Some(root) = cuda_path
            .map(PathBuf::from)
            .filter(|root| root.is_absolute())
        {
            let bin = root.join("bin");
            dirs.extend([bin.join("x64"), bin]);
        }
        dirs
    }

    /// The System32 directory, from `GetSystemDirectoryW` rather than the environment
    #[cfg(windows)]
    pub(crate) fn system32() -> Option<PathBuf> {
        use std::ffi::OsString;
        use std::os::windows::ffi::OsStringExt;

        #[link(name = "kernel32")]
        extern "system" {
            fn GetSystemDirectoryW(buffer: *mut u16, size: u32) -> u32;
        }

        // the system directory is never longer than MAX_PATH
        let mut buffer = [0u16; 260];
        let len = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), buffer.len() as u32) };
        let len = usize::try_from(len)
            .ok()
            .filter(|len| (1..buffer.len()).contains(len))?;
        Some(OsString::from_wide(&buffer[..len]).into())
    }

    /// Opens `name` by the steps in the module docs, or returns `None`
    ///
    /// # Safety
    /// Loading a library runs its initialisation routines.
    #[cfg(windows)]
    pub(crate) unsafe fn open(
        name: &str,
        dirs: &[PathBuf],
    ) -> Option<libloading::os::windows::Library> {
        use libloading::os::windows::{
            Library, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
        };

        if let Ok(loaded) = Library::open_already_loaded(name) {
            return Some(loaded);
        }

        if let Ok(lib) = Library::load_with_flags(name, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) {
            return Some(lib);
        }

        let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        dirs.iter()
            .map(|dir| dir.join(name))
            .filter(|file| file.is_file())
            .find_map(|file| Library::load_with_flags(&file, flags).ok())
    }
}

#[cfg(test)]
mod tests {
    use std::string::String;
    use std::vec::Vec;

    use super::Names;

    fn linux(major: &'static str, minor: &'static str) -> Names<'static> {
        Names {
            prefix: "lib",
            suffix: ".so",
            pointer_width: "64",
            major,
            minor,
            cudnn_major: "9",
        }
    }

    fn windows(major: &'static str, minor: &'static str) -> Names<'static> {
        Names {
            prefix: "",
            suffix: ".dll",
            pointer_width: "64",
            major,
            minor,
            cudnn_major: "9",
        }
    }

    fn position(names: &[String], name: &str) -> usize {
        names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap_or_else(|| panic!("{name} is not a candidate: {names:?}"))
    }

    /// The build's own soname comes first, then every other major newest first, and the
    /// unversioned dev symlink last, so a library the process already holds under its
    /// soname is the one reused, and a host with two toolkits loads this build's
    #[test]
    fn linux_sonames_come_first_starting_with_the_builds_own() {
        // sonames in NVIDIA's CUDA 12 and 13 libraries, and cuDNN 9's
        let cases = [
            ("12", "cublas", "libcublas.so.12"),
            ("13", "cublas", "libcublas.so.13"),
            ("12", "cufft", "libcufft.so.11"),
            ("13", "cufft", "libcufft.so.12"),
            ("12", "cusolver", "libcusolver.so.11"),
            ("13", "cusolver", "libcusolver.so.12"),
            ("13", "cusparse", "libcusparse.so.12"),
            ("13", "curand", "libcurand.so.10"),
            ("13", "cufile", "libcufile.so.0"),
            ("12", "cudnn", "libcudnn.so.9"),
            ("12", "nccl", "libnccl.so.2"),
            ("12", "nvToolsExt", "libnvToolsExt.so.1"),
            ("12", "cuda", "libcuda.so.1"),
        ];
        for (major, lib, own) in cases {
            let names = linux(major, "0").candidates(lib);
            let sonames: Vec<&String> = names.iter().filter(|name| name.contains(".so.")).collect();
            assert_eq!(sonames[0], own, "{names:?}");

            // after its own, a library's other majors come newest first
            let others: Vec<&str> = sonames[1..]
                .iter()
                .filter_map(|name| name.rsplit_once(".so.").map(|(_, version)| version))
                .collect();
            let expected: Vec<&str> = super::OTHER_MAJORS
                .into_iter()
                .filter(|version| !own.ends_with(&std::format!(".so.{version}")))
                .collect();
            assert_eq!(others, expected, "{lib} for CUDA {major}");

            let unversioned = std::format!("lib{lib}.so");
            assert!(position(&names, sonames.last().unwrap()) < position(&names, &unversioned));
        }
    }

    /// The DLL names in NVIDIA's CUDA 12.4, CUDA 13.0 and cuDNN 9 redistributables are
    /// candidates for both CUDA 12 and CUDA 13 builds, with the build's own major first
    #[test]
    fn windows_names_cover_cuda_12_and_13() {
        let cases = [
            ("cublas", "cublas64_12.dll", "cublas64_13.dll"),
            ("cublasLt", "cublasLt64_12.dll", "cublasLt64_13.dll"),
            ("nvrtc", "nvrtc64_120_0.dll", "nvrtc64_130_0.dll"),
        ];
        for (lib, cuda12, cuda13) in cases {
            let names = windows("12", "4").candidates(lib);
            assert!(position(&names, cuda12) < position(&names, cuda13));
            let names = windows("13", "0").candidates(lib);
            assert!(position(&names, cuda13) < position(&names, cuda12));
        }

        // cuFFT numbers its DLLs on its own: CUDA 12's is cufft64_11.dll
        let names = windows("12", "4").candidates("cufft");
        assert!(position(&names, "cufft64_11.dll") < position(&names, "cufft64_12.dll"));
        let names = windows("13", "0").candidates("cufft");
        assert!(position(&names, "cufft64_12.dll") < position(&names, "cufft64_11.dll"));

        for (major, minor) in [("12", "4"), ("13", "0")] {
            let names = windows(major, minor).candidates("cudnn");
            assert!(position(&names, "cudnn64_9.dll") < position(&names, "cudnn.dll"));
        }
    }

    /// Relative and empty `PATH` entries resolve against the current directory, so the
    /// full-path search skips them, and CUDA 13's `bin\x64` is searched before `bin`
    #[test]
    fn windows_search_dirs_skip_relative_entries() {
        let root = std::env::temp_dir();
        let path = std::env::join_paths([
            root.join("first"),
            "relative".into(),
            "".into(),
            root.join("second"),
        ])
        .unwrap();
        let cuda = root.join("cuda");

        let dirs = super::windows::search_dirs(&path, Some(cuda.as_os_str()));
        let expected = std::vec![
            root.join("first"),
            root.join("second"),
            cuda.join("bin").join("x64"),
            cuda.join("bin"),
        ];
        assert_eq!(dirs, expected);

        let relative_cuda = super::windows::search_dirs(&path, Some("cuda".as_ref()));
        assert_eq!(relative_cuda, expected[..2]);
    }

    /// Once a library has loaded, [`super::open_culib`] returns it rather than open a
    /// name earlier in the list that appeared since, as `is_culib_present` used to
    ///
    /// Stubs built with `cc` use a name no real library has. The earlier name's stub
    /// leaves a marker file when it loads, which shows whether a second search ran. The
    /// test runs itself again as a child with the stubs' directory on `LD_LIBRARY_PATH`,
    /// which the dynamic loader reads only at startup.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_loaded_library_is_not_searched_for_again() {
        use std::path::{Path, PathBuf};
        use std::process::Command;
        use std::sync::OnceLock;

        const DIR: &str = "CUDARC_CACHE_TEST_DIR";
        const NAME: &str = "cudarc_cache_probe";

        if let Some(dir) = std::env::var_os(DIR) {
            let dir = PathBuf::from(dir);
            let marker = dir.join("probed");
            static CELL: OnceLock<libloading::Library> = OnceLock::new();
            let first = unsafe { super::open_culib(&CELL, &[NAME], false) }.unwrap();

            let earliest = &super::get_lib_name_candidates(NAME)[0];
            std::fs::copy(dir.join("marker.so"), dir.join(earliest)).unwrap();
            let again = unsafe { super::open_culib(&CELL, &[NAME], false) }.unwrap();
            assert!(core::ptr::eq(first, again));
            assert!(
                !marker.exists(),
                "searched again after the library had loaded"
            );

            // control: a second search would have found and loaded the marker
            drop(unsafe { libloading::Library::new(earliest) }.unwrap());
            assert!(marker.exists());
            return;
        }

        let dir = std::env::temp_dir().join(std::format!("cudarc-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cc = |file: &str, source: &str| {
            let source_file: &Path = &dir.join(std::format!("{file}.c"));
            std::fs::write(source_file, source).unwrap();
            let status = Command::new("cc")
                .args(["-shared", "-fPIC", "-o"])
                .arg(dir.join(file))
                .arg(source_file)
                .status()
                .expect("building the stub libraries needs a C compiler on PATH as `cc`");
            assert!(status.success(), "cc failed for {file}");
        };
        // the unversioned name, which comes last
        cc(&std::format!("lib{NAME}.so"), "int stub;\n");
        cc(
            "marker.so",
            "#include <stdio.h>\n\
             __attribute__((constructor)) static void mark(void) {\n\
                 FILE *file = fopen(\"probed\", \"w\");\n\
                 if (file) fclose(file);\n\
             }\n",
        );

        let test = "loading::tests::a_loaded_library_is_not_searched_for_again";
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .current_dir(&dir)
            .env(DIR, &dir)
            .env("LD_LIBRARY_PATH", &dir)
            .output()
            .unwrap();
        let stdout = std::string::String::from_utf8_lossy(&output.stdout);
        let stderr = std::string::String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stdout}{stderr}");
        assert!(
            stdout.contains("1 passed"),
            "the child ran no test:\n{stdout}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Copies of a system DLL stand in for CUDA's, under names no other directory has.
    /// This is the only test that changes the working directory or `PATH`, so it runs
    /// every case in turn and puts both back when it ends
    #[cfg(windows)]
    #[test]
    fn windows_search_never_loads_from_the_current_directory() {
        use std::env;
        use std::ffi::OsString;
        use std::fs;
        use std::os::windows::ffi::OsStringExt;
        use std::path::{Path, PathBuf};

        use libloading::os::windows::{Library as WindowsLibrary, LOAD_LIBRARY_SEARCH_SYSTEM32};

        /// Puts the working directory and environment back even when an assertion fails
        struct Restore(PathBuf, Option<OsString>, Option<OsString>);

        impl Drop for Restore {
            fn drop(&mut self) {
                env::set_current_dir(&self.0).unwrap();
                for (var, value) in [("PATH", &self.1), ("CUDA_PATH", &self.2)] {
                    match value {
                        Some(value) => env::set_var(var, value),
                        None => env::remove_var(var),
                    }
                }
            }
        }

        let pid = std::process::id();
        let root = env::temp_dir().join(std::format!("cudarc-dll-search-{pid}"));
        let dir = |name: &str| {
            let dir = root.join(name);
            fs::create_dir_all(&dir).unwrap();
            dir
        };
        let (cwd, relative, absolute) = (dir("cwd"), dir("cwd/relative"), dir("absolute"));
        let (broken, cuda_bin) = (dir("broken"), dir("cuda/bin/x64"));
        let system = PathBuf::from(env::var_os("SystemRoot").unwrap()).join("System32");
        let place = |dir: &Path, name: &str| {
            fs::copy(system.join("version.dll"), dir.join(name)).unwrap();
        };
        let probe = |tag: &str| std::format!("cudarc_probe_{pid}_{tag}.dll");
        let opens = |name: &str| unsafe { super::open_first(&[name.into()], false) }.is_some();

        let _restore = Restore(
            env::current_dir().unwrap(),
            env::var_os("PATH"),
            env::var_os("CUDA_PATH"),
        );
        env::set_current_dir(&cwd).unwrap();
        let path = env::join_paths([PathBuf::from("relative"), absolute.clone(), broken.clone()]);
        env::set_var("PATH", path.unwrap());
        env::set_var("CUDA_PATH", root.join("cuda"));

        let planted = probe("planted");
        place(&cwd, &planted);
        assert!(!opens(&planted), "loaded from the current directory");

        let behind_relative = probe("relative");
        place(&relative, &behind_relative);
        assert!(
            !opens(&behind_relative),
            "loaded through a relative PATH entry"
        );

        let on_path = probe("absolute");
        place(&absolute, &on_path);
        assert!(opens(&on_path), "missed an absolute PATH entry");

        // the copy on PATH fails to load, and the default search's retry would take the
        // one in the current directory
        let unloadable = probe("broken");
        fs::write(broken.join(&unloadable), b"not a DLL").unwrap();
        place(&cwd, &unloadable);
        assert!(!opens(&unloadable), "fell back to the default search");

        let cuda_13 = probe("cuda13");
        place(&cuda_bin, &cuda_13);
        assert!(opens(&cuda_13), r"missed %CUDA_PATH%\bin\x64");

        // a module already loaded under the name is reused, as a bare-name load would
        let preloaded = probe("preloaded");
        place(&absolute, &preloaded);
        let held = unsafe { libloading::Library::new(absolute.join(&preloaded)) }.unwrap();
        env::set_var("PATH", "");
        assert!(opens(&preloaded), "did not reuse a loaded module");
        drop(held);
        env::set_var("PATH", &absolute);

        // the driver loads from System32 alone, even when a `cuda.dll` sits on PATH or a
        // module of that name is already loaded from elsewhere
        let system32 = super::windows::system32().unwrap();
        // Windows paths compare without case, and GetSystemDirectoryW often answers in
        // a different case from %SystemRoot%
        let lower = |path: &Path| path.to_string_lossy().to_lowercase();
        assert_eq!(lower(&system32), lower(&system));
        if !system.join("cuda.dll").exists() {
            place(&cwd, "cuda.dll");
            place(&absolute, "cuda.dll");
            let driver = unsafe { super::open_first(&["cuda.dll".into()], true) };
            assert!(driver.is_none(), "loaded the driver from outside System32");

            let held = unsafe { libloading::Library::new(absolute.join("cuda.dll")) }.unwrap();
            let driver = unsafe { super::open_first(&["cuda.dll".into()], true) };
            assert!(
                driver.is_none(),
                "reused a driver name loaded from outside System32"
            );
            drop(held);
        }

        // the file a module was loaded from, handing the module back so it stays loaded
        let module_path = |lib: WindowsLibrary| -> (PathBuf, WindowsLibrary) {
            #[link(name = "kernel32")]
            extern "system" {
                fn GetModuleFileNameW(module: isize, name: *mut u16, size: u32) -> u32;
            }

            let handle = lib.into_raw();
            let mut buffer = [0u16; 1024];
            let len = unsafe { GetModuleFileNameW(handle, buffer.as_mut_ptr(), 1024) };
            let path = OsString::from_wide(&buffer[..len as usize]).into();
            (path, unsafe { WindowsLibrary::from_raw(handle) })
        };
        let path_of = |lib: WindowsLibrary| lower(&module_path(lib).0);

        // the driver path does load a System32 DLL, and loads that file even while a
        // copy of the same name from another directory is loaded. The name is a System32
        // DLL this process hasn't loaded, so the copy loaded next is the only module of
        // that name and a bare-name load would return it
        let name = ["version.dll", "msimg32.dll", "wtsapi32.dll", "dbghelp.dll"]
            .into_iter()
            .find(|name| {
                system.join(name).is_file() && WindowsLibrary::open_already_loaded(*name).is_err()
            })
            .expect("every candidate System32 DLL is already loaded");
        fs::copy(system.join(name), absolute.join(name)).unwrap();
        let other_copy = absolute.join(name);
        let held = unsafe { libloading::Library::new(&other_copy) }.unwrap();
        let (held_path, held) = module_path(held.into());
        assert_eq!(lower(&held_path), lower(&other_copy));

        // control: a bare name, even restricted to System32, returns the copy from
        // elsewhere, so the check below catches a driver load that regressed to one
        let bare = unsafe { WindowsLibrary::load_with_flags(name, LOAD_LIBRARY_SEARCH_SYSTEM32) };
        assert_eq!(path_of(bare.unwrap()), lower(&other_copy));

        let driver = unsafe { super::open_first(&[name.into()], true) };
        let driver = driver.expect("did not load a DLL from System32");
        assert_eq!(path_of(driver.into()), lower(&system.join(name)));
        drop(held);

        // control: the default search does load a DLL from the current directory, so the
        // refusals above came from the search, not the setup
        let control = probe("control");
        place(&cwd, &control);
        assert!(unsafe { libloading::Library::new(&control) }.is_ok());
    }
}
