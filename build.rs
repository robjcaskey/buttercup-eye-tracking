use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/sam31_cuda_stream.cpp");
    for variable in ["LIBTORCH", "LIBTORCH_CXX11_ABI", "CUDA_PATH"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    if env::var_os("CARGO_FEATURE_SAM31").is_none() {
        return;
    }
    let torch = PathBuf::from(env::var_os("LIBTORCH").expect("SAM31 requires LIBTORCH"));
    let cuda = PathBuf::from(env::var_os("CUDA_PATH").unwrap_or_else(|| "/usr/local/cuda".into()));
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/sam31_cuda_stream.cpp")
        .include(torch.join("include"))
        .include(cuda.join("include"))
        .define("_GLIBCXX_USE_CXX11_ABI", env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".into()).as_str())
        .compile("buttercup_cuda_stream");
    println!("cargo:rustc-link-search=native={}", torch.join("lib").display());
    println!("cargo:rustc-link-lib=c10_cuda");
}
