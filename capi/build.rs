fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_language(cbindgen::Language::C)
        .with_include_guard("SYARA_X_H")
        .with_documentation(true)
        .with_tab_width(4)
        .generate()
        .expect("Unable to generate C bindings")
        .write_to_file(format!("{crate_dir}/syara_x.h"));
}
