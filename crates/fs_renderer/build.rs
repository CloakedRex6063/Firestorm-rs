use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use walkdir::WalkDir;

fn embed_shaders_to_rs(output_path: &PathBuf, embed_path: &PathBuf) {
    let mut out_file = File::create(embed_path.join("shaders.rs")).unwrap();

    for entry in fs::read_dir(output_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.extension().map_or(false, |ext| ext == "cso") {
            let bytes = fs::read(&path).unwrap();
            let var_name = path
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .replace('.', "_");

            write!(out_file, "pub static {}: &[u8] = &[\n    ", var_name).unwrap();

            for (i, byte) in bytes.iter().enumerate() {
                write!(out_file, "0x{:02X}, ", byte).unwrap();
                if (i + 1) % 16 == 0 {
                    write!(out_file, "\n    ").unwrap();
                }
            }

            writeln!(out_file, "\n];\n").unwrap();

            fs::remove_file(&path).unwrap();
        }
    }
    println!(
        "cargo:warning=Embedded shaders to {}",
        output_path.join("shaders.rs").display()
    );
}

fn gen_shaders(shader_path: &PathBuf, output_path: &PathBuf, dxc_path: &PathBuf) {
    for file in WalkDir::new(shader_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && e.path().extension().unwrap_or_default() == "hlsl")
    {
        let file = file.path();
        let stem = file.file_stem().unwrap().to_string_lossy();

        if stem.len() < 3 {
            continue;
        }

        let target = match &stem[stem.len() - 2..] {
            "ps" => "ps_6_6",
            "cs" => "cs_6_6",
            "vs" => "vs_6_6",
            "ms" => "ms_6_6",
            "as" => "as_6_6",
            _ => {
                println!(
                    "cargo:warning=Unrecognised shader type for file {}",
                    file.display()
                );
                continue;
            }
        };

        let output_file = output_path.join(format!("{}.cso", stem));
        println!(
            "cargo:warning=Compiling shader {} to {}",
            file.display(),
            output_file.display()
        );

        let status = Command::new(&dxc_path)
            .args(&[
                "-T",
                target,
                "-E",
                "main",
                "-Fo",
                output_file.to_str().unwrap(),
                "-Zi",
                "-Od",
                "-Qembed_debug",
                file.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to execute dxc");

        if !status.success() {
            panic!("Failed to compile shader {}", file.display());
        }
    }
}
fn main() {
    let shader_dir = PathBuf::from("shaders");
    let dxc_path = PathBuf::from("external/dxil/bin/dxc.exe");
    let out_path = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let embed_path = PathBuf::from("src");

    gen_shaders(&shader_dir, &out_path, &dxc_path);
    embed_shaders_to_rs(&out_path, &embed_path);

    println!("cargo:rerun-if-changed=shaders/");
}
