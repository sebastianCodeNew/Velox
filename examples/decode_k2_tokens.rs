// Decode K2-Horizon token IDs to text using the Velox native BPE tokenizer.
// Usage: cargo run --release --example decode_k2_tokens -- <id,id,...> [k2_tokenizer.json]

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: decode_k2_tokens <comma,separated,ids> [tokenizer.json]");
        std::process::exit(2);
    }
    let tok_path = if args.len() > 2 {
        args[2].clone()
    } else if Path::new("k2_tokenizer.json").exists() {
        "k2_tokenizer.json".to_string()
    } else {
        "tokenizer.json".to_string()
    };

    let tokens: Vec<u32> = args[1]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .filter_map(|s| s.trim().parse::<u32>().ok())
        .collect();

    let tokenizer = veloxc::Tokenizer::from_file(&tok_path)
        .unwrap_or_else(|e| {
            eprintln!("failed to load tokenizer '{tok_path}': {e}");
            std::process::exit(1)
        });

    println!("{}", tokenizer.decode(&tokens));
}
