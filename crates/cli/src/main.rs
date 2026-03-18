mod app;
mod cli;

use clap::Parser;

#[tokio::main]
async fn main() {
    let cli = cli::Cli::parse();
    if let Err(e) = app::run(cli).await {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
