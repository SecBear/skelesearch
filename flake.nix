{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      crane,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        # crane v2: mkLib replaced the old crane.lib.${system} attribute
        craneLib = crane.mkLib pkgs;
        src = craneLib.cleanCargoSource ./.;

        # cmake and pkg-config are needed at build time for CozoDB/RocksDB.
        commonArgs = {
          inherit src;
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.pkg-config
          ];
        };
      in
      {
        packages = {
          # Release packages always compile against RocksDB for production
          # durability; SQLite remains the dev-shell default.
          skelesearch-cli = craneLib.buildPackage (
            commonArgs
            // {
              pname = "skelesearch-cli";
              cargoExtraArgs = "-p skelesearch-cli --features skelesearch-core/storage-rocksdb";
            }
          );

          skelesearch-mcp = craneLib.buildPackage (
            commonArgs
            // {
              pname = "skelesearch-mcp";
              cargoExtraArgs = "-p skelesearch-mcp --features skelesearch-core/storage-rocksdb";
            }
          );

          # Alias expected by Claude plugin hook: hooks/session-start calls
          # `nix run .#mcp-server`.
          mcp-server = self.packages.${system}.skelesearch-mcp;

          default = self.packages.${system}.skelesearch-cli;
        };

        devShells.default = pkgs.mkShell {
          # clang provides a C++20-capable compiler needed for optional
          # RocksDB builds during development; cmake/pkg-config mirror
          # the release nativeBuildInputs so `cargo build` works locally.
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.pkg-config
          ];
          buildInputs = [
            pkgs.rustc
            pkgs.cargo
            pkgs.clippy
            pkgs.rustfmt
            pkgs.clang
            # Native TLS (reqwest -> openssl-sys)
            pkgs.openssl
            # Benchmark scripts (TypeScript)
            pkgs.bun
            # ContextBench adapter (Python); uv also provides hf CLI
            # for model downloads: uv tool run --from huggingface_hub hf download ...
            pkgs.python312
            pkgs.uv
            # Repo cloning for benchmarks
            pkgs.git
          ];
          # numpy and other Python C-extensions need libstdc++.so.6 on NixOS.
          # stdenv.cc.cc.lib provides it; LD_LIBRARY_PATH makes it discoverable.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
          ];
        };
      }
    );
}
