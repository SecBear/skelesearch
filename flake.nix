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

        # buildDepsOnly uses a dummy source tree containing Cargo manifests and
        # synthetic crate sources. Keep the real sqlite3-sys shim in that dummy
        # tree so the patched path dependency resolves during dependency builds.
        dummySrc = pkgs.runCommand "skelesearch-dummy-src" { } ''
          cp -r ${craneLib.mkDummySrc { inherit src; }} $out
          chmod -R +w $out
          rm -rf $out/vendor/sqlite3-sys
          mkdir -p $out/vendor
          cp -r ${src}/vendor/sqlite3-sys $out/vendor/sqlite3-sys
        '';

        # cmake/pkg-config are needed for RocksDB; onnxruntime satisfies ort-sys
        # without network downloads so Nix builds stay reproducible/offline.
        commonArgs = {
          inherit src dummySrc;
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.pkg-config
          ];
          buildInputs = [
            pkgs.onnxruntime
          ];
          ORT_STRATEGY = "system";
          ORT_LIB_LOCATION = "${pkgs.lib.getLib pkgs.onnxruntime}/lib";
          ORT_PREFER_DYNAMIC_LINK = "1";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib pkgs.onnxruntime)
          ];
          DYLD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            (pkgs.lib.getLib pkgs.onnxruntime)
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
              cargoExtraArgs = "-p skelesearch-mcp --features storage-rocksdb";
            }
          );

          skelesearch-daemon = craneLib.buildPackage (
            commonArgs
            // {
              pname = "skelesearch-daemon";
              cargoExtraArgs = "-p skelesearch-daemon --features storage-rocksdb";
            }
          );

          onnxruntime-lib = pkgs.lib.getLib pkgs.onnxruntime;

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
            # libc++ runtime/headers for Darwin links that pass -lc++
            pkgs.llvmPackages.libcxx
            # bindgen needs libclang for RocksDB (cozorocks) builds
            pkgs.llvmPackages.libclang.lib
            # Native TLS (reqwest -> openssl-sys)
            pkgs.openssl
            # ONNX Runtime for fastembed / local rerankers without network downloads
            pkgs.onnxruntime
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
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          ORT_STRATEGY = "system";
          ORT_LIB_LOCATION = "${pkgs.lib.getLib pkgs.onnxruntime}/lib";
          ORT_PREFER_DYNAMIC_LINK = "1";
          # Force clang for RocksDB C++ compilation (g++ may lack headers on NixOS)
          CXX = "clang++";
          CC = "clang";
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            (pkgs.lib.getLib pkgs.onnxruntime)
          ];
        };
      }
    );
}
