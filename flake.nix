{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, crane, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        # crane v2: mkLib replaced the old crane.lib.${system} attribute
        craneLib = crane.mkLib pkgs;
        src = craneLib.cleanCargoSource ./.;

        # Some transitive deps (ring, aws-lc-sys) require Security and
        # SystemConfiguration on Darwin.
        darwinFrameworks = pkgs.lib.optionals pkgs.stdenv.isDarwin (
          with pkgs.darwin.apple_sdk.frameworks; [
            Security
            SystemConfiguration
          ]
        );

        # cmake and pkg-config are needed at build time for CozoDB/RocksDB.
        commonArgs = {
          inherit src;
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = darwinFrameworks;
        };
      in {
        packages = {
          # Release packages always compile against RocksDB for production
          # durability; SQLite remains the dev-shell default.
          skelesearch-cli = craneLib.buildPackage (commonArgs // {
            cargoExtraArgs = "-p skelesearch-cli --features skelesearch-core/storage-rocksdb";
          });

          skelesearch-mcp = craneLib.buildPackage (commonArgs // {
            cargoExtraArgs = "-p skelesearch-mcp --features skelesearch-core/storage-rocksdb";
          });

          # Alias expected by Claude plugin hook: hooks/session-start calls
          # `nix run .#mcp-server`.
          mcp-server = self.packages.${system}.skelesearch-mcp;

          default = self.packages.${system}.skelesearch-cli;
        };

        devShells.default = pkgs.mkShell {
          # clang provides a C++20-capable compiler needed for optional
          # RocksDB builds during development; cmake/pkg-config mirror
          # the release nativeBuildInputs so `cargo build` works locally.
          buildInputs = [
            pkgs.rustc
            pkgs.cargo
            pkgs.cmake
            pkgs.pkg-config
            pkgs.clang
          ] ++ darwinFrameworks;
        };
      });
}
