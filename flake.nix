{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];

      overlays = [ ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
      pkgsFor = system:
        import nixpkgs {
          inherit system overlays;
          config.allowUnfree = false;
        };
    in
    {
      # Expose standalone tools, e.g. `nix run .#actionlint`.
      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        {
          inherit (pkgs) actionlint zizmor;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        {
          default = pkgs.mkShellNoCC {
            packages = with pkgs; [ uv ];
          };
        }
      );

      apps = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          ci = pkgs.writeShellApplication {
            name = "niro-ci";
            runtimeInputs = with pkgs; [
              actionlint
              uv
              zizmor
            ];
            text = ''
              echo ":: Sync dependencies"
              uv sync --locked

              echo ":: Test with branch coverage"
              uv run --no-sync coverage run --branch --source=niro -m pytest

              echo ":: Report coverage"
              uv run --no-sync coverage report

              echo ":: Type check"
              uv run --no-sync ty check

              echo ":: Format check"
              uv run --no-sync ruff format --check

              echo ":: Lint"
              uv run --no-sync ruff check

              echo ":: Check GitHub Actions workflows"
              actionlint

              echo ":: Audit GitHub Actions workflows"
              zizmor --offline --persona=regular --collect=workflows .
            '';
          };
        in
        {
          ci = {
            type = "app";
            program = "${ci}/bin/niro-ci";
            meta.description = "Run niro's local CI checks";
          };
        }
      );
    };
}
