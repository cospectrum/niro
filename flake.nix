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
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system overlays;
            config.allowUnfree = false;
          };
        in
        {
          inherit (pkgs) actionlint zizmor;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system overlays;
            config.allowUnfree = false;
          };
        in
        {
          default = pkgs.mkShellNoCC {
            packages = with pkgs; [ uv ];
          };
        }
      );
    };
}
