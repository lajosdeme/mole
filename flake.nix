{
  description = "mole — a deep research agent in Go, exposed over MCP";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    gitignore = {
      url = "github:hercules-ci/gitignore.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      gitignore,
    }:
    let
      systems = nixpkgs.lib.systems.flakeExposed;
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # `mole version` should say which code is running. Flakes only ever see
      # commit hashes (self.rev / dirtyShortRev), never tags; tags reach the
      # binary only via goreleaser. So a nix-built mole reports the short commit,
      # or "dev" outside a flake checkout.
      version =
        if self ? rev && self.rev != null then
          self.shortRev
        else if self ? dirtyShortRev && self.dirtyShortRev != null then
          self.dirtyShortRev
        else
          "dev";

      packages =
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        pkgs.callPackage ./nix/package.nix {
          inherit version;
          src = gitignore.lib.gitignoreSource ./.;
        };
    in
    {
      packages = forAllSystems (system: {
        default = (packages system).default;
        mole = (packages system).mole;
        "mole-mcp" = (packages system).mole-mcp;
      });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${(packages system).default}/bin/mole";
        };
        mole = {
          type = "app";
          program = "${(packages system).mole}/bin/mole";
        };
        "mole-mcp" = {
          type = "app";
          program = "${(packages system).mole-mcp}/bin/mole-mcp";
        };
      });

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            name = "mole-dev";

            packages = with pkgs; [
              go
              goreleaser
              gopls
            ];
          };
        }
      );

      # Keep the flake itself formatted: nix fmt
      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt);
    };
}
