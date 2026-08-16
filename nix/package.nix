{
  lib,
  buildGoModule,
  src,
  version,
}:
let
  mole = buildGoModule {
    pname = "mole";
    inherit src version;
    subPackages = [
      "cmd/mole"
      "cmd/mole-mcp"
    ];
    vendorHash = "sha256-z/wZZlTVXlhwlCz7un9WN+CMSg1F/2O6HSvLRtrqu1g=";

    # Parity with .goreleaser.yaml, which passes -trimpath to both builds.
    buildFlags = [
      "-trimpath"
    ];
    ldflags = [
      "-s"
      "-w"
      "-X main.version=${version}"
    ];

    env.CGO_ENABLED = "0";

    checkPhase = ''
      runHook preCheck
      # When run with the default tmp dir on macOS, the paths used for the Unix
      # sockets in the tests get too long, and tests fail as a result.
      # Overriding `TMPDIR` to the shorter `/tmp` overcomes this and allows the
      # checkPhase to succeed.
      TMPDIR=/tmp go test ./...
      runHook postCheck
    '';

    meta = {
      description = "mole — a deep research agent in Go";
      homepage = "https://github.com/lajosdeme/mole";
      license = lib.licenses.asl20;
      mainProgram = "mole";
    };
  };
in
{
  inherit mole;
  default = mole;
  # Same derivation; both binaries come from the one build.
  "mole-mcp" = mole;
}
