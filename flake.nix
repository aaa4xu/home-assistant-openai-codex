{
  description = "OpenAI Codex custom integration for Home Assistant on NixOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      inherit (nixpkgs) lib;

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = lib.genAttrs supportedSystems;

      pkgsFor = system:
        import nixpkgs {
          inherit system;
        };

      mkOpenAICodex =
        pkgs:
        { home-assistant ? pkgs.home-assistant }:
        let
          buildHomeAssistantComponent =
            pkgs.buildHomeAssistantComponent.override {
              inherit home-assistant;
            };
        in
        pkgs.callPackage ./nix/package.nix {
          inherit buildHomeAssistantComponent home-assistant;
        };
    in
    {
      lib.mkOpenAICodex = mkOpenAICodex;

      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
        in
        rec {
          "openai-codex" = mkOpenAICodex pkgs { };
          default = self.packages.${system}."openai-codex";
        }
      );

      checks = forAllSystems (
        system:
        {
          "openai-codex" = self.packages.${system}."openai-codex";
        }
      );

      overlays.default = final: prev: {
        "openai-codex-home-assistant" = mkOpenAICodex final { };

        home-assistant-custom-components =
          (prev.home-assistant-custom-components or { })
          // {
            openai_codex = mkOpenAICodex final { };
          };
      };

      nixosModules.default = self.nixosModules."openai-codex";

      nixosModules."openai-codex" =
        { config, lib, pkgs, ... }:
        let
          cfg = config.services.home-assistant.openaiCodex;

          component = mkOpenAICodex pkgs {
            home-assistant = config.services.home-assistant.package;
          };
        in
        {
          options.services.home-assistant.openaiCodex = {
            enable = lib.mkEnableOption "OpenAI Codex custom integration for Home Assistant";

            package = lib.mkOption {
              type = lib.types.package;
              default = component;
              defaultText = lib.literalExpression ''
                inputs.openai-codex.lib.mkOpenAICodex pkgs {
                  home-assistant = config.services.home-assistant.package;
                }
              '';
              description = ''
                Package for the OpenAI Codex Home Assistant custom component.
                By default this is built against the configured Home Assistant package.
              '';
            };
          };

          config = lib.mkIf cfg.enable {
            services.home-assistant = {
              customComponents = [
                cfg.package
              ];

              # openai_codex declares conversation as a dependency and uses
              # assist_pipeline/intent as after_dependencies in manifest.json.
              extraComponents = [
                "assist_pipeline"
                "conversation"
                "intent"
              ];
            };
          };
        };
    };
}
