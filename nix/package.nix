{
  lib,
  buildHomeAssistantComponent,
  home-assistant,
}:

buildHomeAssistantComponent rec {
  owner = "aaa4xu";
  domain = "openai_codex";
  version = "0.1.0";

  src = lib.cleanSourceWith {
    src = ../.;
    filter =
      path: type:
      lib.cleanSourceFilter path type
      && (
        let
          root = toString ../.;
          pathStr = toString path;
          rel = lib.removePrefix "${root}/" pathStr;
        in
        pathStr == root
        || rel == "custom_components"
        || rel == "custom_components/${domain}"
        || lib.hasPrefix "custom_components/${domain}/" rel
      );
  };

  dependencies = with home-assistant.python.pkgs; [
    aiortc
    av
    openai
  ];

  meta = {
    description = "OpenAI Codex conversation, STT, and TTS for Home Assistant";
    homepage = "https://github.com/${owner}/home-assistant-openai-codex";
    license = lib.licenses.mit;
    platforms = lib.platforms.linux;
  };
}
