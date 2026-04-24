{
  lib,
  buildHomeAssistantComponent,
  openai,
}:

buildHomeAssistantComponent rec {
  owner = "aaa4xu";
  domain = "openai_codex";
  version = "0.1.0";

  src = lib.cleanSourceWith {
    src = ../.;
    filter =
      path: type:
      let
        rel = lib.removePrefix "${toString ../.}/" (toString path);
      in
      rel == "custom_components"
      || rel == "custom_components/${domain}"
      || lib.hasPrefix "custom_components/${domain}/" rel;
  };

  dependencies = [
    openai
  ];

  meta = {
    description = "OpenAI Codex conversation agent for Home Assistant";
    homepage = "https://github.com/aaa4xu/home-assistant-openai-codex";
    license = lib.licenses.mit;
  };
}
