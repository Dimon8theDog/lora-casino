// LoRA Casino inside ComfyUI: a button in the top bar and a command (so it is also in the command search and can
// get a keyboard shortcut). Both open the table in a new browser tab; the table itself is served by the plugin.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const open = () => window.open((api.api_base || "") + "/lora-casino/", "lora-casino");

// The button's icon (two playing cards) is drawn here: ComfyUI's own icon classes only cover the icons ComfyUI itself
// uses, so a plugin has to bring its picture along. It takes the colour of the text around it.
const CARDS = "data:image/svg+xml," + encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
  + '<rect x="3.5" y="7" width="10.5" height="13.5" rx="2"/><path d="M9 3.5h9.5a2 2 0 0 1 2 2V17"/><path d="M8.75 11.5v4.5"/></svg>');
const style = document.createElement("style");
style.textContent = `.lora-casino-icon { display: inline-block; width: 1rem; height: 1rem; background-color: currentColor;
  -webkit-mask: url("${CARDS}") center / contain no-repeat; mask: url("${CARDS}") center / contain no-repeat; }`;
document.head.appendChild(style);

app.registerExtension({
  name: "LoRACasino",
  commands: [{ id: "LoRACasino.Open", label: "Open LoRA Casino", icon: "lora-casino-icon", function: open }],
  menuCommands: [{ path: ["Extensions"], commands: ["LoRACasino.Open"] }],
  actionBarButtons: [{ icon: "lora-casino-icon", label: "LoRA Casino", tooltip: "Open LoRA Casino: deal pictures, keep the hand you like as a LoRA", onClick: open }],
});
