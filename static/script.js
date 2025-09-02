
const chat = document.getElementById("chat");
const input = document.getElementById("msg");
const sendBtn = document.getElementById("sendBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const fileInput = document.getElementById("fileInput");
const summaryText = document.getElementById("summaryText");
const newChatBtn = document.getElementById("newChatBtn");

let turns = [];

function fmtTime(ts){ try { return new Date(ts).toLocaleTimeString(); } catch { return ""; } }

function addMsg(role, text, ts = new Date().toISOString()) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "user" : "bot");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = fmtTime(ts);
  wrap.appendChild(bubble);
  wrap.appendChild(meta);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  const ts = new Date().toISOString();
  turns.push({ role: "user", text, ts });
  addMsg("user", text, ts);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turns }),
    });
    const data = await res.json();
    const reply = data.reply || "Okay.";
    const ts2 = new Date().toISOString();
    turns.push({ role: "assistant", text: reply, ts: ts2 });
    addMsg("assistant", reply, ts2);
  } catch (e) {
    const ts2 = new Date().toISOString();
    turns.push({ role: "assistant", text: "Error contacting backend.", ts: ts2 });
    addMsg("assistant", "Error contacting backend.", ts2);
  }
}

async function summarize() {
  try {
    const res = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turns })
    });
    const data = await res.json();
    summaryText.textContent = data.summary || "(no summary)";
  } catch (e) {
    summaryText.textContent = "Error summarizing.";
  }
}

fileInput.addEventListener("change", async (evt) => {
  const file = evt.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/load_dialog", { method: "POST", body: form });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  chat.innerHTML = "";
  turns = [];
  (data.turns || []).forEach(row => {
    const role = row.role || ((row.speaker||"").toLowerCase().includes("person1") ? "user" : "assistant");
    const text = row.text || "";
    const ts = row.ts || new Date().toISOString();
    turns.push({ role, text, ts });
    addMsg(role, text, ts);
  });
  summaryText.textContent = "(no summary yet)";
});

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });
summarizeBtn.addEventListener("click", summarize);
newChatBtn.addEventListener("click", () => { turns = []; chat.innerHTML = ""; summaryText.textContent = "(no summary yet)"; input.focus(); });
