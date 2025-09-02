// static/script.js
const chat = document.getElementById("chat");
const input = document.getElementById("msg");
const sendBtn = document.getElementById("sendBtn");
const summarizeBtn = document.getElementById("summarizeBtn");
const fileInput = document.getElementById("fileInput");
const summaryText = document.getElementById("summaryText");
const newChatBtn = document.getElementById("newChatBtn");

let turns = [];

function addMsg(role, text) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "user" : "bot");
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  turns.push({ role: "user", text });
  addMsg("user", text);
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turns }),
    });
    const data = await res.json();
    if (!res.ok) {
      addMsg("assistant", "⚠️ " + (data.detail || "Chat error"));
      return;
    }
    turns.push({ role: "assistant", text: data.reply || "Okay." });
    addMsg("assistant", data.reply || "Okay.");
  } catch (e) {
    addMsg("assistant", "⚠️ Network error");
    console.error(e);
  }
}

async function summarize() {
  try {
    const res = await fetch("/api/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turns }),
    });
    const data = await res.json();
    if (!res.ok) {
      summaryText.textContent = "⚠️ " + (data.detail || "Summarize error");
      return;
    }
    summaryText.textContent = data.summary || "(no summary)";
  } catch (e) {
    summaryText.textContent = "⚠️ Network error";
    console.error(e);
  }
}

fileInput.addEventListener("change", async (evt) => {
  const file = evt.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/load_dialog", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      addMsg("assistant", "⚠️ " + (data.detail || "Failed to load dialog"));
      return;
    }
    chat.innerHTML = "";
    turns = [];
    (data.turns || []).forEach(row => {
      const role = row.role || ((row.speaker || "").toLowerCase().includes("person1") ? "user" : "assistant");
      const text = row.text || "";
      turns.push({ role, text });
      addMsg(role, text);
    });
    summaryText.textContent = "(no summary yet)";
  } catch (e) {
    addMsg("assistant", "⚠️ Network error");
    console.error(e);
  }
});

sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });
summarizeBtn.addEventListener("click", summarize);
newChatBtn?.addEventListener("click", () => { turns = []; chat.innerHTML = ""; summaryText.textContent = "(no summary yet)"; });
