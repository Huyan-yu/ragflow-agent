/**
 * RAGFlow Agent — 前端逻辑
 */

const API_BASE = "http://localhost:8000";

// 状态
let sessionId = null;

// 元素
const chatContainer = document.getElementById("chat-container");
const msgInput = document.getElementById("msg-input");
const btnSend = document.getElementById("btn-send");
const sessionLabel = document.getElementById("session-id");
const fileInput = document.getElementById("file-input");

// ---------- 工具函数 ----------

function removeEmptyState() {
    const es = chatContainer.querySelector(".empty-state");
    if (es) es.remove();
}

function addMsg(role, content) {
    removeEmptyState();
    const div = document.createElement("div");
    div.className = `msg ${role}`;

    if (role === "user") {
        div.innerHTML = `
            <div class="msg-avatar">👤</div>
            <div class="msg-bubble"></div>
        `;
        div.querySelector(".msg-bubble").textContent = content;
    } else {
        div.innerHTML = `
            <div class="msg-avatar">🤖</div>
            <div class="msg-content"></div>
        `;
        div.querySelector(".msg-content").textContent = content;
    }

    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return div;
}

function addToolCard(toolName, args, result) {
    const lastMsg = chatContainer.querySelector(".msg.bot:last-child");
    if (!lastMsg) return;
    const contentEl = lastMsg.querySelector(".msg-content");

    const card = document.createElement("div");
    card.className = "tool-card";
    card.innerHTML = `
        <div class="tool-name">${escapeHtml(toolName)}</div>
        <div style="font-size:11px;color:var(--text2);margin-top:2px">${escapeHtml(JSON.stringify(args))}</div>
        <div class="tool-result">${escapeHtml(result || "")}</div>
    `;
    contentEl.appendChild(card);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addSourceTags(sources) {
    if (!sources?.length) return;
    const lastMsg = chatContainer.querySelector(".msg.bot:last-child");
    if (!lastMsg) return;
    const contentEl = lastMsg.querySelector(".msg-content");

    const row = document.createElement("div");
    sources.forEach(s => {
        const tag = document.createElement("span");
        tag.className = "source-tag";
        tag.textContent = `📄 ${s}`;
        row.appendChild(tag);
    });
    contentEl.appendChild(row);
}

function addTypingIndicator() {
    const div = document.createElement("div");
    div.className = "msg bot";
    div.id = "typing-indicator";
    div.innerHTML = `
        <div class="msg-avatar">🤖</div>
        <div class="typing"><span></span><span></span><span></span></div>
    `;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return div;
}

function removeTypingIndicator() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
}

// ---------- 对话 ----------

async function sendMessage(text) {
    if (!text?.trim()) return;
    msgInput.value = "";

    addMsg("user", text);
    const typing = addTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text,
                session_id: sessionId || undefined,
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        sessionId = data.session_id;
        sessionLabel.textContent = `session: ${data.session_id.slice(0, 8)}…`;

        removeTypingIndicator();
        addMsg("bot", data.reply);

        // 工具调用卡片
        (data.tool_calls || []).forEach(tc => {
            addToolCard(tc.tool, tc.args, tc.result);
        });
        addSourceTags(data.sources);

    } catch (e) {
        removeTypingIndicator();
        addMsg("bot", `⚠️ 请求失败：${e.message}\n请确认后端已启动（http://localhost:8000）`);
    }
}

// 输入框事件
btnSend.addEventListener("click", () => {
    sendMessage(msgInput.value);
});

msgInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage(msgInput.value);
    }
});

// 推荐问题
document.querySelectorAll(".suggestion").forEach(btn => {
    btn.addEventListener("click", () => sendMessage(btn.dataset.q));
});

// 新对话
document.getElementById("btn-new").addEventListener("click", () => {
    sessionId = null;
    sessionLabel.textContent = "无会话";
    chatContainer.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">🤖</div>
            <p>试试问我：</p>
            <div class="suggestions">
                <button class="suggestion" data-q="帮我计算 (2+3)*4 平方根是多少？">帮我计算 (2+3)*4 平方根是多少？</button>
                <button class="suggestion" data-q="广州今天天气怎么样？">广州今天天气怎么样？</button>
                <button class="suggestion" data-q="出差住宿标准是多少？">出差住宿标准是多少？</button>
                <button class="suggestion" data-q="RAGFlow 平台支持哪些模型？">RAGFlow 平台支持哪些模型？</button>
            </div>
        </div>
    `;
    document.querySelectorAll(".suggestion").forEach(btn => {
        btn.addEventListener("click", () => sendMessage(btn.dataset.q));
    });
});

// ---------- 知识库管理 ----------

document.getElementById("btn-import").addEventListener("click", () => {
    fileInput.click();
});

fileInput.addEventListener("change", async () => {
    const files = [...fileInput.files];
    if (!files.length) return;

    const formData = new FormData();
    files.forEach(f => formData.append("files", f));

    document.getElementById("kb-status").textContent = "正在导入...";
    try {
        const res = await fetch(`${API_BASE}/api/kb/import`, {
            method: "POST",
            body: formData,
        });
        const data = await res.json();
        document.getElementById("kb-status").textContent =
            `✅ 已导入 ${data.imported.join("、")}\n新增分块：${data.new_chunks}`;
    } catch (e) {
        document.getElementById("kb-status").textContent = `❌ 导入失败：${e.message}`;
    }
    fileInput.value = "";
});

document.getElementById("btn-status").addEventListener("click", async () => {
    try {
        const res = await fetch(`${API_BASE}/api/kb/status`);
        const data = await res.json();
        document.getElementById("kb-status").textContent =
            `has_index: ${data.has_index}\nindex_dir: ${data.index_dir}`;
    } catch (e) {
        document.getElementById("kb-status").textContent = `❌ ${e.message}`;
    }
});

document.getElementById("btn-reset").addEventListener("click", async () => {
    if (!confirm("确认清空知识库？")) return;
    try {
        await fetch(`${API_BASE}/api/kb`, { method: "DELETE" });
        document.getElementById("kb-status").textContent = "✅ 知识库已清空";
    } catch (e) {
        document.getElementById("kb-status").textContent = `❌ ${e.message}`;
    }
});

// 加载时获取知识库状态
document.getElementById("btn-status").click();
