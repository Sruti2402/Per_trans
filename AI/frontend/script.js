const chatBox = document.getElementById("chatBox");
const contextSlider = document.getElementById("contextSlider");
const rangeDisplay = document.getElementById("rangeDisplay");
const selectedInfo = document.getElementById("selectedInfo");
const translationOutput = document.getElementById("translationOutput");
const summaryOutput = document.getElementById("summaryOutput");

let messagesData = [];
let selectedIndex = null;
let contextSize = 100;

const API_URL = "http://127.0.0.1:8000";

/* ==========================
   Slider
========================== */

contextSlider.addEventListener("input", () => {

    contextSize = parseInt(contextSlider.value);

    rangeDisplay.textContent =
        `${contextSize} Messages Selected`;

    if (selectedIndex !== null) {

        highlightContextWindow();

        selectMessage(selectedIndex);
    }
});

/* ==========================
   Load Messages
========================== */

async function loadMessages() {

    try {

        const response = await fetch("./data/messages.json");

        if (!response.ok) {
            throw new Error("Cannot load messages.json");
        }

        messagesData = await response.json();

        console.log(
            "Messages Loaded:",
            messagesData.length
        );

        renderMessages();

    } catch (err) {

        console.error(err);

        chatBox.innerHTML = `
            <h2 style="color:red">
                Failed to load messages.json
            </h2>
            <p>Open using Live Server</p>
        `;
    }
}

/* ==========================
   Render Messages
========================== */

function renderMessages() {

    chatBox.innerHTML = "";

    messagesData.forEach((msg, index) => {

        const messageDiv =
            document.createElement("div");

        const type =
            Math.random() > 0.5
                ? "sent"
                : "received";

        messageDiv.className =
            `message ${type}`;

        messageDiv.dataset.index = index;

        messageDiv.innerHTML = `
            <div class="sender">
                ${msg.sender}
            </div>

            <div class="message-text">
                ${msg.text}
            </div>

            <div class="time">
                ${new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
        })}
            </div>
        `;

        messageDiv.addEventListener(
            "click",
            () => selectMessage(index)
        );

        chatBox.appendChild(messageDiv);
    });

    chatBox.scrollTop =
        chatBox.scrollHeight;
}

/* ==========================
   Select Message
========================== */

function selectMessage(index) {

    selectedIndex = index;

    highlightContextWindow();

    const selectedElement =
        document.querySelector(`[data-index="${index}"]`);

    if (selectedElement) {
        selectedElement.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }

    const half = Math.floor(contextSize / 2);

    let start = index - half;
    let end = index + half;

    if (start < 0) {
        end += -start;
        start = 0;
    }

    if (end >= messagesData.length) {
        start -= (end - messagesData.length + 1);
        end = messagesData.length - 1;

        if (start < 0)
            start = 0;
    }

    selectedInfo.textContent =
        `Selected Message #${index + 1} | Context: ${start + 1}-${end + 1}`;
}

/* ==========================
   Highlight Context
========================== */

function highlightContextWindow() {

    document.querySelectorAll(".message")
        .forEach(msg => msg.classList.remove("selected-window"));

    if (selectedIndex === null) return;

    const half = Math.floor(contextSize / 2);

    let start = selectedIndex - half;
    let end = selectedIndex + half;

    if (start < 0) {
        end += -start;
        start = 0;
    }

    if (end >= messagesData.length) {
        start -= (end - messagesData.length + 1);
        end = messagesData.length - 1;

        if (start < 0)
            start = 0;
    }

    for (let i = start; i <= end; i++) {

        const msg = document.querySelector(
            `[data-index="${i}"]`
        );

        if (msg)
            msg.classList.add("selected-window");
    }
}

/* ==========================
   Get Context Window
========================== */

function getContextWindow() {

    if (selectedIndex === null)
        return [];

    const half = Math.floor(contextSize / 2);

    let start = selectedIndex - half;
    let end = selectedIndex + half;

    if (start < 0) {
        end += -start;
        start = 0;
    }

    if (end >= messagesData.length) {
        start -= (end - messagesData.length + 1);
        end = messagesData.length - 1;

        if (start < 0)
            start = 0;
    }

    return messagesData.slice(start, end + 1);
}

/* ==========================
   Translate
========================== */

async function translateSelected() {

    if (selectedIndex === null) {

        alert(
            "Please select a message first"
        );

        return;
    }

    const selectedMessage =
        messagesData[selectedIndex];

    const context =
        getContextWindow();

    const targetLanguage =
        document.getElementById(
            "languageSelect"
        ).value;

    translationOutput.innerHTML =
        "<p>Translating...</p>";

    try {

        const response = await fetch(
            `${API_URL}/translate`,
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify({
                    context_messages: context,
                    target_language: targetLanguage
                })
            }
        );

        if (!response.ok) {
            const text = await response.text();
            throw new Error(
                `Server Error ${response.status} - ${text}`
            );
        }

        const data = await response.json();

        // Save for evaluation page
        localStorage.setItem(
            "translation",
            JSON.stringify(data.translations)
        );

        localStorage.setItem(
            "context",
            JSON.stringify(context)
        );

        if (!Array.isArray(data.translations)) {
            throw new Error("Invalid translation response from server.");
        }

        let html = "";

        data.translations.forEach((msg, index) => {
            html += `
                <div class="translated-card">
                    <p><b>${index + 1}. ${msg.sender}</b></p>
                    <p><b>Original:</b> ${msg.original}</p>
                    <p><b>Translated:</b> ${msg.translation}</p>
                </div>
            `;
        });

        html += `
            <p><strong>Messages Translated:</strong> ${data.translations.length
            }</p>
        `;

        translationOutput.innerHTML = html;

    } catch (err) {

        console.error(err);

        translationOutput.innerHTML = `
            <p style="color:red">
                Backend connection failed.
            </p>
            <p>
                ${err.message}
            </p>
        `;
    }
}

/* ==========================
   Summarize
========================== */

async function summarizeSelected() {

    if (selectedIndex === null) {

        alert(
            "Please select a message first"
        );

        return;
    }

    const context =
        getContextWindow();

    summaryOutput.innerHTML =
        "<p>Generating summary...</p>";

    try {

        const response = await fetch(
            `${API_URL}/summarize`,
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body: JSON.stringify({
                    context_messages:
                        context
                })
            }
        );

        if (!response.ok) {
            const text = await response.text();
            throw new Error(
                `Server Error ${response.status} - ${text}`
            );
        }

        const data =
            await response.json();

        localStorage.setItem(
            "summary",
            data.summary
        );

        summaryOutput.innerHTML = `
            <div>
                <p>${data.summary}</p>
                <p><strong>Messages Used:</strong> ${context.length}</p>
            </div>
        `;

    } catch (err) {

        console.error(err);

        summaryOutput.innerHTML = `
            <p style="color:red">
                Backend connection failed.
            </p>
            <p>
                ${err.message}
            </p>
        `;
    }
}

/* ==========================
   Open Evaluation Page
========================== */

function openEvaluation() {

    window.open(
        "evaluate.html",
        "_blank"
    );

}

/* ==========================
   Start
========================== */

loadMessages();