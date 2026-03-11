/**
 * Smart Campus Assistant — Frontend Logic
 * 
 * Handles: sending questions, rendering responses, 
 * loading states, fallback display, and error handling.
 */

// ══════════════════════════════════════════
// Configuration
// ══════════════════════════════════════════

const API_BASE = "http://localhost:8000/api";

// DOM Elements
const chatMessages = document.getElementById("chatMessages");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const charCount = document.getElementById("charCount");
const statusBadge = document.getElementById("statusBadge");


// ══════════════════════════════════════════
// Event Listeners
// ══════════════════════════════════════════

// Send on Enter key
questionInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

// Character counter
questionInput.addEventListener("input", function () {
    const len = this.value.length;
    charCount.textContent = len + " / 500";
    charCount.style.color = len > 450 ? "#c0392b" : "";
});

// Check API health on load
window.addEventListener("load", checkHealth);


// ══════════════════════════════════════════
// Main Functions
// ══════════════════════════════════════════

/**
 * Send a question to the backend API.
 */
async function sendQuestion() {
    const question = questionInput.value.trim();

    // Validate input
    if (question.length < 3) {
        shakeInput();
        return;
    }

    // Disable input while processing
    setInputState(false);

    // Show user message
    addUserMessage(question);
    questionInput.value = "";
    charCount.textContent = "0 / 500";

    // Show typing indicator
    const typingId = showTypingIndicator();

    try {
        const response = await fetch(API_BASE + "/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: question }),
        });

        // Remove typing indicator
        removeTypingIndicator(typingId);

        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            const detail = errorData?.detail || "Server error (" + response.status + ")";
            addErrorMessage(detail);
            return;
        }

        const data = await response.json();

        // Check if it's a fallback response (has "message" field instead of "answer")
        if (data.message) {
            addFallbackMessage(data);
        } else {
            addBotMessage(data);
        }

    } catch (error) {
        removeTypingIndicator(typingId);
        console.error("API Error:", error);
        addErrorMessage("Could not connect to the server. Make sure the backend is running.");
    } finally {
        setInputState(true);
        questionInput.focus();
    }
}


/**
 * Ask a question from a suggestion chip.
 */
function askSuggestion(question) {
    questionInput.value = question;
    charCount.textContent = question.length + " / 500";
    sendQuestion();
}


// ══════════════════════════════════════════
// Message Rendering
// ══════════════════════════════════════════

/**
 * Add a user message bubble.
 */
function addUserMessage(text) {
    const html = `
        <div class="message user-message">
            <div class="message-avatar">S</div>
            <div class="message-content">
                <div class="message-bubble">
                    <p>${escapeHtml(text)}</p>
                </div>
            </div>
        </div>
    `;
    chatMessages.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
}


/**
 * Add a bot answer message with metadata tags.
 */
function addBotMessage(data) {
    const confidenceClass = data.confidence >= 0.8 ? "high" : data.confidence >= 0.6 ? "medium" : "low";
    const confidencePercent = Math.round(data.confidence * 100);
    const categoryLabel = formatCategory(data.category);

    const html = `
        <div class="message bot-message">
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                    <path d="M6 12v5c0 1.1 2.7 2 6 2s6-.9 6-2v-5"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <p>${escapeHtml(data.answer)}</p>
                </div>
                <div class="message-meta">
                    <span class="tag tag-${data.category}">${categoryLabel}</span>
                    <span class="tag tag-confidence ${confidenceClass}">${confidencePercent}% confidence</span>
                    <span class="tag-time">${data.response_time_ms}ms</span>
                </div>
            </div>
        </div>
    `;
    chatMessages.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
}


/**
 * Add a fallback message (low confidence or out-of-scope).
 */
function addFallbackMessage(data) {
    const categoryLabel = formatCategory(data.category);
    const contact = data.staff_contact || {};

    const html = `
        <div class="message bot-message">
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                    <path d="M6 12v5c0 1.1 2.7 2 6 2s6-.9 6-2v-5"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-bubble fallback-bubble">
                    <p class="fallback-title">⚠️ Low Confidence</p>
                    <p>${escapeHtml(data.message)}</p>
                    ${data.suggestion ? '<p style="margin-top:6px;font-size:13px;color:#5f6570;">' + escapeHtml(data.suggestion) + '</p>' : ''}
                    ${contact.email ? `
                        <div class="staff-contact">
                            <strong>Campus Support</strong>
                            📧 ${escapeHtml(contact.email)}<br>
                            📞 ${escapeHtml(contact.phone || '')}<br>
                            📍 ${escapeHtml(contact.office || '')}
                        </div>
                    ` : ''}
                </div>
                <div class="message-meta">
                    <span class="tag tag-${data.category}">${categoryLabel}</span>
                    <span class="tag tag-confidence low">Fallback</span>
                </div>
            </div>
        </div>
    `;
    chatMessages.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
}


/**
 * Add an error message.
 */
function addErrorMessage(text) {
    const html = `
        <div class="message bot-message">
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                    <path d="M6 12v5c0 1.1 2.7 2 6 2s6-.9 6-2v-5"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-bubble error-bubble">
                    <p>❌ ${escapeHtml(text)}</p>
                </div>
            </div>
        </div>
    `;
    chatMessages.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
}


// ══════════════════════════════════════════
// UI Helpers
// ══════════════════════════════════════════

/**
 * Show a typing/loading indicator.
 */
function showTypingIndicator() {
    const id = "typing-" + Date.now();
    const html = `
        <div class="message bot-message" id="${id}">
            <div class="message-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                    <path d="M6 12v5c0 1.1 2.7 2 6 2s6-.9 6-2v-5"/>
                </svg>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
        </div>
    `;
    chatMessages.insertAdjacentHTML("beforeend", html);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

/**
 * Enable/disable input during API call.
 */
function setInputState(enabled) {
    questionInput.disabled = !enabled;
    sendBtn.disabled = !enabled;
}

/**
 * Shake the input to indicate validation error.
 */
function shakeInput() {
    questionInput.style.animation = "none";
    questionInput.offsetHeight; // Force reflow
    questionInput.style.animation = "shake 0.4s ease";
    questionInput.style.borderColor = "#c0392b";
    setTimeout(() => {
        questionInput.style.borderColor = "";
        questionInput.style.animation = "";
    }, 600);
}

/**
 * Scroll chat to the bottom.
 */
function scrollToBottom() {
    const container = document.querySelector(".chat-container");
    container.scrollTop = container.scrollHeight;
}

/**
 * Format category enum to display label.
 */
function formatCategory(category) {
    const labels = {
        schedule: "📅 Schedule",
        general_info: "ℹ️ General Info",
        technical_issue: "🔧 Technical",
        out_of_scope: "❌ Off-topic",
    };
    return labels[category] || category;
}

/**
 * Escape HTML to prevent XSS.
 */
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Check API health and update status badge.
 */
async function checkHealth() {
    try {
        const response = await fetch(API_BASE + "/health");
        const data = await response.json();

        if (data.status === "healthy") {
            statusBadge.innerHTML = '<span class="status-dot"></span> Online';
            statusBadge.style.color = "#2d8a56";
        } else {
            statusBadge.innerHTML = '<span class="status-dot" style="background:#e67e22"></span> Degraded';
            statusBadge.style.color = "#e67e22";
        }
    } catch {
        statusBadge.innerHTML = '<span class="status-dot" style="background:#c0392b;animation:none"></span> Offline';
        statusBadge.style.color = "#c0392b";
    }
}

// CSS for shake animation (injected)
const style = document.createElement("style");
style.textContent = `@keyframes shake { 0%,100%{transform:translateX(0)} 25%{transform:translateX(-6px)} 50%{transform:translateX(6px)} 75%{transform:translateX(-4px)} }`;
document.head.appendChild(style);
