// main.js

import { loadTrain, loadInference, loadBulkInference, loadPrepareDataset } from "./modes.js";


const socketio = io();


const fsBtn = document.getElementById("fullscreen-btn");
const output = document.getElementById("output-container");

fsBtn.addEventListener("click", async () => {
    if (!document.fullscreenElement) {
        await output.requestFullscreen();
        document.body.classList.add("fullscreen-active");
    } else {
        await document.exitFullscreen();
        document.body.classList.remove("fullscreen-active");
    }
});

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        output.classList.remove("fullscreen");
        document.body.classList.remove("fullscreen-active");
    }
});

/* -------------------------------
   LIVE LOG PANEL
-------------------------------- */
const logPanel = document.getElementById("log-panel");
const logHeader = document.getElementById("log-header");
const logBody = document.getElementById("log-body");
const toggleBtn = document.getElementById("toggle-log");

/* -------------------------------
   TOGGLE LOG PANEL
-------------------------------- */
toggleBtn.addEventListener("click", () => {
    const isClosed = logPanel.classList.toggle("closed");
    toggleBtn.textContent = isClosed ? "+" : "–";
    if (isClosed) {
        logPanel.style.height = "42px";  // reset to header height
    } else {
        logPanel.style.height = "300px"; // restore default
    }
});

/* -------------------------------
   DRAGGING BEHAVIOR
-------------------------------- */
let dragging = false;
let offsetX = 0;
let offsetY = 0;

logHeader.addEventListener("mousedown", e => {
    dragging = true;
    offsetX = e.clientX - logPanel.offsetLeft;
    offsetY = e.clientY - logPanel.offsetTop;
});

document.addEventListener("mousemove", e => {
    if (!dragging) return;

    logPanel.style.left = `${e.clientX - offsetX}px`;
    logPanel.style.top = `${e.clientY - offsetY}px`;

    logPanel.style.right = "auto";
    logPanel.style.bottom = "auto";
});

document.addEventListener("mouseup", () => {
    dragging = false;
});

/* -------------------------------
   SOCKET_IO HANDLER
-------------------------------- */

socketio.on('single_inference_result', data => {

    const outputContent = document.getElementById('output-content');
    outputContent.innerHTML = '';

    const result_url = data?.progress?.result?.result_url;

    if (result_url) {
        const img = document.createElement('img');
        img.src = result_url + '?' + new Date().getTime();
        img.alt = 'Inference Result';

        outputContent.appendChild(img);
    } 
    else {
        outputContent.innerHTML = '<p>Image not available</p>';
        console.warn('result_url is undefined:', data);
    }

});

/* -------------------------------
   LOG STREAM HANDLER
-------------------------------- */

socketio.on("live_logs", payload => {
    appendLog(payload);
});

/* -------------------------------
   LOG APPENDER
-------------------------------- */

function appendLog(payload) {
    const line = document.createElement("div");
    line.className = "log-line";
    let message = "";
    let level = "";
    if (payload?.progress) {
        const { level: lvl, status, logs } = payload.progress;
        level = lvl || "";
        message = status || "";
        if (logs && typeof logs === "object") {
            const details = Object.entries(logs)
                .map(([key, value]) => `${key}: ${value}`)
                .join(" | ");
            message += ` — ${details}`;
        }
    }
    else if (typeof payload === "string") {
        message = payload;
    }
    else if (payload?.message) {
        message = payload.message;
        level = payload.level || "";
    }
    else {
        message = JSON.stringify(payload);
    }
    if (level) {
        line.classList.add(level);
    }
    line.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logBody.appendChild(line);
    logBody.scrollTop = logBody.scrollHeight;
}

document.addEventListener("DOMContentLoaded", () => {

    const form       = document.getElementById("dynamic-form");
    const modeSelect = document.getElementById("mode");
    const fieldsRoot = document.getElementById("dynamic-fields");


    async function loadMode() {
        fieldsRoot.innerHTML = "";
        const mode = modeSelect.value;

        if (mode === "train") {
            await loadTrain(form, fieldsRoot);
        } else if (mode === "inference") {
            await loadInference(form, fieldsRoot);
        } else if (mode === "bulk_inference") {
            await loadBulkInference(form, fieldsRoot);
        } else if (mode === "prepare_dataset") {
            loadPrepareDataset(form, fieldsRoot);
        } else {
            await loadInference(form, fieldsRoot);
        }
    }


    form.addEventListener("submit", (e) => {
        e.preventDefault();
        fetch(form.action, {
            method: "POST",
            body: new FormData(form),
            credentials: "include"
        });
    });


    loadMode();
    modeSelect.addEventListener("change", loadMode);

});

document.querySelector("h1").addEventListener("click", () => {
    document.querySelector(".container").classList.toggle("collapsed");
});



// ============================
// Socket.IO Event Listeners
// ============================

// socketio.on('error', data => {
//     const output_container = document.getElementById('output-container');
//     output_container.innerHTML = '';

//     const status = document.createElement('p');
//     statusText.textContent = data.error;
// });



// // Bulk inference result
// socketio.on('bulk_inference_result', data => {
//     const output_container = document.getElementById('output-container');
//     output_container.innerHTML = '';

//     const status = data?.progress?.status;
//     const statusText = document.createElement('p');
//     statusText.textContent = status;

//     output_container.appendChild(statusText);
// });

// // Training progress
// socketio.on('train_progress', data => {
//     const output_container = document.getElementById('output-container');
//     output_container.innerHTML = '';

//     const status = data?.progress?.status || 'No status';
//     const result = data?.progress?.result || {};

//     const statusP = document.createElement('p');
//     statusP.textContent = `Status: ${status}`;
//     output_container.appendChild(statusP);

//     const resultP = document.createElement('p');
//     resultP.textContent = `Result: ${JSON.stringify(result)}`;
//     output_container.appendChild(resultP);
// });

// // Training results
// socketio.on('train_results', data => {
//     const output_container = document.getElementById('output-container');
//     output_container.innerHTML = '';

//     const status = data?.progress?.status;
//     const resultDir = data?.progress?.result?.result_dir;

//     const statusText = document.createElement('p');
//     statusText.textContent = status || 'No status provided.';

//     const result_dir = document.createElement('p');
//     result_dir.textContent = resultDir || 'No result dir.';

//     output_container.appendChild(statusText);
//     output_container.appendChild(result_dir);
// });
