// ============================
// Configuration
// ============================
// Change this once if your server IP/Port changes
const HOST = "http://10.4.71.86:5000";

// Connect to Socket.IO server
const socketio = io(HOST);

// ============================
// Socket.IO Event Listeners
// ============================

socketio.on('error', data => {
    const output_container = document.getElementById('output-container');
    output_container.innerHTML = '';

    const status = document.createElement('p');
    statusText.textContent = data.error;
});

// Single image inference result
socketio.on('Single_inference_result', data => {
    const output_container = document.getElementById('output-container');
    output_container.innerHTML = '';

    const result_url = data?.progress?.result?.result_url;

    if (result_url) {
        const img = document.createElement('img');
        img.src = result_url + '?' + new Date().getTime(); // cache-busting
        img.alt = 'Inference Result';
        output_container.appendChild(img);
    } else {
        output_container.innerHTML = '<p>Image not available</p>';
        console.warn('result_url is undefined:', data);
    }
});

// Bulk inference result
socketio.on('bulk_inference_result', data => {
    const output_container = document.getElementById('output-container');
    output_container.innerHTML = '';

    const status = data?.progress?.status;
    const statusText = document.createElement('p');
    statusText.textContent = status;

    output_container.appendChild(statusText);
});

// Training progress
socketio.on('train_progress', data => {
    const output_container = document.getElementById('output-container');
    output_container.innerHTML = '';

    const status = data?.progress?.status || 'No status';
    const result = data?.progress?.result || {};

    const statusP = document.createElement('p');
    statusP.textContent = `Status: ${status}`;
    output_container.appendChild(statusP);

    const resultP = document.createElement('p');
    resultP.textContent = `Result: ${JSON.stringify(result)}`;
    output_container.appendChild(resultP);
});

// Training results
socketio.on('train_results', data => {
    const output_container = document.getElementById('output-container');
    output_container.innerHTML = '';

    const status = data?.progress?.status;
    const resultDir = data?.progress?.result?.result_dir;

    const statusText = document.createElement('p');
    statusText.textContent = status || 'No status provided.';

    const result_dir = document.createElement('p');
    result_dir.textContent = resultDir || 'No result dir.';

    output_container.appendChild(statusText);
    output_container.appendChild(result_dir);
});

// ============================
// Form Handling
// ============================
// Toggle forms based on dropdown
document.addEventListener("DOMContentLoaded", function () {
    const modeSelect = document.getElementById("mode");
    const trainForm = document.querySelector('form[action="/ai_train"]');
    const inferenceForm = document.querySelector('form[action="/ai_inference"]');
    const prepareDatasetForm = document.querySelector('form[action="/ai_prepare_dataset"]');
    const bulkinferenceForm = document.querySelector('form[action="/predict"]');

    const trainModelSelect = trainForm.querySelector('select[name="models_list"]');
    const dynamicContainer = document.getElementById("dynamic-train-params");
    let lastLoadedModel = null;

    trainModelSelect.addEventListener("change", async () => {
        const model = trainModelSelect.value;
        if (!model || model === lastLoadedModel) return;
        lastLoadedModel = model;

        dynamicContainer.innerHTML = `<p style="font-size:14px;color:#777">Loading...</p>`;

        try {
            const response = await fetch(`${HOST}/get_train_params/${model}`);
            const params = await response.json();
            dynamicContainer.innerHTML = "";

            params.forEach(param => {
                let inputField = "";

                switch (param.type) {
                    case "select":
                        inputField = `
                            <select name="${param.name}" class="form-input">
                                ${param.choices.map(c => `<option value="${c}">${c}</option>`).join("")}
                            </select>`;
                        break;

                    case "text":
                        inputField = `
                            <input name="${param.name}" class="form-input" type="text" 
                                value="${param.default ?? ''}" placeholder="${param.label}">`;
                        break;

                    case "int":
                        inputField = `
                            <input name="${param.name}" class="form-input" type="number"
                                value="${param.default ?? ''}" min="${param.min ?? ''}" max="${param.max ?? ''}"
                                step="1">`;
                        break;

                    case "float":
                        inputField = `
                            <input name="${param.name}" class="form-input" type="number"
                                value="${param.default ?? ''}" step="0.01">`;
                        break;

                    case "range":
                        inputField = `
                            <input name="${param.name}" class="form-input" type="range"
                                value="${param.default ?? ''}" min="${param.min ?? ''}" max="${param.max ?? ''}">`;
                        break;

                    default:
                        inputField = `<input name="${param.name}" class="form-input" type="text">`;
                }

                dynamicContainer.innerHTML += `
                    <div class="form-group">
                        <label class="form-label">${param.label}</label>
                        ${inputField}
                    </div>`;
            });
        } catch (err) {
            dynamicContainer.innerHTML = `<p style="color:red">Error loading parameters</p>`;
        }
    });

    // Load default if pre-selected
    if (trainModelSelect.value) {
        trainModelSelect.dispatchEvent(new Event("change"));
    }

    function toggleForms() {
        const selectedMode = modeSelect.value;
        trainForm.style.display = inferenceForm.style.display = prepareDatasetForm.style.display = bulkinferenceForm.style.display = "none";
        document.querySelector(`form[action="/${selectedMode === "train" ? "ai_train" : selectedMode === "inference" ? "ai_inference" : selectedMode === "prepare_dataset" ? "ai_prepare_dataset" : "predict"}"]`).style.display = "block";
    }

    toggleForms();
    modeSelect.addEventListener("change", toggleForms);

    document.querySelectorAll("form").forEach(form => {
        form.addEventListener("submit", e => {
            e.preventDefault();
            const formData = new FormData(form);
            const actionUrl = form.action.startsWith("http") ? form.action : HOST + form.action;
            fetch(actionUrl, { method: 'POST', body: formData });
        });
    });
});

// document.addEventListener("DOMContentLoaded", function () {
//     const modeSelect = document.getElementById("mode");
//     const trainForm = document.querySelector('form[action="/ai_train"]');
//     const inferenceForm = document.querySelector('form[action="/ai_inference"]');
//     const prepareDatasetForm = document.querySelector('form[action="/ai_prepare_dataset"]');
//     const bulkinferenceForm = document.querySelector('form[action="/predict"]');

//     // Toggle forms based on dropdown
//     function toggleForms() {
//         const selectedMode = modeSelect.value;

//         trainForm.style.display = "none";
//         inferenceForm.style.display = "none";
//         prepareDatasetForm.style.display = "none";
//         bulkinferenceForm.style.display = "none";

//         if (selectedMode === "train") {
//             trainForm.style.display = "block";
//         } else if (selectedMode === "inference") {
//             inferenceForm.style.display = "block";
//         } else if (selectedMode === "prepare_dataset") {
//             prepareDatasetForm.style.display = "block";
//         } else if (selectedMode === "bulk_inference") {
//             bulkinferenceForm.style.display = "block";
//         }
//     }

//     toggleForms(); // Initial call
//     modeSelect.addEventListener("change", toggleForms);

//     // Intercept all form submissions and send them to HOST
//     const allForms = document.querySelectorAll("form");
//     allForms.forEach(form => {
//         form.addEventListener("submit", function (event) {
//             event.preventDefault();
//             const formData = new FormData(form);

//             // Use absolute URL if action is relative
//             let actionUrl = form.action.startsWith("http") ? form.action : HOST + form.action;

//             fetch(actionUrl, {
//                 method: 'POST',
//                 body: formData
//             })
//             .then(response => {
//                 if (!response.ok) {
//                     console.error(`Request failed: ${response.status}`);
//                 }
//             })
//             .catch(err => console.error("Fetch error:", err));
//         });
//     });
// });
