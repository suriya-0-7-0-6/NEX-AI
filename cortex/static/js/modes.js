// modes.js

import { HOST } from "./config.js";
import { fetchJSON, setHTML } from "./utils.js";
import { renderFields } from "./fields.js";

export async function loadTrain(form, fieldsRoot) {

    setHTML(fieldsRoot, "Loading models...");

    const data = await fetchJSON(`${HOST}/get_all_model_archs`);
    if (!data) return;

    setHTML(fieldsRoot, `
        <div class="form-group">
            <label class="form-label">Model</label>
            <select id="models_list" name="models_list" class="form-input">
                <option value="">-- Select Model --</option>
                ${data.all_model_archs.map(m =>
                    `<option value="${m}">${m}</option>`
                ).join("")}
            </select>
        </div>

        <div id="train-params"></div>
    `);

    const modelSelect = document.getElementById("models_list");
    const paramsRoot  = document.getElementById("train-params");

    modelSelect.addEventListener("change", async () => {
        if (!modelSelect.value) {
            paramsRoot.innerHTML = "";
            return;
        }

        paramsRoot.innerHTML = "Loading training fields...";

        const params = await fetchJSON(
            `${HOST}/get_train_form_fields/${modelSelect.value}`
        );

        renderFields(params, paramsRoot);
    });

    form.action = `${HOST}/ai_train`;
}


export async function loadInference(form, fieldsRoot) {

    setHTML(fieldsRoot, "Loading problems...");

    const data = await fetchJSON(`${HOST}/get_all_problem_ids`);
    if (!data) return;

    setHTML(fieldsRoot, `
        <div class="form-group">
            <label class="form-label">Problem ID</label>
            <select id="problem_id" name="problem_id" class="form-input">
                <option value="">-- Select Problem --</option>
                ${data.problem_ids.map(p =>
                    `<option value="${p}">${p}</option>`
                ).join("")}
            </select>
        </div>

        <div id="inference-params"></div>
    `);

    const problemSelect = document.getElementById("problem_id");
    const paramsRoot   = document.getElementById("inference-params");

    problemSelect.addEventListener("change", async () => {
        if (!problemSelect.value) {
            paramsRoot.innerHTML = "";
            return;
        }

        paramsRoot.innerHTML = "Loading inference fields...";

        const params = await fetchJSON(
            `${HOST}/get_inference_form_fields/${problemSelect.value}`
        );

        renderFields(params, paramsRoot);
    });

    form.action = `${HOST}/ai_inference`;
}


export async function loadBulkInference(form, fieldsRoot) {
    await loadInference(form, fieldsRoot);
    form.action = `${HOST}/predict`;
}


export async function loadPrepareDataset(form, fieldsRoot) {
    setHTML(fieldsRoot, "Loading models...");

    const data = await fetchJSON(`${HOST}/get_all_model_archs`);
    if (!data) return;

    setHTML(fieldsRoot, `
        <div class="form-group">
            <label class="form-label">Model</label>
            <select id="models_list" name="models_list" class="form-input">
                <option value="">-- Select Model --</option>
                ${data.all_model_archs.map(m =>
                    `<option value="${m}">${m}</option>`
                ).join("")}
            </select>
        </div>

        <div id="prepare-dataset-params"></div>
    `);

    const modelSelect = document.getElementById("models_list");
    const paramsRoot  = document.getElementById("prepare-dataset-params");

    modelSelect.addEventListener("change", async () => {
        if (!modelSelect.value) {
            paramsRoot.innerHTML = "";
            return;
        }

        paramsRoot.innerHTML = "Loading required fields...";

        const params = await fetchJSON(
            `${HOST}/get_prepare_dataset_form_fields/${modelSelect.value}`
        );

        renderFields(params, paramsRoot);
    });


    form.action = `${HOST}/ai_prepare_dataset`;
}
