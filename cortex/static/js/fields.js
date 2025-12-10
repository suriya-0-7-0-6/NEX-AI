// fields.js

export function buildField(parameter) {
    let inputHTML = "";

    switch (parameter.type) {

        case "select":
            inputHTML = `
                <select name="${parameter.name}" class="form-input">
                    ${parameter.choices.map(c =>
                        `<option value="${c}">${c}</option>`
                    ).join("")}
                </select>
            `;
            break;

        case "int":
            inputHTML = `
                <input type="number"
                       step="1"
                       name="${parameter.name}"
                       value="${parameter.default || ""}"
                       class="form-input">
            `;
            break;

        case "float":
            inputHTML = `
                <input type="number"
                       step="0.01"
                       name="${parameter.name}"
                       value="${parameter.default || ""}"
                       class="form-input">
            `;
            break;

        case "range":
            inputHTML = `
                <input type="range"
                       name="${parameter.name}"
                       min="${parameter.min || ""}"
                       max="${parameter.max || ""}"
                       value="${parameter.default || ""}"
                       class="form-input">
            `;
            break;

        case "file":
            inputHTML = `
                <input type="file"
                       name="${parameter.name}"
                       accept="${parameter.accept || ""}"
                       class="form-input">
            `;
            break;

        default:
            inputHTML = `
                <input type="text"
                       name="${parameter.name}"
                       placeholder="${parameter.label}"
                       value="${parameter.default || ""}"
                       class="form-input">
            `;
    }

    return `
        <div class="form-group">
            <label class="form-label">${parameter.label}</label>
            ${inputHTML}
        </div>
    `;
}


export function renderFields(fields, target) {
    if (!fields || fields.length === 0) {
        target.innerHTML =
            "<p style='color:#777'>No fields defined.</p>";
        return;
    }

    let html = "";
    fields.forEach(p => html += buildField(p));
    target.innerHTML = html;
}
