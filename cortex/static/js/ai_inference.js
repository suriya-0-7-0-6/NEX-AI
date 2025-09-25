const socketio = io('http://localhost:5000')

socketio.on('result', data => {
    output_container = document.getElementById('output-container');
    output_container.innerHTML = '';
    img = document.createElement('img');
    img.src = data.result_url + '?' + new Date().getTime();
    img.alt = 'Inference Result';
    // img.style.width = '100%';
    // img.style.height = 'auto';

    output_container.appendChild(img);
})

document.addEventListener("DOMContentLoaded", function () {
    const modeSelect = document.getElementById("mode");
    const trainForm = document.querySelector('form[action="/ai_train"]');
    const inferenceForm = document.querySelector('form[action="/ai_inference"]');
    const prepareDatasetForm = document.querySelector('form[action="/ai_prepare_dataset"]');

    function toggleForms() {
        const selectedMode = modeSelect.value;

        trainForm.style.display = "none";
        inferenceForm.style.display = "none";
        prepareDatasetForm.style.display = "none";

        if (selectedMode === "train") {
            trainForm.style.display = "block";
        } else if (selectedMode === "inference") {
            inferenceForm.style.display = "block";
        } else if (selectedMode === "prepare_dataset") {
            prepareDatasetForm.style.display = "block";
        }
    }

    toggleForms();

    modeSelect.addEventListener("change", toggleForms);
});
