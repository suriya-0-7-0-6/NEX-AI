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

const mode = document.getElementById('mode');
mode.addEventListener('change', function() {
    const uploadInput = document.getElementsByClassName('upload-input-group')[0];
    const problemIdInput = document.getElementsByClassName('problem-id-group')[0];
    const modelsInput = document.getElementsByClassName('models-group')[0];

    if (this.value === 'infer') {
        problemIdInput.classList.remove('hidden');
        uploadInput.classList.remove('hidden');
    } else {
        problemIdInput.classList.add('hidden');
        uploadInput.classList.add('hidden');
    }

    if (this.value === 'train') {
        problemIdInput.classList.add('hidden');
        modelsInput.classList.remove('hidden');
    } else {
        problemIdInput.classList.remove('hidden');
        modelsInput.classList.add('hidden');
    }
});

const ai_form = document.querySelector('.inference-form');
mode.addEventListener('change', function() {
    if (this.value === 'train') {
        ai_form.action = '/train_model';
    } else if (this.value === 'infer') {
        ai_form.action = '/ai_inference';
    }
});

const aiForm = document.querySelector('.inference-form');

aiForm.addEventListener('submit', function(event) {
    event.preventDefault();  
    const formData = new FormData(aiForm);
    fetch(aiForm.action, {
        method: 'POST',
        body: formData
    })
});