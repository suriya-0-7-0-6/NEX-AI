const socketio = io('http://localhost:5000')

socketio.on('result', data => {
    output_container = document.getElementById('output-container');
    output_container.innerHTML = '';
    img = document.createElement('img');
    img.src = data.result_url + '?' + new Date().getTime();
    img.alt = 'Inference Result';
    img.style.width = '100%';
    img.style.height = 'auto';

    output_container.appendChild(img);
})