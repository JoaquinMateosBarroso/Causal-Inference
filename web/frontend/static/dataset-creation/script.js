async function callDatasetCreation() {
    const loading_container = document.getElementById('loading-container');
    loading_container.style.display = 'flex';
    loading_container.scrollIntoView({ behavior: 'smooth' });
    document.getElementById('obtain-causalities-button').disabled = true;
    
    function extractFormValuesToJson(formId) {
        const form = document.getElementById(formId);
        if (!form)
            return null;

        const formData = new FormData(form);
        const json = {};
        formData.forEach((value, key) => {
            type = form.elements[key].type;
            json[key] = (type !== 'number') ? value : parseFloat(value);
        });
        return json;
    }
    const dataset_parameters = extractFormValuesToJson('dataset-params-container');

    const formData = new FormData();
    formData.append('dataset_parameters_str', JSON.stringify(dataset_parameters));
    
    // Get the current URL path
    const url = window.location.pathname.split('/').slice(0, -1).join('/');

    const response = await fetch(url, {
        method: 'PUT',
        body: formData
    })


    if (!response.ok) {
        console.error("Failed to fetch files");
        return;
    }

    const data = await response.json();
    const zip = new JSZip();

    // Process each file separately
    data.files.forEach(file => {
        const binaryData = atob(file.content); // Decode Base64
        const arrayBuffer = new Uint8Array(binaryData.length);
        
        for (let i = 0; i < binaryData.length; i++) {
            arrayBuffer[i] = binaryData.charCodeAt(i);
        }
        
        zip.file(file.filename, arrayBuffer, { binary: true });
    });

    // Generate ZIP and trigger download
    zip.generateAsync({ type: "blob" }).then(function(content) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(content);
        a.download = "datasets.zip";
        a.click();
    });

    document.getElementById('loading-container').style.display = 'none';
    document.getElementById('obtain-causalities-button').disabled = false;
}

