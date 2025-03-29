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
        .then(response => {
            if (!response.ok)
                throw new Error('Network response was not ok');dataset_parameters_str
        })

    const blob = await response.blob();
    const zip = new JSZip();
    zip.file("data.csv", blob, { binary: true });

    zip.generateAsync({ type: "blob" }).then(function(content) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(content);
        a.download = "files.zip";
        a.click();
    });
}
