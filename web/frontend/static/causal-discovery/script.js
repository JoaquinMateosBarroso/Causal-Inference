let chosenDatasetFile = null;

function loadCSV(file=null) {
    if (file === null)
        file = document.getElementById('load-csv-button').files[0];

    chosenDatasetFile = file;

    const reader = new FileReader();
    reader.onload = function (e) {
        const csvData = e.target.result;
        const parsedData = d3.csvParse(csvData);
        displayColumnNames(parsedData.columns);
    };
    reader.readAsText(file);

    document.getElementById('load-csv-button').disabled = true;
    // Show the calling buttons
    Array.from(document.getElementsByClassName('calling-button')).forEach( (element) => {
        element.style.display = 'inline-block';})
}

function loadGroupCSV(file=null) {
    loadCSV(file);
    document.getElementById('group-adder').style.display = 'flex';
}


function loadZIP() {
    file = document.getElementById('load-zip-button').files[0];
    chosenDatasetFile = file;
    if (file) {
        document.getElementById('load-zip-button').disabled = true;
        // Show the calling buttons
        Array.from(document.getElementsByClassName('calling-button')).forEach( (element) => {
            element.style.display = 'inline-block';})
    }
}

function displayColumnNames(columns) {
    const columnContainer = document.getElementById('group-column-1');

    columns.forEach(column => {
        let columnBox = document.createElement('div');
        columnBox.className = 'column-item';
        columnBox.id = `node-${column}`;
        columnBox.textContent = column;
        columnBox.setAttribute('draggable', true);
        columnBox.setAttribute('ondragstart', 'drag(event)');

        columnContainer.appendChild(columnBox);
    });

    let callingButtons = document.getElementsByClassName('calling-button');
    
    for (let i=0; i<callingButtons.length; i++) {
        callingButtons[i].style.visibility = 'visible';
    }
}



function allowDrop(event) {
    event.preventDefault();
}

function drag(event) {
    event.dataTransfer.setData("text", event.target.id);
}

function drop(event, columnId) {
    event.preventDefault();
    let data = event.dataTransfer.getData("text");
    let draggedElement = document.getElementById(data);

    target = document.getElementById(columnId);
    target.appendChild(draggedElement);
}

function addGroup() {
    const columnContainer = document.getElementById('group-columns');

    // Create the new group container
    const groupContainer = document.createElement('div');
    groupContainer.className = 'column';
    const groupIndex = columnContainer.children.length + 1;
    groupContainer.id = `group-${groupIndex}`;
    groupContainer.setAttribute('ondrop', 'drop(event, this.id)');
    groupContainer.setAttribute('ondragover', 'allowDrop(event)');

    columnContainer.appendChild(groupContainer);

    const columnTitle = document.createElement('h2');
    columnTitle.textContent = `Group ${groupIndex}`;

    groupContainer.appendChild(columnTitle);
}

async function callAlgorithm() {
    loading_container = document.getElementById('loading-container');
    loading_container.style.display = 'flex';
    loading_container.scrollIntoView({ behavior: 'smooth' });
    document.getElementById('obtain-causalities-button').disabled = true;

    algorithm_name = document.getElementById('algorithm-options').value;
    
    function extractFormValuesToJson(formId) {
        const form = document.getElementById(formId);
        if (!form) {
            return null; // Or handle the error as needed
        }
        const formData = new FormData(form);
        const json = {};
        formData.forEach((value, key) => {
            type = form.elements[key].type;
            json[key] = (type !== 'number') ? value : parseFloat(value);
        });
        return json;
    }
    algorithm_parameters = extractFormValuesToJson('algorithm-params-container');

    const formData = new FormData();
    formData.append('algorithm_parameters_str', JSON.stringify(algorithm_parameters));
    formData.append('datasetFile', chosenDatasetFile);

    const groupColumns = document.getElementById('group-columns').children;
    const groupVariables = Array.from(groupColumns).map(group => {
        return Array.from(group.children)
            .filter(element => element.tagName !== 'H2') // Exclude the group title
            .map(column => column.id.split('-')[1]); // Get the variable names
        });
        
    let urlParams = '';
    groupVariables.forEach((group, index) => {
        urlParams += `group-${index+1}=[${group.toString()}]`;
        if (index < groupVariables.length-1)
            urlParams += '&';
    });
    
    // Get the current URL path
    const baseUrl = window.location.pathname.split('/').slice(0, -1).join('/');
    // Add the algorithm name to the URL and append some parameters
    const url = `${baseUrl}/${algorithm_name}?${urlParams}`;

    fetch(url, {
        method: 'PUT',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            image = data.graph_image;
            if (!image) {
                alert('No graph found. Please reload page and try again.');
                return;
            }
            // Display the graph image in the container
            const container = document.getElementById('graph-container');
            const img = document.createElement('img');
            img.src = image;
            img.style.width = "100%";
            container.appendChild(img);
            
            graph_container = document.getElementById('graph-container');
            graph_container.style.display = 'flex';
            graph_container.scrollIntoView({ behavior: 'smooth' });

            document.getElementById('loading-container').style.display = 'none';
            document.getElementById('obtain-causalities-button').disabled = false;
        })
}

async function callBenchmark() {
    
    loading_container = document.getElementById('loading-container');
    loading_container.style.display = 'flex';
    loading_container.scrollIntoView({ behavior: 'smooth' });
    document.getElementById('obtain-causalities-button').disabled = true;

    
    function extractFormValuesToJson(formId) {
        const form = document.getElementById(formId);
        if (!form) {
            return null; // Or handle the error as needed
        }
        const formData = new FormData(form);
        const json = {};
        formData.forEach((value, key) => {
            type = form.elements[key].type;
            json[key] = (type !== 'number') ? value : parseFloat(value);
        });
        return json;
    }
    
    // Read parameters of all algorithms
    let algorithmCounter = 0;
    let algorithms_parameters = {};
    while (document.getElementById(`algorithm-options-${algorithmCounter}`)) {
        let current_parameters = extractFormValuesToJson(`algorithm-params-container-${algorithmCounter}`);
        algorithm_name = document.getElementById(`algorithm-options-${algorithmCounter++}`).value;
        algorithms_parameters[algorithm_name] = current_parameters;
    }

    const formData = new FormData();
    formData.append('algorithms_parameters_str', JSON.stringify(algorithms_parameters));
    formData.append('datasetFile', chosenDatasetFile);
    
    
    
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
    const graphs_container = document.getElementById('graphs-container');
    graphs_container.innerHTML = ""; // Clear previous graphs

    // Process each file separately
    data.files.forEach(file => {
        const binaryData = atob(file.content); // Decode Base64
        const arrayBuffer = new Uint8Array(binaryData.length);
        
        for (let i = 0; i < binaryData.length; i++)
            arrayBuffer[i] = binaryData.charCodeAt(i);
        
        console.log(file.filename);
        if (file.filename && (file.filename.endsWith(".png") || file.filename.endsWith(".pdf"))) {
            const img = document.createElement("img");
            if (file.filename.endsWith(".png")) {
                img.src = `data:image/png;base64,${file.content}`;
                img.style.width = "100%";
                graphs_container.appendChild(img);
            } else if (file.filename.endsWith(".pdf")) {
                const iFrame = document.createElement("iframe");
                iFrame.src = `data:application/pdf;base64,${file.content}#toolbar=0&scrollbar=0`;
                iFrame.width = "70%";
                iFrame.height = "500px";
                graphs_container.appendChild(iFrame);
            }
            graphs_container.appendChild(document.createElement("br"));
        }
        zip.file(file.filename, arrayBuffer, { binary: true });
    });
    graphs_container.style.display = 'flex';

    // Generate ZIP and trigger download
    zip.generateAsync({ type: "blob" }).then(function(content) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(content);
        a.download = "results.zip";
        a.click();
    });
    
    graphs_container.scrollIntoView({ behavior: 'smooth' });
    document.getElementById('loading-container').style.display = 'none';
    document.getElementById('obtain-causalities-button').disabled = false;
}




