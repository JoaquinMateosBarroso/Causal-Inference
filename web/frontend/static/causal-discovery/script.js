let chosenDatasetFile = null;

function loadCSV() {
    file = document.getElementById('load-csv-button').files[0];

    if (file) {
        chosenDatasetFile = file;

        const reader = new FileReader();
        reader.onload = function (e) {
            const csvData = e.target.result;
            const parsedData = d3.csvParse(csvData);
            displayColumnNames(parsedData.columns);
        };
        reader.readAsText(file);

        document.getElementById('load-csv-button').disabled = true;
        Array.from(document.getElementsByClassName('calling-button')).forEach( (element) => {
            console.log(element);
            element.style.display = 'inline-block';})
    }
}


function chooseToyDataset(datasetName) {
    fetch('/frontend/static/toy-datasets/' + datasetName)
        .then(response => response.blob())
        .then(loadCSV);
}

function displayColumnNames(columns) {
    const columnContainer = document.getElementById('default-column');

    columns.forEach(column => {
        let columnBox = document.createElement('div');
        columnBox.className = 'column-item';
        columnBox.id = column;
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


function callAlgorithm() {
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
    
    function getVariablesFromElement(element) {
        return Array.from(element.children)
            .map(column => column.id)
            .slice(1); // remove the first element which is the column name
    }

    const defaultVariables =  getVariablesFromElement(document.getElementById('default-column'));
    
    let urlParams = '';
    urlParams += `defaultVariables=${defaultVariables.toString()}`;
    
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
                if (!data.graph) {
                    alert('No graph found. Please try again.');
                    return;
                }
                drawGraph(data.graph);
            }
            else {
                const container = document.getElementById('graph-container');
                const img = document.createElement('img');
                img.src = image;
                img.style.width = "100%";
                container.appendChild(img);
            }
            
            graph_container = document.getElementById('graph-container');
            graph_container.style.display = 'flex';
            graph_container.scrollIntoView({ behavior: 'smooth' });

            document.getElementById('loading-container').style.display = 'none';
            document.getElementById('obtain-causalities-button').disabled = false;
        })
    
        async function drawGraph(graph) {
            // Transform data into nodes and edges
            const nodes = [];
            const edges = [];
    
            Object.keys(graph).forEach((key) => {
                nodes.push({ id: key, label: key });
                graph[key].forEach((target) => {
                    edges.push({ from: key, to: target });
                });
            });
    
            // Create a network
            const container = document.getElementById('graph-container');
            
            // Scroll to the graph
            container.scrollIntoView({ behavior: 'smooth' });
        }
}
