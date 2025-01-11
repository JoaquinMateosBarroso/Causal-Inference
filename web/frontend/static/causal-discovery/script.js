let chosenDatasetFile = null;

function loadCSV(file) {
    if (file) {
        chosenDatasetFile = file;

        const reader = new FileReader();
        reader.onload = function (e) {
            const csvData = e.target.result;
            const parsedData = d3.csvParse(csvData);
            displayColumnNames(parsedData.columns);
        };
        reader.readAsText(file);

        document.getElementById('csvFileInput').disabled = true;
        document.getElementById('toyDatasetChosal').disabled = true;
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

document.getElementById('csvFileInput').addEventListener('change', (event) => {
    const file = event.target.files[0];
    loadCSV(file);    
});


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


function callBasicPC() {
    document.getElementById('loading-container').style.visibility = 'visible';
    document.getElementById('obtain-causalities-button').disabled = true;

    const formData = new FormData();
    formData.append('datasetFile', chosenDatasetFile);
    
    function getFeaturesFromElement(element) {
        return Array.from(element.children)
            .map(column => column.id)
            .slice(1); // remove the first element which is the column name
    }
    const defaultFeatures =  getFeaturesFromElement(document.getElementById('default-column'));
    const exogeneousFeatures = getFeaturesFromElement(document.getElementById('exogeneous-column'));
    const endogeneousFeatures = getFeaturesFromElement(document.getElementById('endogeneous-column'));
    
    let urlParams = '';
    urlParams += `defaultFeatures=${defaultFeatures.toString()}`;
    urlParams += `&exogeneousFeatures=${exogeneousFeatures.toString()}`;
    urlParams += `&endogeneousFeatures=${endogeneousFeatures.toString()}`;
    
    
    const baseUrl = '/basic-pc';
    const url = `${baseUrl}?${urlParams}`;

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
            console.log(data);
            drawGraph(data.graph);
            document.getElementById('loading-container').style.visibility = 'hidden';
            document.getElementById('graph-container').style.visibility = 'visible';
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
        const visData = {
            nodes: new vis.DataSet(nodes),
            edges: new vis.DataSet(edges)
        };
        const options = {
            edges: {
                arrows: {
                    to: true // Show arrows indicating the direction of edges
                }
            }
        };

        // Initialize the network
        const network = new vis.Network(container, visData, options);

        // Scroll to the graph
        container.scrollIntoView({ behavior: 'smooth' });
    }
}

function callCausalDiscovery_TimeSeries() {
    document.getElementById('loading-container').style.visibility = 'visible';
    document.getElementById('obtain-causalities-button').disabled = true;

    const formData = new FormData();
    formData.append('datasetFile', chosenDatasetFile);

    // The algorithm name is the last element in the URL
    const algorithm = window.location.pathname.split('/').at(-1);

    const baseUrl = '/causal-discovery-ts';
    const url = `${baseUrl}/${algorithm}`;

    fetch(url, {
        method: 'PUT',
        body: formData
    })
        .then(response => {
            console.log(response);
            if (!response.ok)
                throw new Error('Network response was not ok');

            return response.json();
        })
        .then(data => {
            drawGraphImage(data.graph_image);
            document.getElementById('loading-container').style.visibility = 'hidden';
            document.getElementById('graph-container').style.visibility = 'visible';
        })
    
    async function drawGraphImage(graph_image) {
        console.log('graph_image', graph_image);
        const container = document.getElementById('graph-container');
        const img = document.createElement('img');
        img.src = graph_image;
        container.appendChild(img);

        // Scroll to the graph
        container.scrollIntoView({ behavior: 'smooth' });
    }
}