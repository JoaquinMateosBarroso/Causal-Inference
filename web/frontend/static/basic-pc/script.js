document.getElementById('csvFileInput').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const csvData = e.target.result;
            const parsedData = d3.csvParse(csvData);
            displayColumnNames(parsedData.columns);
        };
        reader.readAsText(file);
    }
});

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
        callingButtons[i].style.display = 'block';
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

function callBasicPC() {
    

}