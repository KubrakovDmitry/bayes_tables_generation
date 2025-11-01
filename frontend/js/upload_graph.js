async function upload_graph(e) {
    e.preventDefault();

    const graphInput = document.getElementById("graph_file");
    const graphFile = graphInput.files[0];
    const propabilityInput = document.getElementById("propability_file");
    const propabilityFile = propabilityInput.files[0];

    if (!graphFile || !propabilityFile) {
        alert("Пожалуйста, выберите два файла");
        return;
    }

    const formData = new FormData();
    formData.append("graph_file", graphFile);
    formData.append("propability_file", propabilityFile);

    try {
        const response = await fetch("upload-graph-and-propabilities/", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            alert(result.message);
        } else {
            alert("Ошибка " + (result.error));
        }
    } catch (error) {
        alert("Ошибка сети: " + error.message);
    }
}


document.getElementById("graph_form").addEventListener("submit", upload_graph);