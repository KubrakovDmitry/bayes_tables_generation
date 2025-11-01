async function upload_data(e) {
    e.preventDefault();

    const fileInput = document.getElementById("data");
    const file = fileInput.files[0];

    if (!file) {
        alert("Пожалуйста, выберите файл");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("upload-data/", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            alert(result.message);
        } else {
            alert("Ошибка: " + (result.error));
        }
    } catch (error) {
        alert("Ошибка сети: " + error.message);
    }
}

document.getElementById("data_form").addEventListener("submit", upload_data);
