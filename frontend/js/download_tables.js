async function download_tables(e) {
    e.preventDefault();

    try {
        const response = await fetch("generate-bayes-table/");

        if (!response.ok) {
            const errorText = await response.text();
            alert("Ошибка при генерации файла: " + errorText);
            return;
        }

        const blob = await response.blob();

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "bayes_report.xlsx";
        document.body.appendChild(a);
        a.click();

        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

    } catch (error) {
        alert("Ошибка сети: " + error.message);
    }
}

document.getElementById("table_form").addEventListener("submit", download_tables);
