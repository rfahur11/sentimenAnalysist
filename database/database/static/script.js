document.addEventListener("DOMContentLoaded", function () {
    window.currentPage = 1;
    window.pageSize = 10;
    window.predictions = [];
});

function predictText() {
    const text = document.getElementById("inputText").value;
    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("resultText").innerText = JSON.stringify(data, null, 2);
    })
    .catch(error => console.error("Error:", error));
}

function predictCSV() {
    const file = document.getElementById("csvFile").files[0];
    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict_csv", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        window.predictions = data.predictions;
        renderTable(1);
        renderChart();
    })
    .catch(error => console.error("Error:", error));
}

function renderTable(page) {
    const tableBody = document.querySelector("#resultsTable tbody");
    tableBody.innerHTML = "";
    
    let start = (page - 1) * window.pageSize;
    let end = start + window.pageSize;
    let paginatedData = window.predictions.slice(start, end);

    paginatedData.forEach(item => {
        let row = `<tr>
            <td>${item.comment}</td>
            <td>${item.price}</td>
            <td>${item.packaging}</td>
            <td>${item.product}</td>
            <td>${item.aroma}</td>
        </tr>`;
        tableBody.innerHTML += row;
    });

    renderPagination(page);
}

function renderPagination(page) {
    const pagination = document.getElementById("pagination");
    pagination.innerHTML = "";

    let totalPages = Math.ceil(window.predictions.length / window.pageSize);
    let maxPagesToShow = 5;
    let startPage = Math.max(1, page - Math.floor(maxPagesToShow / 2));
    let endPage = Math.min(totalPages, startPage + maxPagesToShow - 1);

    if (endPage - startPage < maxPagesToShow - 1) {
        startPage = Math.max(1, endPage - maxPagesToShow + 1);
    }

    // Tombol ke halaman pertama
    if (page > 1) {
        let firstBtn = document.createElement("button");
        firstBtn.innerText = "1";
        firstBtn.onclick = () => renderTable(1);
        pagination.appendChild(firstBtn);
    }

    // Tampilkan nomor halaman yang tersedia
    for (let i = startPage; i <= endPage; i++) {
        let btn = document.createElement("button");
        btn.innerText = i;
        btn.onclick = () => renderTable(i);
        if (i === page) btn.style.fontWeight = "bold";
        pagination.appendChild(btn);
    }

    // Tombol "Selanjutnya"
    if (page < totalPages) {
        let nextBtn = document.createElement("button");
        nextBtn.innerText = "Selanjutnya";
        nextBtn.onclick = () => renderTable(page + 1);
        pagination.appendChild(nextBtn);
    }

    // Tombol ke halaman terakhir
    if (page < totalPages) {
        let lastBtn = document.createElement("button");
        lastBtn.innerText = totalPages;
        lastBtn.onclick = () => renderTable(totalPages);
        pagination.appendChild(lastBtn);
    }
}

function renderChart() {
    let sentimentCounts = { price: { Positif: 0, Netral: 0, Negatif: 0 },
                            packaging: { Positif: 0, Netral: 0, Negatif: 0 },
                            product: { Positif: 0, Netral: 0, Negatif: 0 },
                            aroma: { Positif: 0, Netral: 0, Negatif: 0 } };

    window.predictions.forEach(item => {
        sentimentCounts.price[item.price]++;
        sentimentCounts.packaging[item.packaging]++;
        sentimentCounts.product[item.product]++;
        sentimentCounts.aroma[item.aroma]++;
    });

    const ctx = document.getElementById("sentimentChart").getContext("2d");
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Positif", "Netral", "Negatif"],
            datasets: [
                { label: "Price", data: Object.values(sentimentCounts.price), backgroundColor: "blue" },
                { label: "Packaging", data: Object.values(sentimentCounts.packaging), backgroundColor: "green" },
                { label: "Product", data: Object.values(sentimentCounts.product), backgroundColor: "red" },
                { label: "Aroma", data: Object.values(sentimentCounts.aroma), backgroundColor: "purple" }
            ]
        },
        options: { responsive: true }
    });
}
